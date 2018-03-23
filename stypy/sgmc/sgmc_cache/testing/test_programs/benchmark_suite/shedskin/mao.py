
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ambient occlusion renderer
3: http://lucille.atso-net.jp/aobench/
4: 
5: Original version of AO bench was written by Syoyo Fujita. The original code(Proce55ing version) is licensed under BSD3 license. You can freely modify, port and distribute AO bench
6: '''
7: 
8: from math import sqrt, sin, cos, fabs
9: import random
10: from array import array
11: import os
12: 
13: random.seed(1)
14: 
15: WIDTH = 128
16: HEIGHT = WIDTH
17: NSUBSAMPLES = 2
18: NAO_SAMPLES = 8
19: 
20: 
21: def Relative(path):
22:     return os.path.join(os.path.dirname(__file__), path)
23: 
24: 
25: class Vector:
26:     def __init__(self, x, y, z):
27:         self.x = x
28:         self.y = y
29:         self.z = z
30: 
31: 
32: class Isect:
33:     def __init__(self):
34:         self.p = Vector(0.0, 0.0, 0.0)
35:         self.n = Vector(0.0, 0.0, 0.0)
36: 
37:     def reset(self):
38:         self.t = 1.0e+17
39:         self.hit = 0
40:         self.p.x = self.p.y = self.p.z = 0.0
41:         self.n.x = self.n.y = self.n.z = 0.0
42: 
43: 
44: class Sphere:
45:     def __init__(self, center, radius):
46:         self.center = center
47:         self.radius = radius
48: 
49: 
50: class Plane:
51:     def __init__(self, p, n):
52:         self.p = p
53:         self.n = n
54: 
55: 
56: class Ray:
57:     def __init__(self):
58:         self.org = Vector(0.0, 0.0, 0.0)
59:         self.dir = Vector(0.0, 0.0, 0.0)
60: 
61:     def reset(self, p, x, y, z):
62:         self.org.x, self.org.y, self.org.z = p.x, p.y, p.z
63:         self.dir.x, self.dir.y, self.dir.z = x, y, z
64: 
65: 
66: def vdot(v0, v1):
67:     return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
68: 
69: 
70: def vcross(c, v0, v1):
71:     c.x = v0.y * v1.z - v0.z * v1.y
72:     c.y = v0.z * v1.x - v0.x * v1.z
73:     c.z = v0.x * v1.y - v0.y * v1.x
74: 
75: 
76: def vnormalize(c):
77:     length = sqrt(vdot(c, c))
78: 
79:     if fabs(length) > 1.0e-17:
80:         c.x /= length
81:         c.y /= length
82:         c.z /= length
83: 
84: 
85: def ray_sphere_intersect(isect, ray, sphere):
86:     rsx = ray.org.x - sphere.center.x
87:     rsy = ray.org.y - sphere.center.y
88:     rsz = ray.org.z - sphere.center.z
89: 
90:     B = rsx * ray.dir.x + rsy * ray.dir.y + rsz * ray.dir.z
91:     C = rsx * rsx + rsy * rsy + rsz * rsz - sphere.radius * sphere.radius
92:     D = B * B - C
93: 
94:     if D > 0.0:
95:         t = -B - sqrt(D)
96:         if t > 0.0:
97:             if t < isect.t:
98:                 isect.t = t
99:                 isect.hit = 1
100: 
101:                 isect.p.x = ray.org.x + ray.dir.x * t
102:                 isect.p.y = ray.org.y + ray.dir.y * t
103:                 isect.p.z = ray.org.z + ray.dir.z * t
104: 
105:                 isect.n.x = isect.p.x - sphere.center.x
106:                 isect.n.y = isect.p.y - sphere.center.y
107:                 isect.n.z = isect.p.z - sphere.center.z
108: 
109:                 vnormalize(isect.n)
110: 
111: 
112: def ray_plane_intersect(isect, ray, plane):
113:     d = -vdot(plane.p, plane.n)
114:     v = vdot(ray.dir, plane.n)
115: 
116:     if abs(v) < 1.0e-17:
117:         return
118: 
119:     t = -(vdot(ray.org, plane.n) + d) / v
120: 
121:     if t > 0.0:
122:         if t < isect.t:
123:             isect.t = t
124:             isect.hit = 1
125: 
126:             isect.p.x = ray.org.x + ray.dir.x * t
127:             isect.p.y = ray.org.y + ray.dir.y * t
128:             isect.p.z = ray.org.z + ray.dir.z * t
129: 
130:             isect.n.x = plane.n.x
131:             isect.n.y = plane.n.y
132:             isect.n.z = plane.n.z
133: 
134: 
135: def ortho_basis(basis, n):
136:     basis[2] = n
137:     basis[1].x = basis[1].y = basis[1].z = 0.0
138: 
139:     if n.x < 0.6 and n.x > -0.6:
140:         basis[1].x = 1.0
141:     elif n.y < 0.6 and n.y > -0.6:
142:         basis[1].y = 1.0
143:     elif n.z < 0.6 and n.z > -0.6:
144:         basis[1].z = 1.0
145:     else:
146:         basis[1].x = 1.0
147: 
148:     vcross(basis[0], basis[1], basis[2])
149:     vnormalize(basis[0])
150: 
151:     vcross(basis[1], basis[2], basis[0])
152:     vnormalize(basis[1])
153: 
154: 
155: def ambient_occlusion(col, isect):
156:     global random_idx
157:     ntheta = NAO_SAMPLES
158:     nphi = NAO_SAMPLES
159:     eps = 0.0001
160: 
161:     p = Vector(isect.p.x + eps * isect.n.x,
162:                isect.p.y + eps * isect.n.y,
163:                isect.p.z + eps * isect.n.z)
164: 
165:     basis = [Vector(0.0, 0.0, 0.0) for x in range(3)]
166:     ortho_basis(basis, isect.n)
167: 
168:     occlusion = 0.0
169:     b0, b1, b2 = basis[0], basis[1], basis[2]
170:     isect = Isect()
171:     ray = Ray()
172: 
173:     for j in xrange(ntheta):
174:         for i in xrange(nphi):
175:             theta = sqrt(random.random())
176:             phi = 2.0 * 3.14159265358979323846 * random.random()
177: 
178:             x = cos(phi) * theta
179:             y = sin(phi) * theta
180:             z = sqrt(1.0 - theta * theta)
181: 
182:             rx = x * b0.x + y * b1.x + z * b2.x
183:             ry = x * b0.y + y * b1.y + z * b2.y
184:             rz = x * b0.z + y * b1.z + z * b2.z
185:             ray.reset(p, rx, ry, rz)
186: 
187:             isect.reset()
188: 
189:             ray_sphere_intersect(isect, ray, sphere1)
190:             ray_sphere_intersect(isect, ray, sphere2)
191:             ray_sphere_intersect(isect, ray, sphere3)
192:             ray_plane_intersect(isect, ray, plane)
193: 
194:             if isect.hit:
195:                 occlusion += 1.0
196: 
197:     occlusion = (ntheta * nphi - occlusion) / float(ntheta * nphi)
198:     col.x = col.y = col.z = occlusion
199: 
200: 
201: def clamp(f):
202:     i = int(f * 255.5)
203:     if i < 0:
204:         i = 0
205:     if i > 255:
206:         i = 255
207:     return i
208: 
209: 
210: def render(w, h, nsubsamples):
211:     img = [0] * (WIDTH * HEIGHT * 3)
212: 
213:     nsubs = float(nsubsamples)
214:     nsubs_nsubs = nsubs * nsubs
215: 
216:     v0 = Vector(0.0, 0.0, 0.0)
217:     col = Vector(0.0, 0.0, 0.0)
218:     isect = Isect()
219:     ray = Ray()
220: 
221:     for y in xrange(h):
222:         for x in xrange(w):
223:             fr = 0.0
224:             fg = 0.0
225:             fb = 0.0
226:             for v in xrange(nsubsamples):
227:                 for u in xrange(nsubsamples):
228:                     px = (x + (u / float(nsubsamples)) - (w / 2.0)) / (w / 2.0)
229:                     py = -(y + (v / float(nsubsamples)) - (h / 2.0)) / (h / 2.0)
230:                     ray.reset(v0, px, py, -1.0)
231:                     vnormalize(ray.dir)
232: 
233:                     isect.reset()
234: 
235:                     ray_sphere_intersect(isect, ray, sphere1)
236:                     ray_sphere_intersect(isect, ray, sphere2)
237:                     ray_sphere_intersect(isect, ray, sphere3)
238:                     ray_plane_intersect(isect, ray, plane)
239: 
240:                     if isect.hit:
241:                         ambient_occlusion(col, isect)
242:                         fr += col.x
243:                         fg += col.y
244:                         fb += col.z
245: 
246:             img[3 * (y * w + x) + 0] = clamp(fr / nsubs_nsubs)
247:             img[3 * (y * w + x) + 1] = clamp(fg / nsubs_nsubs)
248:             img[3 * (y * w + x) + 2] = clamp(fb / nsubs_nsubs)
249: 
250:     return img
251: 
252: 
253: def init_scene():
254:     global sphere1, sphere2, sphere3, plane
255:     sphere1 = Sphere(Vector(-2.0, 0.0, -3.5), 0.5)
256:     sphere2 = Sphere(Vector(-0.5, 0.0, -3.0), 0.5)
257:     sphere3 = Sphere(Vector(1.0, 0.0, -2.2), 0.5)
258:     plane = Plane(Vector(0.0, -0.5, 0.0), Vector(0.0, 1.0, 0.0))
259: 
260: 
261: def save_ppm(img, w, h, fname):
262:     fout = open(Relative(fname), "wb")
263:     ##    print >>fout, "P6"
264:     ##    print >>fout, "%i %i" % (w, h)
265:     ##    print >>fout, "255"
266:     array("B", img).tofile(fout)
267:     fout.close()
268: 
269: 
270: def run():
271:     init_scene()
272:     img = render(WIDTH, HEIGHT, NSUBSAMPLES)
273:     save_ppm(img, WIDTH, HEIGHT, "ao_py.ppm")
274:     return True
275: 
276: 
277: run()
278: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nambient occlusion renderer\nhttp://lucille.atso-net.jp/aobench/\n\nOriginal version of AO bench was written by Syoyo Fujita. The original code(Proce55ing version) is licensed under BSD3 license. You can freely modify, port and distribute AO bench\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from math import sqrt, sin, cos, fabs' statement (line 8)
try:
    from math import sqrt, sin, cos, fabs

except:
    sqrt = UndefinedType
    sin = UndefinedType
    cos = UndefinedType
    fabs = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'math', None, module_type_store, ['sqrt', 'sin', 'cos', 'fabs'], [sqrt, sin, cos, fabs])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import random' statement (line 9)
import random

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from array import array' statement (line 10)
try:
    from array import array

except:
    array = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'array', None, module_type_store, ['array'], [array])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import os' statement (line 11)
import os

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'os', os, module_type_store)


# Call to seed(...): (line 13)
# Processing the call arguments (line 13)
int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
# Processing the call keyword arguments (line 13)
kwargs_14 = {}
# Getting the type of 'random' (line 13)
random_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 13)
seed_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), random_11, 'seed')
# Calling seed(args, kwargs) (line 13)
seed_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), seed_12, *[int_13], **kwargs_14)


# Assigning a Num to a Name (line 15):

# Assigning a Num to a Name (line 15):
int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'int')
# Assigning a type to the variable 'WIDTH' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'WIDTH', int_16)

# Assigning a Name to a Name (line 16):

# Assigning a Name to a Name (line 16):
# Getting the type of 'WIDTH' (line 16)
WIDTH_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'WIDTH')
# Assigning a type to the variable 'HEIGHT' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'HEIGHT', WIDTH_17)

# Assigning a Num to a Name (line 17):

# Assigning a Num to a Name (line 17):
int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
# Assigning a type to the variable 'NSUBSAMPLES' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'NSUBSAMPLES', int_18)

# Assigning a Num to a Name (line 18):

# Assigning a Num to a Name (line 18):
int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'int')
# Assigning a type to the variable 'NAO_SAMPLES' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'NAO_SAMPLES', int_19)

@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 21, 0, False)
    
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

    
    # Call to join(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to dirname(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of '__file__' (line 22)
    file___26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 40), '__file__', False)
    # Processing the call keyword arguments (line 22)
    kwargs_27 = {}
    # Getting the type of 'os' (line 22)
    os_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 22)
    path_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), os_23, 'path')
    # Obtaining the member 'dirname' of a type (line 22)
    dirname_25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), path_24, 'dirname')
    # Calling dirname(args, kwargs) (line 22)
    dirname_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 22, 24), dirname_25, *[file___26], **kwargs_27)
    
    # Getting the type of 'path' (line 22)
    path_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 51), 'path', False)
    # Processing the call keyword arguments (line 22)
    kwargs_30 = {}
    # Getting the type of 'os' (line 22)
    os_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 22)
    path_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), os_20, 'path')
    # Obtaining the member 'join' of a type (line 22)
    join_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), path_21, 'join')
    # Calling join(args, kwargs) (line 22)
    join_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), join_22, *[dirname_call_result_28, path_29], **kwargs_30)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', join_call_result_31)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_32

# Assigning a type to the variable 'Relative' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'Relative', Relative)
# Declaration of the 'Vector' class

class Vector:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector.__init__', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 27):
        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'x' (line 27)
        x_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'x')
        # Getting the type of 'self' (line 27)
        self_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'x' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_34, 'x', x_33)
        
        # Assigning a Name to a Attribute (line 28):
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'y' (line 28)
        y_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'y')
        # Getting the type of 'self' (line 28)
        self_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'y' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_36, 'y', y_35)
        
        # Assigning a Name to a Attribute (line 29):
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'z' (line 29)
        z_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'z')
        # Getting the type of 'self' (line 29)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'z' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_38, 'z', z_37)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Vector' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'Vector', Vector)
# Declaration of the 'Isect' class

class Isect:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Isect.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 34):
        
        # Assigning a Call to a Attribute (line 34):
        
        # Call to Vector(...): (line 34)
        # Processing the call arguments (line 34)
        float_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'float')
        float_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'float')
        float_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'float')
        # Processing the call keyword arguments (line 34)
        kwargs_43 = {}
        # Getting the type of 'Vector' (line 34)
        Vector_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'Vector', False)
        # Calling Vector(args, kwargs) (line 34)
        Vector_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), Vector_39, *[float_40, float_41, float_42], **kwargs_43)
        
        # Getting the type of 'self' (line 34)
        self_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'p' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_45, 'p', Vector_call_result_44)
        
        # Assigning a Call to a Attribute (line 35):
        
        # Assigning a Call to a Attribute (line 35):
        
        # Call to Vector(...): (line 35)
        # Processing the call arguments (line 35)
        float_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'float')
        float_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 29), 'float')
        float_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'float')
        # Processing the call keyword arguments (line 35)
        kwargs_50 = {}
        # Getting the type of 'Vector' (line 35)
        Vector_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'Vector', False)
        # Calling Vector(args, kwargs) (line 35)
        Vector_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), Vector_46, *[float_47, float_48, float_49], **kwargs_50)
        
        # Getting the type of 'self' (line 35)
        self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'n' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_52, 'n', Vector_call_result_51)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Isect.reset.__dict__.__setitem__('stypy_localization', localization)
        Isect.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Isect.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Isect.reset.__dict__.__setitem__('stypy_function_name', 'Isect.reset')
        Isect.reset.__dict__.__setitem__('stypy_param_names_list', [])
        Isect.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Isect.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Isect.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Isect.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Isect.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Isect.reset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Isect.reset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        
        # Assigning a Num to a Attribute (line 38):
        
        # Assigning a Num to a Attribute (line 38):
        float_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'float')
        # Getting the type of 'self' (line 38)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 't' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_54, 't', float_53)
        
        # Assigning a Num to a Attribute (line 39):
        
        # Assigning a Num to a Attribute (line 39):
        int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'int')
        # Getting the type of 'self' (line 39)
        self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'hit' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_56, 'hit', int_55)
        
        # Multiple assignment of 3 elements.
        
        # Assigning a Num to a Attribute (line 40):
        float_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 41), 'float')
        # Getting the type of 'self' (line 40)
        self_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'self')
        # Obtaining the member 'p' of a type (line 40)
        p_59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), self_58, 'p')
        # Setting the type of the member 'z' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), p_59, 'z', float_57)
        
        # Assigning a Attribute to a Attribute (line 40):
        # Getting the type of 'self' (line 40)
        self_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'self')
        # Obtaining the member 'p' of a type (line 40)
        p_61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), self_60, 'p')
        # Obtaining the member 'z' of a type (line 40)
        z_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), p_61, 'z')
        # Getting the type of 'self' (line 40)
        self_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'self')
        # Obtaining the member 'p' of a type (line 40)
        p_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), self_63, 'p')
        # Setting the type of the member 'y' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), p_64, 'y', z_62)
        
        # Assigning a Attribute to a Attribute (line 40):
        # Getting the type of 'self' (line 40)
        self_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'self')
        # Obtaining the member 'p' of a type (line 40)
        p_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), self_65, 'p')
        # Obtaining the member 'y' of a type (line 40)
        y_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), p_66, 'y')
        # Getting the type of 'self' (line 40)
        self_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Obtaining the member 'p' of a type (line 40)
        p_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_68, 'p')
        # Setting the type of the member 'x' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), p_69, 'x', y_67)
        
        # Multiple assignment of 3 elements.
        
        # Assigning a Num to a Attribute (line 41):
        float_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 41), 'float')
        # Getting the type of 'self' (line 41)
        self_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'self')
        # Obtaining the member 'n' of a type (line 41)
        n_72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), self_71, 'n')
        # Setting the type of the member 'z' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), n_72, 'z', float_70)
        
        # Assigning a Attribute to a Attribute (line 41):
        # Getting the type of 'self' (line 41)
        self_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'self')
        # Obtaining the member 'n' of a type (line 41)
        n_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), self_73, 'n')
        # Obtaining the member 'z' of a type (line 41)
        z_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), n_74, 'z')
        # Getting the type of 'self' (line 41)
        self_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'self')
        # Obtaining the member 'n' of a type (line 41)
        n_77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), self_76, 'n')
        # Setting the type of the member 'y' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), n_77, 'y', z_75)
        
        # Assigning a Attribute to a Attribute (line 41):
        # Getting the type of 'self' (line 41)
        self_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'self')
        # Obtaining the member 'n' of a type (line 41)
        n_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), self_78, 'n')
        # Obtaining the member 'y' of a type (line 41)
        y_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), n_79, 'y')
        # Getting the type of 'self' (line 41)
        self_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Obtaining the member 'n' of a type (line 41)
        n_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_81, 'n')
        # Setting the type of the member 'x' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), n_82, 'x', y_80)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_83)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_83


# Assigning a type to the variable 'Isect' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'Isect', Isect)
# Declaration of the 'Sphere' class

class Sphere:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sphere.__init__', ['center', 'radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['center', 'radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 46):
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'center' (line 46)
        center_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'center')
        # Getting the type of 'self' (line 46)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'center' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_85, 'center', center_84)
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'radius' (line 47)
        radius_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'radius')
        # Getting the type of 'self' (line 47)
        self_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'radius' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_87, 'radius', radius_86)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Sphere' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'Sphere', Sphere)
# Declaration of the 'Plane' class

class Plane:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Plane.__init__', ['p', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['p', 'n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 52):
        
        # Assigning a Name to a Attribute (line 52):
        # Getting the type of 'p' (line 52)
        p_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'p')
        # Getting the type of 'self' (line 52)
        self_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member 'p' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_89, 'p', p_88)
        
        # Assigning a Name to a Attribute (line 53):
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'n' (line 53)
        n_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'n')
        # Getting the type of 'self' (line 53)
        self_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'n' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_91, 'n', n_90)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Plane' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'Plane', Plane)
# Declaration of the 'Ray' class

class Ray:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Ray.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 58):
        
        # Assigning a Call to a Attribute (line 58):
        
        # Call to Vector(...): (line 58)
        # Processing the call arguments (line 58)
        float_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'float')
        float_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'float')
        float_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'float')
        # Processing the call keyword arguments (line 58)
        kwargs_96 = {}
        # Getting the type of 'Vector' (line 58)
        Vector_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'Vector', False)
        # Calling Vector(args, kwargs) (line 58)
        Vector_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), Vector_92, *[float_93, float_94, float_95], **kwargs_96)
        
        # Getting the type of 'self' (line 58)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'org' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_98, 'org', Vector_call_result_97)
        
        # Assigning a Call to a Attribute (line 59):
        
        # Assigning a Call to a Attribute (line 59):
        
        # Call to Vector(...): (line 59)
        # Processing the call arguments (line 59)
        float_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'float')
        float_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'float')
        float_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'float')
        # Processing the call keyword arguments (line 59)
        kwargs_103 = {}
        # Getting the type of 'Vector' (line 59)
        Vector_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'Vector', False)
        # Calling Vector(args, kwargs) (line 59)
        Vector_call_result_104 = invoke(stypy.reporting.localization.Localization(__file__, 59, 19), Vector_99, *[float_100, float_101, float_102], **kwargs_103)
        
        # Getting the type of 'self' (line 59)
        self_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'dir' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_105, 'dir', Vector_call_result_104)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Ray.reset.__dict__.__setitem__('stypy_localization', localization)
        Ray.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Ray.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Ray.reset.__dict__.__setitem__('stypy_function_name', 'Ray.reset')
        Ray.reset.__dict__.__setitem__('stypy_param_names_list', ['p', 'x', 'y', 'z'])
        Ray.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Ray.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Ray.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Ray.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Ray.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Ray.reset.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Ray.reset', ['p', 'x', 'y', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, ['p', 'x', 'y', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 62):
        
        # Assigning a Attribute to a Name (line 62):
        # Getting the type of 'p' (line 62)
        p_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 45), 'p')
        # Obtaining the member 'x' of a type (line 62)
        x_107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 45), p_106, 'x')
        # Assigning a type to the variable 'tuple_assignment_1' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_assignment_1', x_107)
        
        # Assigning a Attribute to a Name (line 62):
        # Getting the type of 'p' (line 62)
        p_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 50), 'p')
        # Obtaining the member 'y' of a type (line 62)
        y_109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 50), p_108, 'y')
        # Assigning a type to the variable 'tuple_assignment_2' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_assignment_2', y_109)
        
        # Assigning a Attribute to a Name (line 62):
        # Getting the type of 'p' (line 62)
        p_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 55), 'p')
        # Obtaining the member 'z' of a type (line 62)
        z_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 55), p_110, 'z')
        # Assigning a type to the variable 'tuple_assignment_3' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_assignment_3', z_111)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'tuple_assignment_1' (line 62)
        tuple_assignment_1_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_assignment_1')
        # Getting the type of 'self' (line 62)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Obtaining the member 'org' of a type (line 62)
        org_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_113, 'org')
        # Setting the type of the member 'x' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), org_114, 'x', tuple_assignment_1_112)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'tuple_assignment_2' (line 62)
        tuple_assignment_2_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_assignment_2')
        # Getting the type of 'self' (line 62)
        self_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'self')
        # Obtaining the member 'org' of a type (line 62)
        org_117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), self_116, 'org')
        # Setting the type of the member 'y' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), org_117, 'y', tuple_assignment_2_115)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'tuple_assignment_3' (line 62)
        tuple_assignment_3_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_assignment_3')
        # Getting the type of 'self' (line 62)
        self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'self')
        # Obtaining the member 'org' of a type (line 62)
        org_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 32), self_119, 'org')
        # Setting the type of the member 'z' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 32), org_120, 'z', tuple_assignment_3_118)
        
        # Assigning a Tuple to a Tuple (line 63):
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'x' (line 63)
        x_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'x')
        # Assigning a type to the variable 'tuple_assignment_4' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_4', x_121)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'y' (line 63)
        y_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 48), 'y')
        # Assigning a type to the variable 'tuple_assignment_5' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_5', y_122)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'z' (line 63)
        z_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 51), 'z')
        # Assigning a type to the variable 'tuple_assignment_6' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_6', z_123)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_assignment_4' (line 63)
        tuple_assignment_4_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_4')
        # Getting the type of 'self' (line 63)
        self_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Obtaining the member 'dir' of a type (line 63)
        dir_126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_125, 'dir')
        # Setting the type of the member 'x' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), dir_126, 'x', tuple_assignment_4_124)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_assignment_5' (line 63)
        tuple_assignment_5_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_5')
        # Getting the type of 'self' (line 63)
        self_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'self')
        # Obtaining the member 'dir' of a type (line 63)
        dir_129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), self_128, 'dir')
        # Setting the type of the member 'y' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), dir_129, 'y', tuple_assignment_5_127)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_assignment_6' (line 63)
        tuple_assignment_6_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_6')
        # Getting the type of 'self' (line 63)
        self_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'self')
        # Obtaining the member 'dir' of a type (line 63)
        dir_132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 32), self_131, 'dir')
        # Setting the type of the member 'z' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 32), dir_132, 'z', tuple_assignment_6_130)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_133


# Assigning a type to the variable 'Ray' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'Ray', Ray)

@norecursion
def vdot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'vdot'
    module_type_store = module_type_store.open_function_context('vdot', 66, 0, False)
    
    # Passed parameters checking function
    vdot.stypy_localization = localization
    vdot.stypy_type_of_self = None
    vdot.stypy_type_store = module_type_store
    vdot.stypy_function_name = 'vdot'
    vdot.stypy_param_names_list = ['v0', 'v1']
    vdot.stypy_varargs_param_name = None
    vdot.stypy_kwargs_param_name = None
    vdot.stypy_call_defaults = defaults
    vdot.stypy_call_varargs = varargs
    vdot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vdot', ['v0', 'v1'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vdot', localization, ['v0', 'v1'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vdot(...)' code ##################

    # Getting the type of 'v0' (line 67)
    v0_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'v0')
    # Obtaining the member 'x' of a type (line 67)
    x_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), v0_134, 'x')
    # Getting the type of 'v1' (line 67)
    v1_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'v1')
    # Obtaining the member 'x' of a type (line 67)
    x_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), v1_136, 'x')
    # Applying the binary operator '*' (line 67)
    result_mul_138 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '*', x_135, x_137)
    
    # Getting the type of 'v0' (line 67)
    v0_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'v0')
    # Obtaining the member 'y' of a type (line 67)
    y_140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), v0_139, 'y')
    # Getting the type of 'v1' (line 67)
    v1_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'v1')
    # Obtaining the member 'y' of a type (line 67)
    y_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 32), v1_141, 'y')
    # Applying the binary operator '*' (line 67)
    result_mul_143 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '*', y_140, y_142)
    
    # Applying the binary operator '+' (line 67)
    result_add_144 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '+', result_mul_138, result_mul_143)
    
    # Getting the type of 'v0' (line 67)
    v0_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'v0')
    # Obtaining the member 'z' of a type (line 67)
    z_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 39), v0_145, 'z')
    # Getting the type of 'v1' (line 67)
    v1_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 46), 'v1')
    # Obtaining the member 'z' of a type (line 67)
    z_148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 46), v1_147, 'z')
    # Applying the binary operator '*' (line 67)
    result_mul_149 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 39), '*', z_146, z_148)
    
    # Applying the binary operator '+' (line 67)
    result_add_150 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 37), '+', result_add_144, result_mul_149)
    
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type', result_add_150)
    
    # ################# End of 'vdot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vdot' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vdot'
    return stypy_return_type_151

# Assigning a type to the variable 'vdot' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'vdot', vdot)

@norecursion
def vcross(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'vcross'
    module_type_store = module_type_store.open_function_context('vcross', 70, 0, False)
    
    # Passed parameters checking function
    vcross.stypy_localization = localization
    vcross.stypy_type_of_self = None
    vcross.stypy_type_store = module_type_store
    vcross.stypy_function_name = 'vcross'
    vcross.stypy_param_names_list = ['c', 'v0', 'v1']
    vcross.stypy_varargs_param_name = None
    vcross.stypy_kwargs_param_name = None
    vcross.stypy_call_defaults = defaults
    vcross.stypy_call_varargs = varargs
    vcross.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vcross', ['c', 'v0', 'v1'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vcross', localization, ['c', 'v0', 'v1'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vcross(...)' code ##################

    
    # Assigning a BinOp to a Attribute (line 71):
    
    # Assigning a BinOp to a Attribute (line 71):
    # Getting the type of 'v0' (line 71)
    v0_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'v0')
    # Obtaining the member 'y' of a type (line 71)
    y_153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 10), v0_152, 'y')
    # Getting the type of 'v1' (line 71)
    v1_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'v1')
    # Obtaining the member 'z' of a type (line 71)
    z_155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 17), v1_154, 'z')
    # Applying the binary operator '*' (line 71)
    result_mul_156 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 10), '*', y_153, z_155)
    
    # Getting the type of 'v0' (line 71)
    v0_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'v0')
    # Obtaining the member 'z' of a type (line 71)
    z_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), v0_157, 'z')
    # Getting the type of 'v1' (line 71)
    v1_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 'v1')
    # Obtaining the member 'y' of a type (line 71)
    y_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 31), v1_159, 'y')
    # Applying the binary operator '*' (line 71)
    result_mul_161 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 24), '*', z_158, y_160)
    
    # Applying the binary operator '-' (line 71)
    result_sub_162 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 10), '-', result_mul_156, result_mul_161)
    
    # Getting the type of 'c' (line 71)
    c_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'c')
    # Setting the type of the member 'x' of a type (line 71)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), c_163, 'x', result_sub_162)
    
    # Assigning a BinOp to a Attribute (line 72):
    
    # Assigning a BinOp to a Attribute (line 72):
    # Getting the type of 'v0' (line 72)
    v0_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 10), 'v0')
    # Obtaining the member 'z' of a type (line 72)
    z_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 10), v0_164, 'z')
    # Getting the type of 'v1' (line 72)
    v1_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'v1')
    # Obtaining the member 'x' of a type (line 72)
    x_167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), v1_166, 'x')
    # Applying the binary operator '*' (line 72)
    result_mul_168 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 10), '*', z_165, x_167)
    
    # Getting the type of 'v0' (line 72)
    v0_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'v0')
    # Obtaining the member 'x' of a type (line 72)
    x_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), v0_169, 'x')
    # Getting the type of 'v1' (line 72)
    v1_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'v1')
    # Obtaining the member 'z' of a type (line 72)
    z_172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 31), v1_171, 'z')
    # Applying the binary operator '*' (line 72)
    result_mul_173 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 24), '*', x_170, z_172)
    
    # Applying the binary operator '-' (line 72)
    result_sub_174 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 10), '-', result_mul_168, result_mul_173)
    
    # Getting the type of 'c' (line 72)
    c_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'c')
    # Setting the type of the member 'y' of a type (line 72)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), c_175, 'y', result_sub_174)
    
    # Assigning a BinOp to a Attribute (line 73):
    
    # Assigning a BinOp to a Attribute (line 73):
    # Getting the type of 'v0' (line 73)
    v0_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 10), 'v0')
    # Obtaining the member 'x' of a type (line 73)
    x_177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 10), v0_176, 'x')
    # Getting the type of 'v1' (line 73)
    v1_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'v1')
    # Obtaining the member 'y' of a type (line 73)
    y_179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 17), v1_178, 'y')
    # Applying the binary operator '*' (line 73)
    result_mul_180 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 10), '*', x_177, y_179)
    
    # Getting the type of 'v0' (line 73)
    v0_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'v0')
    # Obtaining the member 'y' of a type (line 73)
    y_182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 24), v0_181, 'y')
    # Getting the type of 'v1' (line 73)
    v1_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'v1')
    # Obtaining the member 'x' of a type (line 73)
    x_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 31), v1_183, 'x')
    # Applying the binary operator '*' (line 73)
    result_mul_185 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 24), '*', y_182, x_184)
    
    # Applying the binary operator '-' (line 73)
    result_sub_186 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 10), '-', result_mul_180, result_mul_185)
    
    # Getting the type of 'c' (line 73)
    c_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'c')
    # Setting the type of the member 'z' of a type (line 73)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 4), c_187, 'z', result_sub_186)
    
    # ################# End of 'vcross(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vcross' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_188)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vcross'
    return stypy_return_type_188

# Assigning a type to the variable 'vcross' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'vcross', vcross)

@norecursion
def vnormalize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'vnormalize'
    module_type_store = module_type_store.open_function_context('vnormalize', 76, 0, False)
    
    # Passed parameters checking function
    vnormalize.stypy_localization = localization
    vnormalize.stypy_type_of_self = None
    vnormalize.stypy_type_store = module_type_store
    vnormalize.stypy_function_name = 'vnormalize'
    vnormalize.stypy_param_names_list = ['c']
    vnormalize.stypy_varargs_param_name = None
    vnormalize.stypy_kwargs_param_name = None
    vnormalize.stypy_call_defaults = defaults
    vnormalize.stypy_call_varargs = varargs
    vnormalize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vnormalize', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vnormalize', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vnormalize(...)' code ##################

    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to sqrt(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to vdot(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'c' (line 77)
    c_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'c', False)
    # Getting the type of 'c' (line 77)
    c_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'c', False)
    # Processing the call keyword arguments (line 77)
    kwargs_193 = {}
    # Getting the type of 'vdot' (line 77)
    vdot_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'vdot', False)
    # Calling vdot(args, kwargs) (line 77)
    vdot_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), vdot_190, *[c_191, c_192], **kwargs_193)
    
    # Processing the call keyword arguments (line 77)
    kwargs_195 = {}
    # Getting the type of 'sqrt' (line 77)
    sqrt_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 77)
    sqrt_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), sqrt_189, *[vdot_call_result_194], **kwargs_195)
    
    # Assigning a type to the variable 'length' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'length', sqrt_call_result_196)
    
    
    # Call to fabs(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'length' (line 79)
    length_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'length', False)
    # Processing the call keyword arguments (line 79)
    kwargs_199 = {}
    # Getting the type of 'fabs' (line 79)
    fabs_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'fabs', False)
    # Calling fabs(args, kwargs) (line 79)
    fabs_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 79, 7), fabs_197, *[length_198], **kwargs_199)
    
    float_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'float')
    # Applying the binary operator '>' (line 79)
    result_gt_202 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 7), '>', fabs_call_result_200, float_201)
    
    # Testing if the type of an if condition is none (line 79)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 4), result_gt_202):
        pass
    else:
        
        # Testing the type of an if condition (line 79)
        if_condition_203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), result_gt_202)
        # Assigning a type to the variable 'if_condition_203' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_203', if_condition_203)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'c' (line 80)
        c_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'c')
        # Obtaining the member 'x' of a type (line 80)
        x_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), c_204, 'x')
        # Getting the type of 'length' (line 80)
        length_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'length')
        # Applying the binary operator 'div=' (line 80)
        result_div_207 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 8), 'div=', x_205, length_206)
        # Getting the type of 'c' (line 80)
        c_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'c')
        # Setting the type of the member 'x' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), c_208, 'x', result_div_207)
        
        
        # Getting the type of 'c' (line 81)
        c_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'c')
        # Obtaining the member 'y' of a type (line 81)
        y_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), c_209, 'y')
        # Getting the type of 'length' (line 81)
        length_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'length')
        # Applying the binary operator 'div=' (line 81)
        result_div_212 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), 'div=', y_210, length_211)
        # Getting the type of 'c' (line 81)
        c_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'c')
        # Setting the type of the member 'y' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), c_213, 'y', result_div_212)
        
        
        # Getting the type of 'c' (line 82)
        c_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'c')
        # Obtaining the member 'z' of a type (line 82)
        z_215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), c_214, 'z')
        # Getting the type of 'length' (line 82)
        length_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'length')
        # Applying the binary operator 'div=' (line 82)
        result_div_217 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 8), 'div=', z_215, length_216)
        # Getting the type of 'c' (line 82)
        c_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'c')
        # Setting the type of the member 'z' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), c_218, 'z', result_div_217)
        
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'vnormalize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vnormalize' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_219)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vnormalize'
    return stypy_return_type_219

# Assigning a type to the variable 'vnormalize' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'vnormalize', vnormalize)

@norecursion
def ray_sphere_intersect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ray_sphere_intersect'
    module_type_store = module_type_store.open_function_context('ray_sphere_intersect', 85, 0, False)
    
    # Passed parameters checking function
    ray_sphere_intersect.stypy_localization = localization
    ray_sphere_intersect.stypy_type_of_self = None
    ray_sphere_intersect.stypy_type_store = module_type_store
    ray_sphere_intersect.stypy_function_name = 'ray_sphere_intersect'
    ray_sphere_intersect.stypy_param_names_list = ['isect', 'ray', 'sphere']
    ray_sphere_intersect.stypy_varargs_param_name = None
    ray_sphere_intersect.stypy_kwargs_param_name = None
    ray_sphere_intersect.stypy_call_defaults = defaults
    ray_sphere_intersect.stypy_call_varargs = varargs
    ray_sphere_intersect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ray_sphere_intersect', ['isect', 'ray', 'sphere'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ray_sphere_intersect', localization, ['isect', 'ray', 'sphere'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ray_sphere_intersect(...)' code ##################

    
    # Assigning a BinOp to a Name (line 86):
    
    # Assigning a BinOp to a Name (line 86):
    # Getting the type of 'ray' (line 86)
    ray_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 10), 'ray')
    # Obtaining the member 'org' of a type (line 86)
    org_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 10), ray_220, 'org')
    # Obtaining the member 'x' of a type (line 86)
    x_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 10), org_221, 'x')
    # Getting the type of 'sphere' (line 86)
    sphere_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'sphere')
    # Obtaining the member 'center' of a type (line 86)
    center_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 22), sphere_223, 'center')
    # Obtaining the member 'x' of a type (line 86)
    x_225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 22), center_224, 'x')
    # Applying the binary operator '-' (line 86)
    result_sub_226 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 10), '-', x_222, x_225)
    
    # Assigning a type to the variable 'rsx' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'rsx', result_sub_226)
    
    # Assigning a BinOp to a Name (line 87):
    
    # Assigning a BinOp to a Name (line 87):
    # Getting the type of 'ray' (line 87)
    ray_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 10), 'ray')
    # Obtaining the member 'org' of a type (line 87)
    org_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 10), ray_227, 'org')
    # Obtaining the member 'y' of a type (line 87)
    y_229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 10), org_228, 'y')
    # Getting the type of 'sphere' (line 87)
    sphere_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'sphere')
    # Obtaining the member 'center' of a type (line 87)
    center_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), sphere_230, 'center')
    # Obtaining the member 'y' of a type (line 87)
    y_232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), center_231, 'y')
    # Applying the binary operator '-' (line 87)
    result_sub_233 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 10), '-', y_229, y_232)
    
    # Assigning a type to the variable 'rsy' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'rsy', result_sub_233)
    
    # Assigning a BinOp to a Name (line 88):
    
    # Assigning a BinOp to a Name (line 88):
    # Getting the type of 'ray' (line 88)
    ray_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 10), 'ray')
    # Obtaining the member 'org' of a type (line 88)
    org_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 10), ray_234, 'org')
    # Obtaining the member 'z' of a type (line 88)
    z_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 10), org_235, 'z')
    # Getting the type of 'sphere' (line 88)
    sphere_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'sphere')
    # Obtaining the member 'center' of a type (line 88)
    center_238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 22), sphere_237, 'center')
    # Obtaining the member 'z' of a type (line 88)
    z_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 22), center_238, 'z')
    # Applying the binary operator '-' (line 88)
    result_sub_240 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 10), '-', z_236, z_239)
    
    # Assigning a type to the variable 'rsz' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'rsz', result_sub_240)
    
    # Assigning a BinOp to a Name (line 90):
    
    # Assigning a BinOp to a Name (line 90):
    # Getting the type of 'rsx' (line 90)
    rsx_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'rsx')
    # Getting the type of 'ray' (line 90)
    ray_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'ray')
    # Obtaining the member 'dir' of a type (line 90)
    dir_243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 14), ray_242, 'dir')
    # Obtaining the member 'x' of a type (line 90)
    x_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 14), dir_243, 'x')
    # Applying the binary operator '*' (line 90)
    result_mul_245 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 8), '*', rsx_241, x_244)
    
    # Getting the type of 'rsy' (line 90)
    rsy_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'rsy')
    # Getting the type of 'ray' (line 90)
    ray_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'ray')
    # Obtaining the member 'dir' of a type (line 90)
    dir_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), ray_247, 'dir')
    # Obtaining the member 'y' of a type (line 90)
    y_249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), dir_248, 'y')
    # Applying the binary operator '*' (line 90)
    result_mul_250 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 26), '*', rsy_246, y_249)
    
    # Applying the binary operator '+' (line 90)
    result_add_251 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 8), '+', result_mul_245, result_mul_250)
    
    # Getting the type of 'rsz' (line 90)
    rsz_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 44), 'rsz')
    # Getting the type of 'ray' (line 90)
    ray_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 50), 'ray')
    # Obtaining the member 'dir' of a type (line 90)
    dir_254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 50), ray_253, 'dir')
    # Obtaining the member 'z' of a type (line 90)
    z_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 50), dir_254, 'z')
    # Applying the binary operator '*' (line 90)
    result_mul_256 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 44), '*', rsz_252, z_255)
    
    # Applying the binary operator '+' (line 90)
    result_add_257 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 42), '+', result_add_251, result_mul_256)
    
    # Assigning a type to the variable 'B' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'B', result_add_257)
    
    # Assigning a BinOp to a Name (line 91):
    
    # Assigning a BinOp to a Name (line 91):
    # Getting the type of 'rsx' (line 91)
    rsx_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'rsx')
    # Getting the type of 'rsx' (line 91)
    rsx_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'rsx')
    # Applying the binary operator '*' (line 91)
    result_mul_260 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 8), '*', rsx_258, rsx_259)
    
    # Getting the type of 'rsy' (line 91)
    rsy_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'rsy')
    # Getting the type of 'rsy' (line 91)
    rsy_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'rsy')
    # Applying the binary operator '*' (line 91)
    result_mul_263 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 20), '*', rsy_261, rsy_262)
    
    # Applying the binary operator '+' (line 91)
    result_add_264 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 8), '+', result_mul_260, result_mul_263)
    
    # Getting the type of 'rsz' (line 91)
    rsz_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'rsz')
    # Getting the type of 'rsz' (line 91)
    rsz_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'rsz')
    # Applying the binary operator '*' (line 91)
    result_mul_267 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 32), '*', rsz_265, rsz_266)
    
    # Applying the binary operator '+' (line 91)
    result_add_268 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 30), '+', result_add_264, result_mul_267)
    
    # Getting the type of 'sphere' (line 91)
    sphere_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 44), 'sphere')
    # Obtaining the member 'radius' of a type (line 91)
    radius_270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 44), sphere_269, 'radius')
    # Getting the type of 'sphere' (line 91)
    sphere_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 60), 'sphere')
    # Obtaining the member 'radius' of a type (line 91)
    radius_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 60), sphere_271, 'radius')
    # Applying the binary operator '*' (line 91)
    result_mul_273 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 44), '*', radius_270, radius_272)
    
    # Applying the binary operator '-' (line 91)
    result_sub_274 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 42), '-', result_add_268, result_mul_273)
    
    # Assigning a type to the variable 'C' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'C', result_sub_274)
    
    # Assigning a BinOp to a Name (line 92):
    
    # Assigning a BinOp to a Name (line 92):
    # Getting the type of 'B' (line 92)
    B_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'B')
    # Getting the type of 'B' (line 92)
    B_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'B')
    # Applying the binary operator '*' (line 92)
    result_mul_277 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 8), '*', B_275, B_276)
    
    # Getting the type of 'C' (line 92)
    C_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'C')
    # Applying the binary operator '-' (line 92)
    result_sub_279 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 8), '-', result_mul_277, C_278)
    
    # Assigning a type to the variable 'D' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'D', result_sub_279)
    
    # Getting the type of 'D' (line 94)
    D_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'D')
    float_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 11), 'float')
    # Applying the binary operator '>' (line 94)
    result_gt_282 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '>', D_280, float_281)
    
    # Testing if the type of an if condition is none (line 94)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 94, 4), result_gt_282):
        pass
    else:
        
        # Testing the type of an if condition (line 94)
        if_condition_283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_gt_282)
        # Assigning a type to the variable 'if_condition_283' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_283', if_condition_283)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 95):
        
        # Assigning a BinOp to a Name (line 95):
        
        # Getting the type of 'B' (line 95)
        B_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'B')
        # Applying the 'usub' unary operator (line 95)
        result___neg___285 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), 'usub', B_284)
        
        
        # Call to sqrt(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'D' (line 95)
        D_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'D', False)
        # Processing the call keyword arguments (line 95)
        kwargs_288 = {}
        # Getting the type of 'sqrt' (line 95)
        sqrt_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 95)
        sqrt_call_result_289 = invoke(stypy.reporting.localization.Localization(__file__, 95, 17), sqrt_286, *[D_287], **kwargs_288)
        
        # Applying the binary operator '-' (line 95)
        result_sub_290 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), '-', result___neg___285, sqrt_call_result_289)
        
        # Assigning a type to the variable 't' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 't', result_sub_290)
        
        # Getting the type of 't' (line 96)
        t_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 't')
        float_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 15), 'float')
        # Applying the binary operator '>' (line 96)
        result_gt_293 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 11), '>', t_291, float_292)
        
        # Testing if the type of an if condition is none (line 96)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 8), result_gt_293):
            pass
        else:
            
            # Testing the type of an if condition (line 96)
            if_condition_294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), result_gt_293)
            # Assigning a type to the variable 'if_condition_294' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_294', if_condition_294)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 't' (line 97)
            t_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 't')
            # Getting the type of 'isect' (line 97)
            isect_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'isect')
            # Obtaining the member 't' of a type (line 97)
            t_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), isect_296, 't')
            # Applying the binary operator '<' (line 97)
            result_lt_298 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), '<', t_295, t_297)
            
            # Testing if the type of an if condition is none (line 97)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 12), result_lt_298):
                pass
            else:
                
                # Testing the type of an if condition (line 97)
                if_condition_299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), result_lt_298)
                # Assigning a type to the variable 'if_condition_299' (line 97)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_299', if_condition_299)
                # SSA begins for if statement (line 97)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 98):
                
                # Assigning a Name to a Attribute (line 98):
                # Getting the type of 't' (line 98)
                t_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 't')
                # Getting the type of 'isect' (line 98)
                isect_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'isect')
                # Setting the type of the member 't' of a type (line 98)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), isect_301, 't', t_300)
                
                # Assigning a Num to a Attribute (line 99):
                
                # Assigning a Num to a Attribute (line 99):
                int_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'int')
                # Getting the type of 'isect' (line 99)
                isect_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'isect')
                # Setting the type of the member 'hit' of a type (line 99)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), isect_303, 'hit', int_302)
                
                # Assigning a BinOp to a Attribute (line 101):
                
                # Assigning a BinOp to a Attribute (line 101):
                # Getting the type of 'ray' (line 101)
                ray_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'ray')
                # Obtaining the member 'org' of a type (line 101)
                org_305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), ray_304, 'org')
                # Obtaining the member 'x' of a type (line 101)
                x_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), org_305, 'x')
                # Getting the type of 'ray' (line 101)
                ray_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'ray')
                # Obtaining the member 'dir' of a type (line 101)
                dir_308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 40), ray_307, 'dir')
                # Obtaining the member 'x' of a type (line 101)
                x_309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 40), dir_308, 'x')
                # Getting the type of 't' (line 101)
                t_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 52), 't')
                # Applying the binary operator '*' (line 101)
                result_mul_311 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 40), '*', x_309, t_310)
                
                # Applying the binary operator '+' (line 101)
                result_add_312 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 28), '+', x_306, result_mul_311)
                
                # Getting the type of 'isect' (line 101)
                isect_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'isect')
                # Obtaining the member 'p' of a type (line 101)
                p_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), isect_313, 'p')
                # Setting the type of the member 'x' of a type (line 101)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), p_314, 'x', result_add_312)
                
                # Assigning a BinOp to a Attribute (line 102):
                
                # Assigning a BinOp to a Attribute (line 102):
                # Getting the type of 'ray' (line 102)
                ray_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'ray')
                # Obtaining the member 'org' of a type (line 102)
                org_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 28), ray_315, 'org')
                # Obtaining the member 'y' of a type (line 102)
                y_317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 28), org_316, 'y')
                # Getting the type of 'ray' (line 102)
                ray_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'ray')
                # Obtaining the member 'dir' of a type (line 102)
                dir_319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), ray_318, 'dir')
                # Obtaining the member 'y' of a type (line 102)
                y_320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), dir_319, 'y')
                # Getting the type of 't' (line 102)
                t_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 52), 't')
                # Applying the binary operator '*' (line 102)
                result_mul_322 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 40), '*', y_320, t_321)
                
                # Applying the binary operator '+' (line 102)
                result_add_323 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 28), '+', y_317, result_mul_322)
                
                # Getting the type of 'isect' (line 102)
                isect_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'isect')
                # Obtaining the member 'p' of a type (line 102)
                p_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), isect_324, 'p')
                # Setting the type of the member 'y' of a type (line 102)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), p_325, 'y', result_add_323)
                
                # Assigning a BinOp to a Attribute (line 103):
                
                # Assigning a BinOp to a Attribute (line 103):
                # Getting the type of 'ray' (line 103)
                ray_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'ray')
                # Obtaining the member 'org' of a type (line 103)
                org_327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 28), ray_326, 'org')
                # Obtaining the member 'z' of a type (line 103)
                z_328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 28), org_327, 'z')
                # Getting the type of 'ray' (line 103)
                ray_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'ray')
                # Obtaining the member 'dir' of a type (line 103)
                dir_330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 40), ray_329, 'dir')
                # Obtaining the member 'z' of a type (line 103)
                z_331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 40), dir_330, 'z')
                # Getting the type of 't' (line 103)
                t_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 52), 't')
                # Applying the binary operator '*' (line 103)
                result_mul_333 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 40), '*', z_331, t_332)
                
                # Applying the binary operator '+' (line 103)
                result_add_334 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 28), '+', z_328, result_mul_333)
                
                # Getting the type of 'isect' (line 103)
                isect_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'isect')
                # Obtaining the member 'p' of a type (line 103)
                p_336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), isect_335, 'p')
                # Setting the type of the member 'z' of a type (line 103)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), p_336, 'z', result_add_334)
                
                # Assigning a BinOp to a Attribute (line 105):
                
                # Assigning a BinOp to a Attribute (line 105):
                # Getting the type of 'isect' (line 105)
                isect_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'isect')
                # Obtaining the member 'p' of a type (line 105)
                p_338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 28), isect_337, 'p')
                # Obtaining the member 'x' of a type (line 105)
                x_339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 28), p_338, 'x')
                # Getting the type of 'sphere' (line 105)
                sphere_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'sphere')
                # Obtaining the member 'center' of a type (line 105)
                center_341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 40), sphere_340, 'center')
                # Obtaining the member 'x' of a type (line 105)
                x_342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 40), center_341, 'x')
                # Applying the binary operator '-' (line 105)
                result_sub_343 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 28), '-', x_339, x_342)
                
                # Getting the type of 'isect' (line 105)
                isect_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'isect')
                # Obtaining the member 'n' of a type (line 105)
                n_345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), isect_344, 'n')
                # Setting the type of the member 'x' of a type (line 105)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), n_345, 'x', result_sub_343)
                
                # Assigning a BinOp to a Attribute (line 106):
                
                # Assigning a BinOp to a Attribute (line 106):
                # Getting the type of 'isect' (line 106)
                isect_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'isect')
                # Obtaining the member 'p' of a type (line 106)
                p_347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 28), isect_346, 'p')
                # Obtaining the member 'y' of a type (line 106)
                y_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 28), p_347, 'y')
                # Getting the type of 'sphere' (line 106)
                sphere_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'sphere')
                # Obtaining the member 'center' of a type (line 106)
                center_350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), sphere_349, 'center')
                # Obtaining the member 'y' of a type (line 106)
                y_351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), center_350, 'y')
                # Applying the binary operator '-' (line 106)
                result_sub_352 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 28), '-', y_348, y_351)
                
                # Getting the type of 'isect' (line 106)
                isect_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'isect')
                # Obtaining the member 'n' of a type (line 106)
                n_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), isect_353, 'n')
                # Setting the type of the member 'y' of a type (line 106)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), n_354, 'y', result_sub_352)
                
                # Assigning a BinOp to a Attribute (line 107):
                
                # Assigning a BinOp to a Attribute (line 107):
                # Getting the type of 'isect' (line 107)
                isect_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'isect')
                # Obtaining the member 'p' of a type (line 107)
                p_356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 28), isect_355, 'p')
                # Obtaining the member 'z' of a type (line 107)
                z_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 28), p_356, 'z')
                # Getting the type of 'sphere' (line 107)
                sphere_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 40), 'sphere')
                # Obtaining the member 'center' of a type (line 107)
                center_359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 40), sphere_358, 'center')
                # Obtaining the member 'z' of a type (line 107)
                z_360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 40), center_359, 'z')
                # Applying the binary operator '-' (line 107)
                result_sub_361 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 28), '-', z_357, z_360)
                
                # Getting the type of 'isect' (line 107)
                isect_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'isect')
                # Obtaining the member 'n' of a type (line 107)
                n_363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), isect_362, 'n')
                # Setting the type of the member 'z' of a type (line 107)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), n_363, 'z', result_sub_361)
                
                # Call to vnormalize(...): (line 109)
                # Processing the call arguments (line 109)
                # Getting the type of 'isect' (line 109)
                isect_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'isect', False)
                # Obtaining the member 'n' of a type (line 109)
                n_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), isect_365, 'n')
                # Processing the call keyword arguments (line 109)
                kwargs_367 = {}
                # Getting the type of 'vnormalize' (line 109)
                vnormalize_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'vnormalize', False)
                # Calling vnormalize(args, kwargs) (line 109)
                vnormalize_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), vnormalize_364, *[n_366], **kwargs_367)
                
                # SSA join for if statement (line 97)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'ray_sphere_intersect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ray_sphere_intersect' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_369)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ray_sphere_intersect'
    return stypy_return_type_369

# Assigning a type to the variable 'ray_sphere_intersect' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'ray_sphere_intersect', ray_sphere_intersect)

@norecursion
def ray_plane_intersect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ray_plane_intersect'
    module_type_store = module_type_store.open_function_context('ray_plane_intersect', 112, 0, False)
    
    # Passed parameters checking function
    ray_plane_intersect.stypy_localization = localization
    ray_plane_intersect.stypy_type_of_self = None
    ray_plane_intersect.stypy_type_store = module_type_store
    ray_plane_intersect.stypy_function_name = 'ray_plane_intersect'
    ray_plane_intersect.stypy_param_names_list = ['isect', 'ray', 'plane']
    ray_plane_intersect.stypy_varargs_param_name = None
    ray_plane_intersect.stypy_kwargs_param_name = None
    ray_plane_intersect.stypy_call_defaults = defaults
    ray_plane_intersect.stypy_call_varargs = varargs
    ray_plane_intersect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ray_plane_intersect', ['isect', 'ray', 'plane'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ray_plane_intersect', localization, ['isect', 'ray', 'plane'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ray_plane_intersect(...)' code ##################

    
    # Assigning a UnaryOp to a Name (line 113):
    
    # Assigning a UnaryOp to a Name (line 113):
    
    
    # Call to vdot(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'plane' (line 113)
    plane_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'plane', False)
    # Obtaining the member 'p' of a type (line 113)
    p_372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 14), plane_371, 'p')
    # Getting the type of 'plane' (line 113)
    plane_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'plane', False)
    # Obtaining the member 'n' of a type (line 113)
    n_374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 23), plane_373, 'n')
    # Processing the call keyword arguments (line 113)
    kwargs_375 = {}
    # Getting the type of 'vdot' (line 113)
    vdot_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'vdot', False)
    # Calling vdot(args, kwargs) (line 113)
    vdot_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), vdot_370, *[p_372, n_374], **kwargs_375)
    
    # Applying the 'usub' unary operator (line 113)
    result___neg___377 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 8), 'usub', vdot_call_result_376)
    
    # Assigning a type to the variable 'd' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'd', result___neg___377)
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to vdot(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'ray' (line 114)
    ray_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'ray', False)
    # Obtaining the member 'dir' of a type (line 114)
    dir_380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), ray_379, 'dir')
    # Getting the type of 'plane' (line 114)
    plane_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'plane', False)
    # Obtaining the member 'n' of a type (line 114)
    n_382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), plane_381, 'n')
    # Processing the call keyword arguments (line 114)
    kwargs_383 = {}
    # Getting the type of 'vdot' (line 114)
    vdot_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'vdot', False)
    # Calling vdot(args, kwargs) (line 114)
    vdot_call_result_384 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), vdot_378, *[dir_380, n_382], **kwargs_383)
    
    # Assigning a type to the variable 'v' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'v', vdot_call_result_384)
    
    
    # Call to abs(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'v' (line 116)
    v_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'v', False)
    # Processing the call keyword arguments (line 116)
    kwargs_387 = {}
    # Getting the type of 'abs' (line 116)
    abs_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'abs', False)
    # Calling abs(args, kwargs) (line 116)
    abs_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 116, 7), abs_385, *[v_386], **kwargs_387)
    
    float_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 16), 'float')
    # Applying the binary operator '<' (line 116)
    result_lt_390 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), '<', abs_call_result_388, float_389)
    
    # Testing if the type of an if condition is none (line 116)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 4), result_lt_390):
        pass
    else:
        
        # Testing the type of an if condition (line 116)
        if_condition_391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_lt_390)
        # Assigning a type to the variable 'if_condition_391' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_391', if_condition_391)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 119):
    
    # Assigning a BinOp to a Name (line 119):
    
    
    # Call to vdot(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'ray' (line 119)
    ray_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'ray', False)
    # Obtaining the member 'org' of a type (line 119)
    org_394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), ray_393, 'org')
    # Getting the type of 'plane' (line 119)
    plane_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'plane', False)
    # Obtaining the member 'n' of a type (line 119)
    n_396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 24), plane_395, 'n')
    # Processing the call keyword arguments (line 119)
    kwargs_397 = {}
    # Getting the type of 'vdot' (line 119)
    vdot_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 10), 'vdot', False)
    # Calling vdot(args, kwargs) (line 119)
    vdot_call_result_398 = invoke(stypy.reporting.localization.Localization(__file__, 119, 10), vdot_392, *[org_394, n_396], **kwargs_397)
    
    # Getting the type of 'd' (line 119)
    d_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'd')
    # Applying the binary operator '+' (line 119)
    result_add_400 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 10), '+', vdot_call_result_398, d_399)
    
    # Applying the 'usub' unary operator (line 119)
    result___neg___401 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 8), 'usub', result_add_400)
    
    # Getting the type of 'v' (line 119)
    v_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'v')
    # Applying the binary operator 'div' (line 119)
    result_div_403 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 8), 'div', result___neg___401, v_402)
    
    # Assigning a type to the variable 't' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 't', result_div_403)
    
    # Getting the type of 't' (line 121)
    t_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 't')
    float_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 11), 'float')
    # Applying the binary operator '>' (line 121)
    result_gt_406 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), '>', t_404, float_405)
    
    # Testing if the type of an if condition is none (line 121)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 4), result_gt_406):
        pass
    else:
        
        # Testing the type of an if condition (line 121)
        if_condition_407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_gt_406)
        # Assigning a type to the variable 'if_condition_407' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_407', if_condition_407)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 't' (line 122)
        t_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 't')
        # Getting the type of 'isect' (line 122)
        isect_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'isect')
        # Obtaining the member 't' of a type (line 122)
        t_410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), isect_409, 't')
        # Applying the binary operator '<' (line 122)
        result_lt_411 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 11), '<', t_408, t_410)
        
        # Testing if the type of an if condition is none (line 122)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 8), result_lt_411):
            pass
        else:
            
            # Testing the type of an if condition (line 122)
            if_condition_412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 8), result_lt_411)
            # Assigning a type to the variable 'if_condition_412' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'if_condition_412', if_condition_412)
            # SSA begins for if statement (line 122)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 123):
            
            # Assigning a Name to a Attribute (line 123):
            # Getting the type of 't' (line 123)
            t_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 't')
            # Getting the type of 'isect' (line 123)
            isect_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'isect')
            # Setting the type of the member 't' of a type (line 123)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), isect_414, 't', t_413)
            
            # Assigning a Num to a Attribute (line 124):
            
            # Assigning a Num to a Attribute (line 124):
            int_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'int')
            # Getting the type of 'isect' (line 124)
            isect_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'isect')
            # Setting the type of the member 'hit' of a type (line 124)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), isect_416, 'hit', int_415)
            
            # Assigning a BinOp to a Attribute (line 126):
            
            # Assigning a BinOp to a Attribute (line 126):
            # Getting the type of 'ray' (line 126)
            ray_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'ray')
            # Obtaining the member 'org' of a type (line 126)
            org_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), ray_417, 'org')
            # Obtaining the member 'x' of a type (line 126)
            x_419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), org_418, 'x')
            # Getting the type of 'ray' (line 126)
            ray_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'ray')
            # Obtaining the member 'dir' of a type (line 126)
            dir_421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 36), ray_420, 'dir')
            # Obtaining the member 'x' of a type (line 126)
            x_422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 36), dir_421, 'x')
            # Getting the type of 't' (line 126)
            t_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 48), 't')
            # Applying the binary operator '*' (line 126)
            result_mul_424 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 36), '*', x_422, t_423)
            
            # Applying the binary operator '+' (line 126)
            result_add_425 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 24), '+', x_419, result_mul_424)
            
            # Getting the type of 'isect' (line 126)
            isect_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'isect')
            # Obtaining the member 'p' of a type (line 126)
            p_427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), isect_426, 'p')
            # Setting the type of the member 'x' of a type (line 126)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), p_427, 'x', result_add_425)
            
            # Assigning a BinOp to a Attribute (line 127):
            
            # Assigning a BinOp to a Attribute (line 127):
            # Getting the type of 'ray' (line 127)
            ray_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'ray')
            # Obtaining the member 'org' of a type (line 127)
            org_429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), ray_428, 'org')
            # Obtaining the member 'y' of a type (line 127)
            y_430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), org_429, 'y')
            # Getting the type of 'ray' (line 127)
            ray_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'ray')
            # Obtaining the member 'dir' of a type (line 127)
            dir_432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 36), ray_431, 'dir')
            # Obtaining the member 'y' of a type (line 127)
            y_433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 36), dir_432, 'y')
            # Getting the type of 't' (line 127)
            t_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 48), 't')
            # Applying the binary operator '*' (line 127)
            result_mul_435 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 36), '*', y_433, t_434)
            
            # Applying the binary operator '+' (line 127)
            result_add_436 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 24), '+', y_430, result_mul_435)
            
            # Getting the type of 'isect' (line 127)
            isect_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'isect')
            # Obtaining the member 'p' of a type (line 127)
            p_438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), isect_437, 'p')
            # Setting the type of the member 'y' of a type (line 127)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), p_438, 'y', result_add_436)
            
            # Assigning a BinOp to a Attribute (line 128):
            
            # Assigning a BinOp to a Attribute (line 128):
            # Getting the type of 'ray' (line 128)
            ray_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'ray')
            # Obtaining the member 'org' of a type (line 128)
            org_440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), ray_439, 'org')
            # Obtaining the member 'z' of a type (line 128)
            z_441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), org_440, 'z')
            # Getting the type of 'ray' (line 128)
            ray_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 36), 'ray')
            # Obtaining the member 'dir' of a type (line 128)
            dir_443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 36), ray_442, 'dir')
            # Obtaining the member 'z' of a type (line 128)
            z_444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 36), dir_443, 'z')
            # Getting the type of 't' (line 128)
            t_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 48), 't')
            # Applying the binary operator '*' (line 128)
            result_mul_446 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 36), '*', z_444, t_445)
            
            # Applying the binary operator '+' (line 128)
            result_add_447 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 24), '+', z_441, result_mul_446)
            
            # Getting the type of 'isect' (line 128)
            isect_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'isect')
            # Obtaining the member 'p' of a type (line 128)
            p_449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), isect_448, 'p')
            # Setting the type of the member 'z' of a type (line 128)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), p_449, 'z', result_add_447)
            
            # Assigning a Attribute to a Attribute (line 130):
            
            # Assigning a Attribute to a Attribute (line 130):
            # Getting the type of 'plane' (line 130)
            plane_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'plane')
            # Obtaining the member 'n' of a type (line 130)
            n_451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 24), plane_450, 'n')
            # Obtaining the member 'x' of a type (line 130)
            x_452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 24), n_451, 'x')
            # Getting the type of 'isect' (line 130)
            isect_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'isect')
            # Obtaining the member 'n' of a type (line 130)
            n_454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), isect_453, 'n')
            # Setting the type of the member 'x' of a type (line 130)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), n_454, 'x', x_452)
            
            # Assigning a Attribute to a Attribute (line 131):
            
            # Assigning a Attribute to a Attribute (line 131):
            # Getting the type of 'plane' (line 131)
            plane_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'plane')
            # Obtaining the member 'n' of a type (line 131)
            n_456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), plane_455, 'n')
            # Obtaining the member 'y' of a type (line 131)
            y_457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), n_456, 'y')
            # Getting the type of 'isect' (line 131)
            isect_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'isect')
            # Obtaining the member 'n' of a type (line 131)
            n_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), isect_458, 'n')
            # Setting the type of the member 'y' of a type (line 131)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), n_459, 'y', y_457)
            
            # Assigning a Attribute to a Attribute (line 132):
            
            # Assigning a Attribute to a Attribute (line 132):
            # Getting the type of 'plane' (line 132)
            plane_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'plane')
            # Obtaining the member 'n' of a type (line 132)
            n_461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 24), plane_460, 'n')
            # Obtaining the member 'z' of a type (line 132)
            z_462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 24), n_461, 'z')
            # Getting the type of 'isect' (line 132)
            isect_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'isect')
            # Obtaining the member 'n' of a type (line 132)
            n_464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), isect_463, 'n')
            # Setting the type of the member 'z' of a type (line 132)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), n_464, 'z', z_462)
            # SSA join for if statement (line 122)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'ray_plane_intersect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ray_plane_intersect' in the type store
    # Getting the type of 'stypy_return_type' (line 112)
    stypy_return_type_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_465)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ray_plane_intersect'
    return stypy_return_type_465

# Assigning a type to the variable 'ray_plane_intersect' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'ray_plane_intersect', ray_plane_intersect)

@norecursion
def ortho_basis(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ortho_basis'
    module_type_store = module_type_store.open_function_context('ortho_basis', 135, 0, False)
    
    # Passed parameters checking function
    ortho_basis.stypy_localization = localization
    ortho_basis.stypy_type_of_self = None
    ortho_basis.stypy_type_store = module_type_store
    ortho_basis.stypy_function_name = 'ortho_basis'
    ortho_basis.stypy_param_names_list = ['basis', 'n']
    ortho_basis.stypy_varargs_param_name = None
    ortho_basis.stypy_kwargs_param_name = None
    ortho_basis.stypy_call_defaults = defaults
    ortho_basis.stypy_call_varargs = varargs
    ortho_basis.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ortho_basis', ['basis', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ortho_basis', localization, ['basis', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ortho_basis(...)' code ##################

    
    # Assigning a Name to a Subscript (line 136):
    
    # Assigning a Name to a Subscript (line 136):
    # Getting the type of 'n' (line 136)
    n_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'n')
    # Getting the type of 'basis' (line 136)
    basis_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'basis')
    int_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 10), 'int')
    # Storing an element on a container (line 136)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 4), basis_467, (int_468, n_466))
    
    # Multiple assignment of 3 elements.
    
    # Assigning a Num to a Attribute (line 137):
    float_469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 43), 'float')
    
    # Obtaining the type of the subscript
    int_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 36), 'int')
    # Getting the type of 'basis' (line 137)
    basis_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'basis')
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 30), basis_471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 137, 30), getitem___472, int_470)
    
    # Setting the type of the member 'z' of a type (line 137)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 30), subscript_call_result_473, 'z', float_469)
    
    # Assigning a Attribute to a Attribute (line 137):
    
    # Obtaining the type of the subscript
    int_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 36), 'int')
    # Getting the type of 'basis' (line 137)
    basis_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'basis')
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 30), basis_475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 137, 30), getitem___476, int_474)
    
    # Obtaining the member 'z' of a type (line 137)
    z_478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 30), subscript_call_result_477, 'z')
    
    # Obtaining the type of the subscript
    int_479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'int')
    # Getting the type of 'basis' (line 137)
    basis_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'basis')
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 17), basis_480, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 137, 17), getitem___481, int_479)
    
    # Setting the type of the member 'y' of a type (line 137)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 17), subscript_call_result_482, 'y', z_478)
    
    # Assigning a Attribute to a Attribute (line 137):
    
    # Obtaining the type of the subscript
    int_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'int')
    # Getting the type of 'basis' (line 137)
    basis_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'basis')
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 17), basis_484, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 137, 17), getitem___485, int_483)
    
    # Obtaining the member 'y' of a type (line 137)
    y_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 17), subscript_call_result_486, 'y')
    
    # Obtaining the type of the subscript
    int_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 10), 'int')
    # Getting the type of 'basis' (line 137)
    basis_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'basis')
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), basis_489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), getitem___490, int_488)
    
    # Setting the type of the member 'x' of a type (line 137)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), subscript_call_result_491, 'x', y_487)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 139)
    n_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 7), 'n')
    # Obtaining the member 'x' of a type (line 139)
    x_493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 7), n_492, 'x')
    float_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 13), 'float')
    # Applying the binary operator '<' (line 139)
    result_lt_495 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 7), '<', x_493, float_494)
    
    
    # Getting the type of 'n' (line 139)
    n_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'n')
    # Obtaining the member 'x' of a type (line 139)
    x_497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 21), n_496, 'x')
    float_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 27), 'float')
    # Applying the binary operator '>' (line 139)
    result_gt_499 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), '>', x_497, float_498)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_500 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 7), 'and', result_lt_495, result_gt_499)
    
    # Testing if the type of an if condition is none (line 139)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 4), result_and_keyword_500):
        
        # Evaluating a boolean operation
        
        # Getting the type of 'n' (line 141)
        n_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'n')
        # Obtaining the member 'y' of a type (line 141)
        y_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 9), n_507, 'y')
        float_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'float')
        # Applying the binary operator '<' (line 141)
        result_lt_510 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), '<', y_508, float_509)
        
        
        # Getting the type of 'n' (line 141)
        n_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'n')
        # Obtaining the member 'y' of a type (line 141)
        y_512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 23), n_511, 'y')
        float_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'float')
        # Applying the binary operator '>' (line 141)
        result_gt_514 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 23), '>', y_512, float_513)
        
        # Applying the binary operator 'and' (line 141)
        result_and_keyword_515 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), 'and', result_lt_510, result_gt_514)
        
        # Testing if the type of an if condition is none (line 141)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 9), result_and_keyword_515):
            
            # Evaluating a boolean operation
            
            # Getting the type of 'n' (line 143)
            n_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 9), n_522, 'z')
            float_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'float')
            # Applying the binary operator '<' (line 143)
            result_lt_525 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), '<', z_523, float_524)
            
            
            # Getting the type of 'n' (line 143)
            n_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), n_526, 'z')
            float_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'float')
            # Applying the binary operator '>' (line 143)
            result_gt_529 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), '>', z_527, float_528)
            
            # Applying the binary operator 'and' (line 143)
            result_and_keyword_530 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), 'and', result_lt_525, result_gt_529)
            
            # Testing if the type of an if condition is none (line 143)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530):
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
            else:
                
                # Testing the type of an if condition (line 143)
                if_condition_531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530)
                # Assigning a type to the variable 'if_condition_531' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'if_condition_531', if_condition_531)
                # SSA begins for if statement (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Attribute (line 144):
                
                # Assigning a Num to a Attribute (line 144):
                float_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'float')
                
                # Obtaining the type of the subscript
                int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'int')
                # Getting the type of 'basis' (line 144)
                basis_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 144)
                getitem___535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), basis_534, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 144)
                subscript_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), getitem___535, int_533)
                
                # Setting the type of the member 'z' of a type (line 144)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), subscript_call_result_536, 'z', float_532)
                # SSA branch for the else part of an if statement (line 143)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 141)
            if_condition_516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 9), result_and_keyword_515)
            # Assigning a type to the variable 'if_condition_516' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'if_condition_516', if_condition_516)
            # SSA begins for if statement (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Attribute (line 142):
            
            # Assigning a Num to a Attribute (line 142):
            float_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'float')
            
            # Obtaining the type of the subscript
            int_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 14), 'int')
            # Getting the type of 'basis' (line 142)
            basis_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'basis')
            # Obtaining the member '__getitem__' of a type (line 142)
            getitem___520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), basis_519, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 142)
            subscript_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), getitem___520, int_518)
            
            # Setting the type of the member 'y' of a type (line 142)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), subscript_call_result_521, 'y', float_517)
            # SSA branch for the else part of an if statement (line 141)
            module_type_store.open_ssa_branch('else')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'n' (line 143)
            n_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 9), n_522, 'z')
            float_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'float')
            # Applying the binary operator '<' (line 143)
            result_lt_525 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), '<', z_523, float_524)
            
            
            # Getting the type of 'n' (line 143)
            n_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), n_526, 'z')
            float_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'float')
            # Applying the binary operator '>' (line 143)
            result_gt_529 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), '>', z_527, float_528)
            
            # Applying the binary operator 'and' (line 143)
            result_and_keyword_530 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), 'and', result_lt_525, result_gt_529)
            
            # Testing if the type of an if condition is none (line 143)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530):
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
            else:
                
                # Testing the type of an if condition (line 143)
                if_condition_531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530)
                # Assigning a type to the variable 'if_condition_531' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'if_condition_531', if_condition_531)
                # SSA begins for if statement (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Attribute (line 144):
                
                # Assigning a Num to a Attribute (line 144):
                float_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'float')
                
                # Obtaining the type of the subscript
                int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'int')
                # Getting the type of 'basis' (line 144)
                basis_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 144)
                getitem___535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), basis_534, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 144)
                subscript_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), getitem___535, int_533)
                
                # Setting the type of the member 'z' of a type (line 144)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), subscript_call_result_536, 'z', float_532)
                # SSA branch for the else part of an if statement (line 143)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 139)
        if_condition_501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 4), result_and_keyword_500)
        # Assigning a type to the variable 'if_condition_501' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'if_condition_501', if_condition_501)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 140):
        
        # Assigning a Num to a Attribute (line 140):
        float_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'float')
        
        # Obtaining the type of the subscript
        int_503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 14), 'int')
        # Getting the type of 'basis' (line 140)
        basis_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'basis')
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), basis_504, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), getitem___505, int_503)
        
        # Setting the type of the member 'x' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), subscript_call_result_506, 'x', float_502)
        # SSA branch for the else part of an if statement (line 139)
        module_type_store.open_ssa_branch('else')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'n' (line 141)
        n_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'n')
        # Obtaining the member 'y' of a type (line 141)
        y_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 9), n_507, 'y')
        float_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'float')
        # Applying the binary operator '<' (line 141)
        result_lt_510 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), '<', y_508, float_509)
        
        
        # Getting the type of 'n' (line 141)
        n_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'n')
        # Obtaining the member 'y' of a type (line 141)
        y_512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 23), n_511, 'y')
        float_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'float')
        # Applying the binary operator '>' (line 141)
        result_gt_514 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 23), '>', y_512, float_513)
        
        # Applying the binary operator 'and' (line 141)
        result_and_keyword_515 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), 'and', result_lt_510, result_gt_514)
        
        # Testing if the type of an if condition is none (line 141)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 9), result_and_keyword_515):
            
            # Evaluating a boolean operation
            
            # Getting the type of 'n' (line 143)
            n_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 9), n_522, 'z')
            float_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'float')
            # Applying the binary operator '<' (line 143)
            result_lt_525 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), '<', z_523, float_524)
            
            
            # Getting the type of 'n' (line 143)
            n_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), n_526, 'z')
            float_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'float')
            # Applying the binary operator '>' (line 143)
            result_gt_529 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), '>', z_527, float_528)
            
            # Applying the binary operator 'and' (line 143)
            result_and_keyword_530 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), 'and', result_lt_525, result_gt_529)
            
            # Testing if the type of an if condition is none (line 143)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530):
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
            else:
                
                # Testing the type of an if condition (line 143)
                if_condition_531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530)
                # Assigning a type to the variable 'if_condition_531' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'if_condition_531', if_condition_531)
                # SSA begins for if statement (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Attribute (line 144):
                
                # Assigning a Num to a Attribute (line 144):
                float_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'float')
                
                # Obtaining the type of the subscript
                int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'int')
                # Getting the type of 'basis' (line 144)
                basis_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 144)
                getitem___535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), basis_534, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 144)
                subscript_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), getitem___535, int_533)
                
                # Setting the type of the member 'z' of a type (line 144)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), subscript_call_result_536, 'z', float_532)
                # SSA branch for the else part of an if statement (line 143)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 141)
            if_condition_516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 9), result_and_keyword_515)
            # Assigning a type to the variable 'if_condition_516' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'if_condition_516', if_condition_516)
            # SSA begins for if statement (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Attribute (line 142):
            
            # Assigning a Num to a Attribute (line 142):
            float_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'float')
            
            # Obtaining the type of the subscript
            int_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 14), 'int')
            # Getting the type of 'basis' (line 142)
            basis_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'basis')
            # Obtaining the member '__getitem__' of a type (line 142)
            getitem___520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), basis_519, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 142)
            subscript_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), getitem___520, int_518)
            
            # Setting the type of the member 'y' of a type (line 142)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), subscript_call_result_521, 'y', float_517)
            # SSA branch for the else part of an if statement (line 141)
            module_type_store.open_ssa_branch('else')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'n' (line 143)
            n_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 9), n_522, 'z')
            float_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'float')
            # Applying the binary operator '<' (line 143)
            result_lt_525 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), '<', z_523, float_524)
            
            
            # Getting the type of 'n' (line 143)
            n_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'n')
            # Obtaining the member 'z' of a type (line 143)
            z_527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), n_526, 'z')
            float_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'float')
            # Applying the binary operator '>' (line 143)
            result_gt_529 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), '>', z_527, float_528)
            
            # Applying the binary operator 'and' (line 143)
            result_and_keyword_530 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), 'and', result_lt_525, result_gt_529)
            
            # Testing if the type of an if condition is none (line 143)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530):
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
            else:
                
                # Testing the type of an if condition (line 143)
                if_condition_531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 9), result_and_keyword_530)
                # Assigning a type to the variable 'if_condition_531' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'if_condition_531', if_condition_531)
                # SSA begins for if statement (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Attribute (line 144):
                
                # Assigning a Num to a Attribute (line 144):
                float_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'float')
                
                # Obtaining the type of the subscript
                int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'int')
                # Getting the type of 'basis' (line 144)
                basis_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 144)
                getitem___535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), basis_534, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 144)
                subscript_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), getitem___535, int_533)
                
                # Setting the type of the member 'z' of a type (line 144)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), subscript_call_result_536, 'z', float_532)
                # SSA branch for the else part of an if statement (line 143)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Num to a Attribute (line 146):
                
                # Assigning a Num to a Attribute (line 146):
                float_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'float')
                
                # Obtaining the type of the subscript
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
                # Getting the type of 'basis' (line 146)
                basis_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'basis')
                # Obtaining the member '__getitem__' of a type (line 146)
                getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), basis_539, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 146)
                subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___540, int_538)
                
                # Setting the type of the member 'x' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_541, 'x', float_537)
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to vcross(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Obtaining the type of the subscript
    int_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 17), 'int')
    # Getting the type of 'basis' (line 148)
    basis_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), basis_544, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_546 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), getitem___545, int_543)
    
    
    # Obtaining the type of the subscript
    int_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'int')
    # Getting the type of 'basis' (line 148)
    basis_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 21), basis_548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), getitem___549, int_547)
    
    
    # Obtaining the type of the subscript
    int_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 37), 'int')
    # Getting the type of 'basis' (line 148)
    basis_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 31), basis_552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_554 = invoke(stypy.reporting.localization.Localization(__file__, 148, 31), getitem___553, int_551)
    
    # Processing the call keyword arguments (line 148)
    kwargs_555 = {}
    # Getting the type of 'vcross' (line 148)
    vcross_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'vcross', False)
    # Calling vcross(args, kwargs) (line 148)
    vcross_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), vcross_542, *[subscript_call_result_546, subscript_call_result_550, subscript_call_result_554], **kwargs_555)
    
    
    # Call to vnormalize(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Obtaining the type of the subscript
    int_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 21), 'int')
    # Getting the type of 'basis' (line 149)
    basis_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), basis_559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_561 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), getitem___560, int_558)
    
    # Processing the call keyword arguments (line 149)
    kwargs_562 = {}
    # Getting the type of 'vnormalize' (line 149)
    vnormalize_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'vnormalize', False)
    # Calling vnormalize(args, kwargs) (line 149)
    vnormalize_call_result_563 = invoke(stypy.reporting.localization.Localization(__file__, 149, 4), vnormalize_557, *[subscript_call_result_561], **kwargs_562)
    
    
    # Call to vcross(...): (line 151)
    # Processing the call arguments (line 151)
    
    # Obtaining the type of the subscript
    int_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 17), 'int')
    # Getting the type of 'basis' (line 151)
    basis_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), basis_566, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), getitem___567, int_565)
    
    
    # Obtaining the type of the subscript
    int_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 27), 'int')
    # Getting the type of 'basis' (line 151)
    basis_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 21), basis_570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), getitem___571, int_569)
    
    
    # Obtaining the type of the subscript
    int_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 37), 'int')
    # Getting the type of 'basis' (line 151)
    basis_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 31), basis_574, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_576 = invoke(stypy.reporting.localization.Localization(__file__, 151, 31), getitem___575, int_573)
    
    # Processing the call keyword arguments (line 151)
    kwargs_577 = {}
    # Getting the type of 'vcross' (line 151)
    vcross_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'vcross', False)
    # Calling vcross(args, kwargs) (line 151)
    vcross_call_result_578 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), vcross_564, *[subscript_call_result_568, subscript_call_result_572, subscript_call_result_576], **kwargs_577)
    
    
    # Call to vnormalize(...): (line 152)
    # Processing the call arguments (line 152)
    
    # Obtaining the type of the subscript
    int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 21), 'int')
    # Getting the type of 'basis' (line 152)
    basis_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 15), basis_581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), getitem___582, int_580)
    
    # Processing the call keyword arguments (line 152)
    kwargs_584 = {}
    # Getting the type of 'vnormalize' (line 152)
    vnormalize_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'vnormalize', False)
    # Calling vnormalize(args, kwargs) (line 152)
    vnormalize_call_result_585 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), vnormalize_579, *[subscript_call_result_583], **kwargs_584)
    
    
    # ################# End of 'ortho_basis(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ortho_basis' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_586)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ortho_basis'
    return stypy_return_type_586

# Assigning a type to the variable 'ortho_basis' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'ortho_basis', ortho_basis)

@norecursion
def ambient_occlusion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ambient_occlusion'
    module_type_store = module_type_store.open_function_context('ambient_occlusion', 155, 0, False)
    
    # Passed parameters checking function
    ambient_occlusion.stypy_localization = localization
    ambient_occlusion.stypy_type_of_self = None
    ambient_occlusion.stypy_type_store = module_type_store
    ambient_occlusion.stypy_function_name = 'ambient_occlusion'
    ambient_occlusion.stypy_param_names_list = ['col', 'isect']
    ambient_occlusion.stypy_varargs_param_name = None
    ambient_occlusion.stypy_kwargs_param_name = None
    ambient_occlusion.stypy_call_defaults = defaults
    ambient_occlusion.stypy_call_varargs = varargs
    ambient_occlusion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ambient_occlusion', ['col', 'isect'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ambient_occlusion', localization, ['col', 'isect'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ambient_occlusion(...)' code ##################

    # Marking variables as global (line 156)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 156, 4), 'random_idx')
    
    # Assigning a Name to a Name (line 157):
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'NAO_SAMPLES' (line 157)
    NAO_SAMPLES_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'NAO_SAMPLES')
    # Assigning a type to the variable 'ntheta' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'ntheta', NAO_SAMPLES_587)
    
    # Assigning a Name to a Name (line 158):
    
    # Assigning a Name to a Name (line 158):
    # Getting the type of 'NAO_SAMPLES' (line 158)
    NAO_SAMPLES_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'NAO_SAMPLES')
    # Assigning a type to the variable 'nphi' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'nphi', NAO_SAMPLES_588)
    
    # Assigning a Num to a Name (line 159):
    
    # Assigning a Num to a Name (line 159):
    float_589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 10), 'float')
    # Assigning a type to the variable 'eps' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'eps', float_589)
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to Vector(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'isect' (line 161)
    isect_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'isect', False)
    # Obtaining the member 'p' of a type (line 161)
    p_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), isect_591, 'p')
    # Obtaining the member 'x' of a type (line 161)
    x_593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), p_592, 'x')
    # Getting the type of 'eps' (line 161)
    eps_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'eps', False)
    # Getting the type of 'isect' (line 161)
    isect_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'isect', False)
    # Obtaining the member 'n' of a type (line 161)
    n_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 33), isect_595, 'n')
    # Obtaining the member 'x' of a type (line 161)
    x_597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 33), n_596, 'x')
    # Applying the binary operator '*' (line 161)
    result_mul_598 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 27), '*', eps_594, x_597)
    
    # Applying the binary operator '+' (line 161)
    result_add_599 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '+', x_593, result_mul_598)
    
    # Getting the type of 'isect' (line 162)
    isect_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'isect', False)
    # Obtaining the member 'p' of a type (line 162)
    p_601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 15), isect_600, 'p')
    # Obtaining the member 'y' of a type (line 162)
    y_602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 15), p_601, 'y')
    # Getting the type of 'eps' (line 162)
    eps_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'eps', False)
    # Getting the type of 'isect' (line 162)
    isect_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 33), 'isect', False)
    # Obtaining the member 'n' of a type (line 162)
    n_605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 33), isect_604, 'n')
    # Obtaining the member 'y' of a type (line 162)
    y_606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 33), n_605, 'y')
    # Applying the binary operator '*' (line 162)
    result_mul_607 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 27), '*', eps_603, y_606)
    
    # Applying the binary operator '+' (line 162)
    result_add_608 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 15), '+', y_602, result_mul_607)
    
    # Getting the type of 'isect' (line 163)
    isect_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'isect', False)
    # Obtaining the member 'p' of a type (line 163)
    p_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), isect_609, 'p')
    # Obtaining the member 'z' of a type (line 163)
    z_611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), p_610, 'z')
    # Getting the type of 'eps' (line 163)
    eps_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'eps', False)
    # Getting the type of 'isect' (line 163)
    isect_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'isect', False)
    # Obtaining the member 'n' of a type (line 163)
    n_614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 33), isect_613, 'n')
    # Obtaining the member 'z' of a type (line 163)
    z_615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 33), n_614, 'z')
    # Applying the binary operator '*' (line 163)
    result_mul_616 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 27), '*', eps_612, z_615)
    
    # Applying the binary operator '+' (line 163)
    result_add_617 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 15), '+', z_611, result_mul_616)
    
    # Processing the call keyword arguments (line 161)
    kwargs_618 = {}
    # Getting the type of 'Vector' (line 161)
    Vector_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'Vector', False)
    # Calling Vector(args, kwargs) (line 161)
    Vector_call_result_619 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), Vector_590, *[result_add_599, result_add_608, result_add_617], **kwargs_618)
    
    # Assigning a type to the variable 'p' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'p', Vector_call_result_619)
    
    # Assigning a ListComp to a Name (line 165):
    
    # Assigning a ListComp to a Name (line 165):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 165)
    # Processing the call arguments (line 165)
    int_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 50), 'int')
    # Processing the call keyword arguments (line 165)
    kwargs_628 = {}
    # Getting the type of 'range' (line 165)
    range_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 44), 'range', False)
    # Calling range(args, kwargs) (line 165)
    range_call_result_629 = invoke(stypy.reporting.localization.Localization(__file__, 165, 44), range_626, *[int_627], **kwargs_628)
    
    comprehension_630 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), range_call_result_629)
    # Assigning a type to the variable 'x' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'x', comprehension_630)
    
    # Call to Vector(...): (line 165)
    # Processing the call arguments (line 165)
    float_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'float')
    float_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'float')
    float_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 30), 'float')
    # Processing the call keyword arguments (line 165)
    kwargs_624 = {}
    # Getting the type of 'Vector' (line 165)
    Vector_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'Vector', False)
    # Calling Vector(args, kwargs) (line 165)
    Vector_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 165, 13), Vector_620, *[float_621, float_622, float_623], **kwargs_624)
    
    list_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), list_631, Vector_call_result_625)
    # Assigning a type to the variable 'basis' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'basis', list_631)
    
    # Call to ortho_basis(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'basis' (line 166)
    basis_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'basis', False)
    # Getting the type of 'isect' (line 166)
    isect_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 23), 'isect', False)
    # Obtaining the member 'n' of a type (line 166)
    n_635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 23), isect_634, 'n')
    # Processing the call keyword arguments (line 166)
    kwargs_636 = {}
    # Getting the type of 'ortho_basis' (line 166)
    ortho_basis_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'ortho_basis', False)
    # Calling ortho_basis(args, kwargs) (line 166)
    ortho_basis_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 166, 4), ortho_basis_632, *[basis_633, n_635], **kwargs_636)
    
    
    # Assigning a Num to a Name (line 168):
    
    # Assigning a Num to a Name (line 168):
    float_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'float')
    # Assigning a type to the variable 'occlusion' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'occlusion', float_638)
    
    # Assigning a Tuple to a Tuple (line 169):
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 23), 'int')
    # Getting the type of 'basis' (line 169)
    basis_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'basis')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 17), basis_640, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 169, 17), getitem___641, int_639)
    
    # Assigning a type to the variable 'tuple_assignment_7' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_7', subscript_call_result_642)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 33), 'int')
    # Getting the type of 'basis' (line 169)
    basis_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 27), 'basis')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 27), basis_644, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_646 = invoke(stypy.reporting.localization.Localization(__file__, 169, 27), getitem___645, int_643)
    
    # Assigning a type to the variable 'tuple_assignment_8' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_8', subscript_call_result_646)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 43), 'int')
    # Getting the type of 'basis' (line 169)
    basis_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'basis')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 37), basis_648, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_650 = invoke(stypy.reporting.localization.Localization(__file__, 169, 37), getitem___649, int_647)
    
    # Assigning a type to the variable 'tuple_assignment_9' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_9', subscript_call_result_650)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_assignment_7' (line 169)
    tuple_assignment_7_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_7')
    # Assigning a type to the variable 'b0' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'b0', tuple_assignment_7_651)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_assignment_8' (line 169)
    tuple_assignment_8_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_8')
    # Assigning a type to the variable 'b1' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'b1', tuple_assignment_8_652)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_assignment_9' (line 169)
    tuple_assignment_9_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'tuple_assignment_9')
    # Assigning a type to the variable 'b2' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'b2', tuple_assignment_9_653)
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to Isect(...): (line 170)
    # Processing the call keyword arguments (line 170)
    kwargs_655 = {}
    # Getting the type of 'Isect' (line 170)
    Isect_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'Isect', False)
    # Calling Isect(args, kwargs) (line 170)
    Isect_call_result_656 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), Isect_654, *[], **kwargs_655)
    
    # Assigning a type to the variable 'isect' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'isect', Isect_call_result_656)
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to Ray(...): (line 171)
    # Processing the call keyword arguments (line 171)
    kwargs_658 = {}
    # Getting the type of 'Ray' (line 171)
    Ray_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 10), 'Ray', False)
    # Calling Ray(args, kwargs) (line 171)
    Ray_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 171, 10), Ray_657, *[], **kwargs_658)
    
    # Assigning a type to the variable 'ray' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'ray', Ray_call_result_659)
    
    
    # Call to xrange(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'ntheta' (line 173)
    ntheta_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'ntheta', False)
    # Processing the call keyword arguments (line 173)
    kwargs_662 = {}
    # Getting the type of 'xrange' (line 173)
    xrange_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 173)
    xrange_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), xrange_660, *[ntheta_661], **kwargs_662)
    
    # Assigning a type to the variable 'xrange_call_result_663' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'xrange_call_result_663', xrange_call_result_663)
    # Testing if the for loop is going to be iterated (line 173)
    # Testing the type of a for loop iterable (line 173)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 4), xrange_call_result_663)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 173, 4), xrange_call_result_663):
        # Getting the type of the for loop variable (line 173)
        for_loop_var_664 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 4), xrange_call_result_663)
        # Assigning a type to the variable 'j' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'j', for_loop_var_664)
        # SSA begins for a for statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'nphi' (line 174)
        nphi_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'nphi', False)
        # Processing the call keyword arguments (line 174)
        kwargs_667 = {}
        # Getting the type of 'xrange' (line 174)
        xrange_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 174)
        xrange_call_result_668 = invoke(stypy.reporting.localization.Localization(__file__, 174, 17), xrange_665, *[nphi_666], **kwargs_667)
        
        # Assigning a type to the variable 'xrange_call_result_668' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'xrange_call_result_668', xrange_call_result_668)
        # Testing if the for loop is going to be iterated (line 174)
        # Testing the type of a for loop iterable (line 174)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 174, 8), xrange_call_result_668)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 174, 8), xrange_call_result_668):
            # Getting the type of the for loop variable (line 174)
            for_loop_var_669 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 174, 8), xrange_call_result_668)
            # Assigning a type to the variable 'i' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'i', for_loop_var_669)
            # SSA begins for a for statement (line 174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 175):
            
            # Assigning a Call to a Name (line 175):
            
            # Call to sqrt(...): (line 175)
            # Processing the call arguments (line 175)
            
            # Call to random(...): (line 175)
            # Processing the call keyword arguments (line 175)
            kwargs_673 = {}
            # Getting the type of 'random' (line 175)
            random_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'random', False)
            # Obtaining the member 'random' of a type (line 175)
            random_672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), random_671, 'random')
            # Calling random(args, kwargs) (line 175)
            random_call_result_674 = invoke(stypy.reporting.localization.Localization(__file__, 175, 25), random_672, *[], **kwargs_673)
            
            # Processing the call keyword arguments (line 175)
            kwargs_675 = {}
            # Getting the type of 'sqrt' (line 175)
            sqrt_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 175)
            sqrt_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 175, 20), sqrt_670, *[random_call_result_674], **kwargs_675)
            
            # Assigning a type to the variable 'theta' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'theta', sqrt_call_result_676)
            
            # Assigning a BinOp to a Name (line 176):
            
            # Assigning a BinOp to a Name (line 176):
            float_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'float')
            float_678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 24), 'float')
            # Applying the binary operator '*' (line 176)
            result_mul_679 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 18), '*', float_677, float_678)
            
            
            # Call to random(...): (line 176)
            # Processing the call keyword arguments (line 176)
            kwargs_682 = {}
            # Getting the type of 'random' (line 176)
            random_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 49), 'random', False)
            # Obtaining the member 'random' of a type (line 176)
            random_681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 49), random_680, 'random')
            # Calling random(args, kwargs) (line 176)
            random_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 176, 49), random_681, *[], **kwargs_682)
            
            # Applying the binary operator '*' (line 176)
            result_mul_684 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 47), '*', result_mul_679, random_call_result_683)
            
            # Assigning a type to the variable 'phi' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'phi', result_mul_684)
            
            # Assigning a BinOp to a Name (line 178):
            
            # Assigning a BinOp to a Name (line 178):
            
            # Call to cos(...): (line 178)
            # Processing the call arguments (line 178)
            # Getting the type of 'phi' (line 178)
            phi_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'phi', False)
            # Processing the call keyword arguments (line 178)
            kwargs_687 = {}
            # Getting the type of 'cos' (line 178)
            cos_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'cos', False)
            # Calling cos(args, kwargs) (line 178)
            cos_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), cos_685, *[phi_686], **kwargs_687)
            
            # Getting the type of 'theta' (line 178)
            theta_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'theta')
            # Applying the binary operator '*' (line 178)
            result_mul_690 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 16), '*', cos_call_result_688, theta_689)
            
            # Assigning a type to the variable 'x' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'x', result_mul_690)
            
            # Assigning a BinOp to a Name (line 179):
            
            # Assigning a BinOp to a Name (line 179):
            
            # Call to sin(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 'phi' (line 179)
            phi_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'phi', False)
            # Processing the call keyword arguments (line 179)
            kwargs_693 = {}
            # Getting the type of 'sin' (line 179)
            sin_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'sin', False)
            # Calling sin(args, kwargs) (line 179)
            sin_call_result_694 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), sin_691, *[phi_692], **kwargs_693)
            
            # Getting the type of 'theta' (line 179)
            theta_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'theta')
            # Applying the binary operator '*' (line 179)
            result_mul_696 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 16), '*', sin_call_result_694, theta_695)
            
            # Assigning a type to the variable 'y' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'y', result_mul_696)
            
            # Assigning a Call to a Name (line 180):
            
            # Assigning a Call to a Name (line 180):
            
            # Call to sqrt(...): (line 180)
            # Processing the call arguments (line 180)
            float_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 21), 'float')
            # Getting the type of 'theta' (line 180)
            theta_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'theta', False)
            # Getting the type of 'theta' (line 180)
            theta_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 35), 'theta', False)
            # Applying the binary operator '*' (line 180)
            result_mul_701 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 27), '*', theta_699, theta_700)
            
            # Applying the binary operator '-' (line 180)
            result_sub_702 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 21), '-', float_698, result_mul_701)
            
            # Processing the call keyword arguments (line 180)
            kwargs_703 = {}
            # Getting the type of 'sqrt' (line 180)
            sqrt_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 180)
            sqrt_call_result_704 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), sqrt_697, *[result_sub_702], **kwargs_703)
            
            # Assigning a type to the variable 'z' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'z', sqrt_call_result_704)
            
            # Assigning a BinOp to a Name (line 182):
            
            # Assigning a BinOp to a Name (line 182):
            # Getting the type of 'x' (line 182)
            x_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'x')
            # Getting the type of 'b0' (line 182)
            b0_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), 'b0')
            # Obtaining the member 'x' of a type (line 182)
            x_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 21), b0_706, 'x')
            # Applying the binary operator '*' (line 182)
            result_mul_708 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 17), '*', x_705, x_707)
            
            # Getting the type of 'y' (line 182)
            y_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 28), 'y')
            # Getting the type of 'b1' (line 182)
            b1_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 32), 'b1')
            # Obtaining the member 'x' of a type (line 182)
            x_711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 32), b1_710, 'x')
            # Applying the binary operator '*' (line 182)
            result_mul_712 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 28), '*', y_709, x_711)
            
            # Applying the binary operator '+' (line 182)
            result_add_713 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 17), '+', result_mul_708, result_mul_712)
            
            # Getting the type of 'z' (line 182)
            z_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'z')
            # Getting the type of 'b2' (line 182)
            b2_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 43), 'b2')
            # Obtaining the member 'x' of a type (line 182)
            x_716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 43), b2_715, 'x')
            # Applying the binary operator '*' (line 182)
            result_mul_717 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 39), '*', z_714, x_716)
            
            # Applying the binary operator '+' (line 182)
            result_add_718 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 37), '+', result_add_713, result_mul_717)
            
            # Assigning a type to the variable 'rx' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'rx', result_add_718)
            
            # Assigning a BinOp to a Name (line 183):
            
            # Assigning a BinOp to a Name (line 183):
            # Getting the type of 'x' (line 183)
            x_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'x')
            # Getting the type of 'b0' (line 183)
            b0_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'b0')
            # Obtaining the member 'y' of a type (line 183)
            y_721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 21), b0_720, 'y')
            # Applying the binary operator '*' (line 183)
            result_mul_722 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 17), '*', x_719, y_721)
            
            # Getting the type of 'y' (line 183)
            y_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'y')
            # Getting the type of 'b1' (line 183)
            b1_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'b1')
            # Obtaining the member 'y' of a type (line 183)
            y_725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 32), b1_724, 'y')
            # Applying the binary operator '*' (line 183)
            result_mul_726 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 28), '*', y_723, y_725)
            
            # Applying the binary operator '+' (line 183)
            result_add_727 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 17), '+', result_mul_722, result_mul_726)
            
            # Getting the type of 'z' (line 183)
            z_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'z')
            # Getting the type of 'b2' (line 183)
            b2_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 43), 'b2')
            # Obtaining the member 'y' of a type (line 183)
            y_730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 43), b2_729, 'y')
            # Applying the binary operator '*' (line 183)
            result_mul_731 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 39), '*', z_728, y_730)
            
            # Applying the binary operator '+' (line 183)
            result_add_732 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 37), '+', result_add_727, result_mul_731)
            
            # Assigning a type to the variable 'ry' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'ry', result_add_732)
            
            # Assigning a BinOp to a Name (line 184):
            
            # Assigning a BinOp to a Name (line 184):
            # Getting the type of 'x' (line 184)
            x_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 17), 'x')
            # Getting the type of 'b0' (line 184)
            b0_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'b0')
            # Obtaining the member 'z' of a type (line 184)
            z_735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 21), b0_734, 'z')
            # Applying the binary operator '*' (line 184)
            result_mul_736 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 17), '*', x_733, z_735)
            
            # Getting the type of 'y' (line 184)
            y_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'y')
            # Getting the type of 'b1' (line 184)
            b1_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 32), 'b1')
            # Obtaining the member 'z' of a type (line 184)
            z_739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 32), b1_738, 'z')
            # Applying the binary operator '*' (line 184)
            result_mul_740 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 28), '*', y_737, z_739)
            
            # Applying the binary operator '+' (line 184)
            result_add_741 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 17), '+', result_mul_736, result_mul_740)
            
            # Getting the type of 'z' (line 184)
            z_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 39), 'z')
            # Getting the type of 'b2' (line 184)
            b2_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 43), 'b2')
            # Obtaining the member 'z' of a type (line 184)
            z_744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 43), b2_743, 'z')
            # Applying the binary operator '*' (line 184)
            result_mul_745 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 39), '*', z_742, z_744)
            
            # Applying the binary operator '+' (line 184)
            result_add_746 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 37), '+', result_add_741, result_mul_745)
            
            # Assigning a type to the variable 'rz' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'rz', result_add_746)
            
            # Call to reset(...): (line 185)
            # Processing the call arguments (line 185)
            # Getting the type of 'p' (line 185)
            p_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'p', False)
            # Getting the type of 'rx' (line 185)
            rx_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'rx', False)
            # Getting the type of 'ry' (line 185)
            ry_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'ry', False)
            # Getting the type of 'rz' (line 185)
            rz_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 33), 'rz', False)
            # Processing the call keyword arguments (line 185)
            kwargs_753 = {}
            # Getting the type of 'ray' (line 185)
            ray_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'ray', False)
            # Obtaining the member 'reset' of a type (line 185)
            reset_748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), ray_747, 'reset')
            # Calling reset(args, kwargs) (line 185)
            reset_call_result_754 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), reset_748, *[p_749, rx_750, ry_751, rz_752], **kwargs_753)
            
            
            # Call to reset(...): (line 187)
            # Processing the call keyword arguments (line 187)
            kwargs_757 = {}
            # Getting the type of 'isect' (line 187)
            isect_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'isect', False)
            # Obtaining the member 'reset' of a type (line 187)
            reset_756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), isect_755, 'reset')
            # Calling reset(args, kwargs) (line 187)
            reset_call_result_758 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), reset_756, *[], **kwargs_757)
            
            
            # Call to ray_sphere_intersect(...): (line 189)
            # Processing the call arguments (line 189)
            # Getting the type of 'isect' (line 189)
            isect_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'isect', False)
            # Getting the type of 'ray' (line 189)
            ray_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 40), 'ray', False)
            # Getting the type of 'sphere1' (line 189)
            sphere1_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 45), 'sphere1', False)
            # Processing the call keyword arguments (line 189)
            kwargs_763 = {}
            # Getting the type of 'ray_sphere_intersect' (line 189)
            ray_sphere_intersect_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'ray_sphere_intersect', False)
            # Calling ray_sphere_intersect(args, kwargs) (line 189)
            ray_sphere_intersect_call_result_764 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), ray_sphere_intersect_759, *[isect_760, ray_761, sphere1_762], **kwargs_763)
            
            
            # Call to ray_sphere_intersect(...): (line 190)
            # Processing the call arguments (line 190)
            # Getting the type of 'isect' (line 190)
            isect_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'isect', False)
            # Getting the type of 'ray' (line 190)
            ray_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 40), 'ray', False)
            # Getting the type of 'sphere2' (line 190)
            sphere2_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 45), 'sphere2', False)
            # Processing the call keyword arguments (line 190)
            kwargs_769 = {}
            # Getting the type of 'ray_sphere_intersect' (line 190)
            ray_sphere_intersect_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'ray_sphere_intersect', False)
            # Calling ray_sphere_intersect(args, kwargs) (line 190)
            ray_sphere_intersect_call_result_770 = invoke(stypy.reporting.localization.Localization(__file__, 190, 12), ray_sphere_intersect_765, *[isect_766, ray_767, sphere2_768], **kwargs_769)
            
            
            # Call to ray_sphere_intersect(...): (line 191)
            # Processing the call arguments (line 191)
            # Getting the type of 'isect' (line 191)
            isect_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 33), 'isect', False)
            # Getting the type of 'ray' (line 191)
            ray_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 40), 'ray', False)
            # Getting the type of 'sphere3' (line 191)
            sphere3_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 45), 'sphere3', False)
            # Processing the call keyword arguments (line 191)
            kwargs_775 = {}
            # Getting the type of 'ray_sphere_intersect' (line 191)
            ray_sphere_intersect_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'ray_sphere_intersect', False)
            # Calling ray_sphere_intersect(args, kwargs) (line 191)
            ray_sphere_intersect_call_result_776 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), ray_sphere_intersect_771, *[isect_772, ray_773, sphere3_774], **kwargs_775)
            
            
            # Call to ray_plane_intersect(...): (line 192)
            # Processing the call arguments (line 192)
            # Getting the type of 'isect' (line 192)
            isect_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 32), 'isect', False)
            # Getting the type of 'ray' (line 192)
            ray_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 39), 'ray', False)
            # Getting the type of 'plane' (line 192)
            plane_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'plane', False)
            # Processing the call keyword arguments (line 192)
            kwargs_781 = {}
            # Getting the type of 'ray_plane_intersect' (line 192)
            ray_plane_intersect_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'ray_plane_intersect', False)
            # Calling ray_plane_intersect(args, kwargs) (line 192)
            ray_plane_intersect_call_result_782 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), ray_plane_intersect_777, *[isect_778, ray_779, plane_780], **kwargs_781)
            
            # Getting the type of 'isect' (line 194)
            isect_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'isect')
            # Obtaining the member 'hit' of a type (line 194)
            hit_784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), isect_783, 'hit')
            # Testing if the type of an if condition is none (line 194)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 194, 12), hit_784):
                pass
            else:
                
                # Testing the type of an if condition (line 194)
                if_condition_785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 12), hit_784)
                # Assigning a type to the variable 'if_condition_785' (line 194)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'if_condition_785', if_condition_785)
                # SSA begins for if statement (line 194)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'occlusion' (line 195)
                occlusion_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'occlusion')
                float_787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 29), 'float')
                # Applying the binary operator '+=' (line 195)
                result_iadd_788 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 16), '+=', occlusion_786, float_787)
                # Assigning a type to the variable 'occlusion' (line 195)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'occlusion', result_iadd_788)
                
                # SSA join for if statement (line 194)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a BinOp to a Name (line 197):
    
    # Assigning a BinOp to a Name (line 197):
    # Getting the type of 'ntheta' (line 197)
    ntheta_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'ntheta')
    # Getting the type of 'nphi' (line 197)
    nphi_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'nphi')
    # Applying the binary operator '*' (line 197)
    result_mul_791 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 17), '*', ntheta_789, nphi_790)
    
    # Getting the type of 'occlusion' (line 197)
    occlusion_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 33), 'occlusion')
    # Applying the binary operator '-' (line 197)
    result_sub_793 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 17), '-', result_mul_791, occlusion_792)
    
    
    # Call to float(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'ntheta' (line 197)
    ntheta_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 52), 'ntheta', False)
    # Getting the type of 'nphi' (line 197)
    nphi_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 61), 'nphi', False)
    # Applying the binary operator '*' (line 197)
    result_mul_797 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 52), '*', ntheta_795, nphi_796)
    
    # Processing the call keyword arguments (line 197)
    kwargs_798 = {}
    # Getting the type of 'float' (line 197)
    float_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 46), 'float', False)
    # Calling float(args, kwargs) (line 197)
    float_call_result_799 = invoke(stypy.reporting.localization.Localization(__file__, 197, 46), float_794, *[result_mul_797], **kwargs_798)
    
    # Applying the binary operator 'div' (line 197)
    result_div_800 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 16), 'div', result_sub_793, float_call_result_799)
    
    # Assigning a type to the variable 'occlusion' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'occlusion', result_div_800)
    
    # Multiple assignment of 3 elements.
    
    # Assigning a Name to a Attribute (line 198):
    # Getting the type of 'occlusion' (line 198)
    occlusion_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'occlusion')
    # Getting the type of 'col' (line 198)
    col_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'col')
    # Setting the type of the member 'z' of a type (line 198)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 20), col_802, 'z', occlusion_801)
    
    # Assigning a Attribute to a Attribute (line 198):
    # Getting the type of 'col' (line 198)
    col_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'col')
    # Obtaining the member 'z' of a type (line 198)
    z_804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 20), col_803, 'z')
    # Getting the type of 'col' (line 198)
    col_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'col')
    # Setting the type of the member 'y' of a type (line 198)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), col_805, 'y', z_804)
    
    # Assigning a Attribute to a Attribute (line 198):
    # Getting the type of 'col' (line 198)
    col_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'col')
    # Obtaining the member 'y' of a type (line 198)
    y_807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), col_806, 'y')
    # Getting the type of 'col' (line 198)
    col_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'col')
    # Setting the type of the member 'x' of a type (line 198)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 4), col_808, 'x', y_807)
    
    # ################# End of 'ambient_occlusion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ambient_occlusion' in the type store
    # Getting the type of 'stypy_return_type' (line 155)
    stypy_return_type_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ambient_occlusion'
    return stypy_return_type_809

# Assigning a type to the variable 'ambient_occlusion' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'ambient_occlusion', ambient_occlusion)

@norecursion
def clamp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'clamp'
    module_type_store = module_type_store.open_function_context('clamp', 201, 0, False)
    
    # Passed parameters checking function
    clamp.stypy_localization = localization
    clamp.stypy_type_of_self = None
    clamp.stypy_type_store = module_type_store
    clamp.stypy_function_name = 'clamp'
    clamp.stypy_param_names_list = ['f']
    clamp.stypy_varargs_param_name = None
    clamp.stypy_kwargs_param_name = None
    clamp.stypy_call_defaults = defaults
    clamp.stypy_call_varargs = varargs
    clamp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clamp', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clamp', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clamp(...)' code ##################

    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to int(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'f' (line 202)
    f_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'f', False)
    float_812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'float')
    # Applying the binary operator '*' (line 202)
    result_mul_813 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 12), '*', f_811, float_812)
    
    # Processing the call keyword arguments (line 202)
    kwargs_814 = {}
    # Getting the type of 'int' (line 202)
    int_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'int', False)
    # Calling int(args, kwargs) (line 202)
    int_call_result_815 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), int_810, *[result_mul_813], **kwargs_814)
    
    # Assigning a type to the variable 'i' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'i', int_call_result_815)
    
    # Getting the type of 'i' (line 203)
    i_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 7), 'i')
    int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 11), 'int')
    # Applying the binary operator '<' (line 203)
    result_lt_818 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 7), '<', i_816, int_817)
    
    # Testing if the type of an if condition is none (line 203)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 4), result_lt_818):
        pass
    else:
        
        # Testing the type of an if condition (line 203)
        if_condition_819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 4), result_lt_818)
        # Assigning a type to the variable 'if_condition_819' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'if_condition_819', if_condition_819)
        # SSA begins for if statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 204):
        
        # Assigning a Num to a Name (line 204):
        int_820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 12), 'int')
        # Assigning a type to the variable 'i' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'i', int_820)
        # SSA join for if statement (line 203)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'i' (line 205)
    i_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 7), 'i')
    int_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 11), 'int')
    # Applying the binary operator '>' (line 205)
    result_gt_823 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 7), '>', i_821, int_822)
    
    # Testing if the type of an if condition is none (line 205)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 205, 4), result_gt_823):
        pass
    else:
        
        # Testing the type of an if condition (line 205)
        if_condition_824 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 4), result_gt_823)
        # Assigning a type to the variable 'if_condition_824' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'if_condition_824', if_condition_824)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 206):
        
        # Assigning a Num to a Name (line 206):
        int_825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 12), 'int')
        # Assigning a type to the variable 'i' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'i', int_825)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'i' (line 207)
    i_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'i')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type', i_826)
    
    # ################# End of 'clamp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clamp' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_827)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clamp'
    return stypy_return_type_827

# Assigning a type to the variable 'clamp' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'clamp', clamp)

@norecursion
def render(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'render'
    module_type_store = module_type_store.open_function_context('render', 210, 0, False)
    
    # Passed parameters checking function
    render.stypy_localization = localization
    render.stypy_type_of_self = None
    render.stypy_type_store = module_type_store
    render.stypy_function_name = 'render'
    render.stypy_param_names_list = ['w', 'h', 'nsubsamples']
    render.stypy_varargs_param_name = None
    render.stypy_kwargs_param_name = None
    render.stypy_call_defaults = defaults
    render.stypy_call_varargs = varargs
    render.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'render', ['w', 'h', 'nsubsamples'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'render', localization, ['w', 'h', 'nsubsamples'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'render(...)' code ##################

    
    # Assigning a BinOp to a Name (line 211):
    
    # Assigning a BinOp to a Name (line 211):
    
    # Obtaining an instance of the builtin type 'list' (line 211)
    list_828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 211)
    # Adding element type (line 211)
    int_829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 10), list_828, int_829)
    
    # Getting the type of 'WIDTH' (line 211)
    WIDTH_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 17), 'WIDTH')
    # Getting the type of 'HEIGHT' (line 211)
    HEIGHT_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'HEIGHT')
    # Applying the binary operator '*' (line 211)
    result_mul_832 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 17), '*', WIDTH_830, HEIGHT_831)
    
    int_833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 34), 'int')
    # Applying the binary operator '*' (line 211)
    result_mul_834 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 32), '*', result_mul_832, int_833)
    
    # Applying the binary operator '*' (line 211)
    result_mul_835 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 10), '*', list_828, result_mul_834)
    
    # Assigning a type to the variable 'img' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'img', result_mul_835)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to float(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'nsubsamples' (line 213)
    nsubsamples_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'nsubsamples', False)
    # Processing the call keyword arguments (line 213)
    kwargs_838 = {}
    # Getting the type of 'float' (line 213)
    float_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'float', False)
    # Calling float(args, kwargs) (line 213)
    float_call_result_839 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), float_836, *[nsubsamples_837], **kwargs_838)
    
    # Assigning a type to the variable 'nsubs' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'nsubs', float_call_result_839)
    
    # Assigning a BinOp to a Name (line 214):
    
    # Assigning a BinOp to a Name (line 214):
    # Getting the type of 'nsubs' (line 214)
    nsubs_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'nsubs')
    # Getting the type of 'nsubs' (line 214)
    nsubs_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 26), 'nsubs')
    # Applying the binary operator '*' (line 214)
    result_mul_842 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 18), '*', nsubs_840, nsubs_841)
    
    # Assigning a type to the variable 'nsubs_nsubs' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'nsubs_nsubs', result_mul_842)
    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to Vector(...): (line 216)
    # Processing the call arguments (line 216)
    float_844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'float')
    float_845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 21), 'float')
    float_846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 26), 'float')
    # Processing the call keyword arguments (line 216)
    kwargs_847 = {}
    # Getting the type of 'Vector' (line 216)
    Vector_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 9), 'Vector', False)
    # Calling Vector(args, kwargs) (line 216)
    Vector_call_result_848 = invoke(stypy.reporting.localization.Localization(__file__, 216, 9), Vector_843, *[float_844, float_845, float_846], **kwargs_847)
    
    # Assigning a type to the variable 'v0' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'v0', Vector_call_result_848)
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to Vector(...): (line 217)
    # Processing the call arguments (line 217)
    float_850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 17), 'float')
    float_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'float')
    float_852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 27), 'float')
    # Processing the call keyword arguments (line 217)
    kwargs_853 = {}
    # Getting the type of 'Vector' (line 217)
    Vector_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 10), 'Vector', False)
    # Calling Vector(args, kwargs) (line 217)
    Vector_call_result_854 = invoke(stypy.reporting.localization.Localization(__file__, 217, 10), Vector_849, *[float_850, float_851, float_852], **kwargs_853)
    
    # Assigning a type to the variable 'col' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'col', Vector_call_result_854)
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to Isect(...): (line 218)
    # Processing the call keyword arguments (line 218)
    kwargs_856 = {}
    # Getting the type of 'Isect' (line 218)
    Isect_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'Isect', False)
    # Calling Isect(args, kwargs) (line 218)
    Isect_call_result_857 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), Isect_855, *[], **kwargs_856)
    
    # Assigning a type to the variable 'isect' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'isect', Isect_call_result_857)
    
    # Assigning a Call to a Name (line 219):
    
    # Assigning a Call to a Name (line 219):
    
    # Call to Ray(...): (line 219)
    # Processing the call keyword arguments (line 219)
    kwargs_859 = {}
    # Getting the type of 'Ray' (line 219)
    Ray_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 10), 'Ray', False)
    # Calling Ray(args, kwargs) (line 219)
    Ray_call_result_860 = invoke(stypy.reporting.localization.Localization(__file__, 219, 10), Ray_858, *[], **kwargs_859)
    
    # Assigning a type to the variable 'ray' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'ray', Ray_call_result_860)
    
    
    # Call to xrange(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'h' (line 221)
    h_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'h', False)
    # Processing the call keyword arguments (line 221)
    kwargs_863 = {}
    # Getting the type of 'xrange' (line 221)
    xrange_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 221)
    xrange_call_result_864 = invoke(stypy.reporting.localization.Localization(__file__, 221, 13), xrange_861, *[h_862], **kwargs_863)
    
    # Assigning a type to the variable 'xrange_call_result_864' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'xrange_call_result_864', xrange_call_result_864)
    # Testing if the for loop is going to be iterated (line 221)
    # Testing the type of a for loop iterable (line 221)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 221, 4), xrange_call_result_864)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 221, 4), xrange_call_result_864):
        # Getting the type of the for loop variable (line 221)
        for_loop_var_865 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 221, 4), xrange_call_result_864)
        # Assigning a type to the variable 'y' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'y', for_loop_var_865)
        # SSA begins for a for statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'w' (line 222)
        w_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'w', False)
        # Processing the call keyword arguments (line 222)
        kwargs_868 = {}
        # Getting the type of 'xrange' (line 222)
        xrange_866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 222)
        xrange_call_result_869 = invoke(stypy.reporting.localization.Localization(__file__, 222, 17), xrange_866, *[w_867], **kwargs_868)
        
        # Assigning a type to the variable 'xrange_call_result_869' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'xrange_call_result_869', xrange_call_result_869)
        # Testing if the for loop is going to be iterated (line 222)
        # Testing the type of a for loop iterable (line 222)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 8), xrange_call_result_869)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 222, 8), xrange_call_result_869):
            # Getting the type of the for loop variable (line 222)
            for_loop_var_870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 8), xrange_call_result_869)
            # Assigning a type to the variable 'x' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'x', for_loop_var_870)
            # SSA begins for a for statement (line 222)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 223):
            
            # Assigning a Num to a Name (line 223):
            float_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 17), 'float')
            # Assigning a type to the variable 'fr' (line 223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'fr', float_871)
            
            # Assigning a Num to a Name (line 224):
            
            # Assigning a Num to a Name (line 224):
            float_872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 17), 'float')
            # Assigning a type to the variable 'fg' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'fg', float_872)
            
            # Assigning a Num to a Name (line 225):
            
            # Assigning a Num to a Name (line 225):
            float_873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 17), 'float')
            # Assigning a type to the variable 'fb' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'fb', float_873)
            
            
            # Call to xrange(...): (line 226)
            # Processing the call arguments (line 226)
            # Getting the type of 'nsubsamples' (line 226)
            nsubsamples_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'nsubsamples', False)
            # Processing the call keyword arguments (line 226)
            kwargs_876 = {}
            # Getting the type of 'xrange' (line 226)
            xrange_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 226)
            xrange_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 226, 21), xrange_874, *[nsubsamples_875], **kwargs_876)
            
            # Assigning a type to the variable 'xrange_call_result_877' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'xrange_call_result_877', xrange_call_result_877)
            # Testing if the for loop is going to be iterated (line 226)
            # Testing the type of a for loop iterable (line 226)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 12), xrange_call_result_877)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 226, 12), xrange_call_result_877):
                # Getting the type of the for loop variable (line 226)
                for_loop_var_878 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 12), xrange_call_result_877)
                # Assigning a type to the variable 'v' (line 226)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'v', for_loop_var_878)
                # SSA begins for a for statement (line 226)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to xrange(...): (line 227)
                # Processing the call arguments (line 227)
                # Getting the type of 'nsubsamples' (line 227)
                nsubsamples_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 'nsubsamples', False)
                # Processing the call keyword arguments (line 227)
                kwargs_881 = {}
                # Getting the type of 'xrange' (line 227)
                xrange_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'xrange', False)
                # Calling xrange(args, kwargs) (line 227)
                xrange_call_result_882 = invoke(stypy.reporting.localization.Localization(__file__, 227, 25), xrange_879, *[nsubsamples_880], **kwargs_881)
                
                # Assigning a type to the variable 'xrange_call_result_882' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'xrange_call_result_882', xrange_call_result_882)
                # Testing if the for loop is going to be iterated (line 227)
                # Testing the type of a for loop iterable (line 227)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 227, 16), xrange_call_result_882)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 227, 16), xrange_call_result_882):
                    # Getting the type of the for loop variable (line 227)
                    for_loop_var_883 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 227, 16), xrange_call_result_882)
                    # Assigning a type to the variable 'u' (line 227)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'u', for_loop_var_883)
                    # SSA begins for a for statement (line 227)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a BinOp to a Name (line 228):
                    
                    # Assigning a BinOp to a Name (line 228):
                    # Getting the type of 'x' (line 228)
                    x_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'x')
                    # Getting the type of 'u' (line 228)
                    u_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 31), 'u')
                    
                    # Call to float(...): (line 228)
                    # Processing the call arguments (line 228)
                    # Getting the type of 'nsubsamples' (line 228)
                    nsubsamples_887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 41), 'nsubsamples', False)
                    # Processing the call keyword arguments (line 228)
                    kwargs_888 = {}
                    # Getting the type of 'float' (line 228)
                    float_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 35), 'float', False)
                    # Calling float(args, kwargs) (line 228)
                    float_call_result_889 = invoke(stypy.reporting.localization.Localization(__file__, 228, 35), float_886, *[nsubsamples_887], **kwargs_888)
                    
                    # Applying the binary operator 'div' (line 228)
                    result_div_890 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 31), 'div', u_885, float_call_result_889)
                    
                    # Applying the binary operator '+' (line 228)
                    result_add_891 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 26), '+', x_884, result_div_890)
                    
                    # Getting the type of 'w' (line 228)
                    w_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 58), 'w')
                    float_893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 62), 'float')
                    # Applying the binary operator 'div' (line 228)
                    result_div_894 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 58), 'div', w_892, float_893)
                    
                    # Applying the binary operator '-' (line 228)
                    result_sub_895 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 55), '-', result_add_891, result_div_894)
                    
                    # Getting the type of 'w' (line 228)
                    w_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 71), 'w')
                    float_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 75), 'float')
                    # Applying the binary operator 'div' (line 228)
                    result_div_898 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 71), 'div', w_896, float_897)
                    
                    # Applying the binary operator 'div' (line 228)
                    result_div_899 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 25), 'div', result_sub_895, result_div_898)
                    
                    # Assigning a type to the variable 'px' (line 228)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'px', result_div_899)
                    
                    # Assigning a BinOp to a Name (line 229):
                    
                    # Assigning a BinOp to a Name (line 229):
                    
                    # Getting the type of 'y' (line 229)
                    y_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'y')
                    # Getting the type of 'v' (line 229)
                    v_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 32), 'v')
                    
                    # Call to float(...): (line 229)
                    # Processing the call arguments (line 229)
                    # Getting the type of 'nsubsamples' (line 229)
                    nsubsamples_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 42), 'nsubsamples', False)
                    # Processing the call keyword arguments (line 229)
                    kwargs_904 = {}
                    # Getting the type of 'float' (line 229)
                    float_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 36), 'float', False)
                    # Calling float(args, kwargs) (line 229)
                    float_call_result_905 = invoke(stypy.reporting.localization.Localization(__file__, 229, 36), float_902, *[nsubsamples_903], **kwargs_904)
                    
                    # Applying the binary operator 'div' (line 229)
                    result_div_906 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 32), 'div', v_901, float_call_result_905)
                    
                    # Applying the binary operator '+' (line 229)
                    result_add_907 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 27), '+', y_900, result_div_906)
                    
                    # Getting the type of 'h' (line 229)
                    h_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 59), 'h')
                    float_909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 63), 'float')
                    # Applying the binary operator 'div' (line 229)
                    result_div_910 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 59), 'div', h_908, float_909)
                    
                    # Applying the binary operator '-' (line 229)
                    result_sub_911 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 56), '-', result_add_907, result_div_910)
                    
                    # Applying the 'usub' unary operator (line 229)
                    result___neg___912 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 25), 'usub', result_sub_911)
                    
                    # Getting the type of 'h' (line 229)
                    h_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 72), 'h')
                    float_914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 76), 'float')
                    # Applying the binary operator 'div' (line 229)
                    result_div_915 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 72), 'div', h_913, float_914)
                    
                    # Applying the binary operator 'div' (line 229)
                    result_div_916 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 25), 'div', result___neg___912, result_div_915)
                    
                    # Assigning a type to the variable 'py' (line 229)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'py', result_div_916)
                    
                    # Call to reset(...): (line 230)
                    # Processing the call arguments (line 230)
                    # Getting the type of 'v0' (line 230)
                    v0_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 'v0', False)
                    # Getting the type of 'px' (line 230)
                    px_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 34), 'px', False)
                    # Getting the type of 'py' (line 230)
                    py_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'py', False)
                    float_922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 42), 'float')
                    # Processing the call keyword arguments (line 230)
                    kwargs_923 = {}
                    # Getting the type of 'ray' (line 230)
                    ray_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'ray', False)
                    # Obtaining the member 'reset' of a type (line 230)
                    reset_918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 20), ray_917, 'reset')
                    # Calling reset(args, kwargs) (line 230)
                    reset_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 230, 20), reset_918, *[v0_919, px_920, py_921, float_922], **kwargs_923)
                    
                    
                    # Call to vnormalize(...): (line 231)
                    # Processing the call arguments (line 231)
                    # Getting the type of 'ray' (line 231)
                    ray_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 31), 'ray', False)
                    # Obtaining the member 'dir' of a type (line 231)
                    dir_927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 31), ray_926, 'dir')
                    # Processing the call keyword arguments (line 231)
                    kwargs_928 = {}
                    # Getting the type of 'vnormalize' (line 231)
                    vnormalize_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'vnormalize', False)
                    # Calling vnormalize(args, kwargs) (line 231)
                    vnormalize_call_result_929 = invoke(stypy.reporting.localization.Localization(__file__, 231, 20), vnormalize_925, *[dir_927], **kwargs_928)
                    
                    
                    # Call to reset(...): (line 233)
                    # Processing the call keyword arguments (line 233)
                    kwargs_932 = {}
                    # Getting the type of 'isect' (line 233)
                    isect_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'isect', False)
                    # Obtaining the member 'reset' of a type (line 233)
                    reset_931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), isect_930, 'reset')
                    # Calling reset(args, kwargs) (line 233)
                    reset_call_result_933 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), reset_931, *[], **kwargs_932)
                    
                    
                    # Call to ray_sphere_intersect(...): (line 235)
                    # Processing the call arguments (line 235)
                    # Getting the type of 'isect' (line 235)
                    isect_935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 41), 'isect', False)
                    # Getting the type of 'ray' (line 235)
                    ray_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 48), 'ray', False)
                    # Getting the type of 'sphere1' (line 235)
                    sphere1_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 53), 'sphere1', False)
                    # Processing the call keyword arguments (line 235)
                    kwargs_938 = {}
                    # Getting the type of 'ray_sphere_intersect' (line 235)
                    ray_sphere_intersect_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'ray_sphere_intersect', False)
                    # Calling ray_sphere_intersect(args, kwargs) (line 235)
                    ray_sphere_intersect_call_result_939 = invoke(stypy.reporting.localization.Localization(__file__, 235, 20), ray_sphere_intersect_934, *[isect_935, ray_936, sphere1_937], **kwargs_938)
                    
                    
                    # Call to ray_sphere_intersect(...): (line 236)
                    # Processing the call arguments (line 236)
                    # Getting the type of 'isect' (line 236)
                    isect_941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 41), 'isect', False)
                    # Getting the type of 'ray' (line 236)
                    ray_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 48), 'ray', False)
                    # Getting the type of 'sphere2' (line 236)
                    sphere2_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 53), 'sphere2', False)
                    # Processing the call keyword arguments (line 236)
                    kwargs_944 = {}
                    # Getting the type of 'ray_sphere_intersect' (line 236)
                    ray_sphere_intersect_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'ray_sphere_intersect', False)
                    # Calling ray_sphere_intersect(args, kwargs) (line 236)
                    ray_sphere_intersect_call_result_945 = invoke(stypy.reporting.localization.Localization(__file__, 236, 20), ray_sphere_intersect_940, *[isect_941, ray_942, sphere2_943], **kwargs_944)
                    
                    
                    # Call to ray_sphere_intersect(...): (line 237)
                    # Processing the call arguments (line 237)
                    # Getting the type of 'isect' (line 237)
                    isect_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 41), 'isect', False)
                    # Getting the type of 'ray' (line 237)
                    ray_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 48), 'ray', False)
                    # Getting the type of 'sphere3' (line 237)
                    sphere3_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 53), 'sphere3', False)
                    # Processing the call keyword arguments (line 237)
                    kwargs_950 = {}
                    # Getting the type of 'ray_sphere_intersect' (line 237)
                    ray_sphere_intersect_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'ray_sphere_intersect', False)
                    # Calling ray_sphere_intersect(args, kwargs) (line 237)
                    ray_sphere_intersect_call_result_951 = invoke(stypy.reporting.localization.Localization(__file__, 237, 20), ray_sphere_intersect_946, *[isect_947, ray_948, sphere3_949], **kwargs_950)
                    
                    
                    # Call to ray_plane_intersect(...): (line 238)
                    # Processing the call arguments (line 238)
                    # Getting the type of 'isect' (line 238)
                    isect_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 40), 'isect', False)
                    # Getting the type of 'ray' (line 238)
                    ray_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'ray', False)
                    # Getting the type of 'plane' (line 238)
                    plane_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 52), 'plane', False)
                    # Processing the call keyword arguments (line 238)
                    kwargs_956 = {}
                    # Getting the type of 'ray_plane_intersect' (line 238)
                    ray_plane_intersect_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'ray_plane_intersect', False)
                    # Calling ray_plane_intersect(args, kwargs) (line 238)
                    ray_plane_intersect_call_result_957 = invoke(stypy.reporting.localization.Localization(__file__, 238, 20), ray_plane_intersect_952, *[isect_953, ray_954, plane_955], **kwargs_956)
                    
                    # Getting the type of 'isect' (line 240)
                    isect_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'isect')
                    # Obtaining the member 'hit' of a type (line 240)
                    hit_959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 23), isect_958, 'hit')
                    # Testing if the type of an if condition is none (line 240)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 240, 20), hit_959):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 240)
                        if_condition_960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 20), hit_959)
                        # Assigning a type to the variable 'if_condition_960' (line 240)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'if_condition_960', if_condition_960)
                        # SSA begins for if statement (line 240)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to ambient_occlusion(...): (line 241)
                        # Processing the call arguments (line 241)
                        # Getting the type of 'col' (line 241)
                        col_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 42), 'col', False)
                        # Getting the type of 'isect' (line 241)
                        isect_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 47), 'isect', False)
                        # Processing the call keyword arguments (line 241)
                        kwargs_964 = {}
                        # Getting the type of 'ambient_occlusion' (line 241)
                        ambient_occlusion_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'ambient_occlusion', False)
                        # Calling ambient_occlusion(args, kwargs) (line 241)
                        ambient_occlusion_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 241, 24), ambient_occlusion_961, *[col_962, isect_963], **kwargs_964)
                        
                        
                        # Getting the type of 'fr' (line 242)
                        fr_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'fr')
                        # Getting the type of 'col' (line 242)
                        col_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'col')
                        # Obtaining the member 'x' of a type (line 242)
                        x_968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 30), col_967, 'x')
                        # Applying the binary operator '+=' (line 242)
                        result_iadd_969 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 24), '+=', fr_966, x_968)
                        # Assigning a type to the variable 'fr' (line 242)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'fr', result_iadd_969)
                        
                        
                        # Getting the type of 'fg' (line 243)
                        fg_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'fg')
                        # Getting the type of 'col' (line 243)
                        col_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'col')
                        # Obtaining the member 'y' of a type (line 243)
                        y_972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 30), col_971, 'y')
                        # Applying the binary operator '+=' (line 243)
                        result_iadd_973 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 24), '+=', fg_970, y_972)
                        # Assigning a type to the variable 'fg' (line 243)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'fg', result_iadd_973)
                        
                        
                        # Getting the type of 'fb' (line 244)
                        fb_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'fb')
                        # Getting the type of 'col' (line 244)
                        col_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 30), 'col')
                        # Obtaining the member 'z' of a type (line 244)
                        z_976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 30), col_975, 'z')
                        # Applying the binary operator '+=' (line 244)
                        result_iadd_977 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 24), '+=', fb_974, z_976)
                        # Assigning a type to the variable 'fb' (line 244)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'fb', result_iadd_977)
                        
                        # SSA join for if statement (line 240)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Subscript (line 246):
            
            # Assigning a Call to a Subscript (line 246):
            
            # Call to clamp(...): (line 246)
            # Processing the call arguments (line 246)
            # Getting the type of 'fr' (line 246)
            fr_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'fr', False)
            # Getting the type of 'nsubs_nsubs' (line 246)
            nsubs_nsubs_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 50), 'nsubs_nsubs', False)
            # Applying the binary operator 'div' (line 246)
            result_div_981 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 45), 'div', fr_979, nsubs_nsubs_980)
            
            # Processing the call keyword arguments (line 246)
            kwargs_982 = {}
            # Getting the type of 'clamp' (line 246)
            clamp_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'clamp', False)
            # Calling clamp(args, kwargs) (line 246)
            clamp_call_result_983 = invoke(stypy.reporting.localization.Localization(__file__, 246, 39), clamp_978, *[result_div_981], **kwargs_982)
            
            # Getting the type of 'img' (line 246)
            img_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'img')
            int_985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 16), 'int')
            # Getting the type of 'y' (line 246)
            y_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'y')
            # Getting the type of 'w' (line 246)
            w_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'w')
            # Applying the binary operator '*' (line 246)
            result_mul_988 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 21), '*', y_986, w_987)
            
            # Getting the type of 'x' (line 246)
            x_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'x')
            # Applying the binary operator '+' (line 246)
            result_add_990 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 21), '+', result_mul_988, x_989)
            
            # Applying the binary operator '*' (line 246)
            result_mul_991 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 16), '*', int_985, result_add_990)
            
            int_992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 34), 'int')
            # Applying the binary operator '+' (line 246)
            result_add_993 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 16), '+', result_mul_991, int_992)
            
            # Storing an element on a container (line 246)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 12), img_984, (result_add_993, clamp_call_result_983))
            
            # Assigning a Call to a Subscript (line 247):
            
            # Assigning a Call to a Subscript (line 247):
            
            # Call to clamp(...): (line 247)
            # Processing the call arguments (line 247)
            # Getting the type of 'fg' (line 247)
            fg_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 45), 'fg', False)
            # Getting the type of 'nsubs_nsubs' (line 247)
            nsubs_nsubs_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 50), 'nsubs_nsubs', False)
            # Applying the binary operator 'div' (line 247)
            result_div_997 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 45), 'div', fg_995, nsubs_nsubs_996)
            
            # Processing the call keyword arguments (line 247)
            kwargs_998 = {}
            # Getting the type of 'clamp' (line 247)
            clamp_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 39), 'clamp', False)
            # Calling clamp(args, kwargs) (line 247)
            clamp_call_result_999 = invoke(stypy.reporting.localization.Localization(__file__, 247, 39), clamp_994, *[result_div_997], **kwargs_998)
            
            # Getting the type of 'img' (line 247)
            img_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'img')
            int_1001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 16), 'int')
            # Getting the type of 'y' (line 247)
            y_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'y')
            # Getting the type of 'w' (line 247)
            w_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'w')
            # Applying the binary operator '*' (line 247)
            result_mul_1004 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 21), '*', y_1002, w_1003)
            
            # Getting the type of 'x' (line 247)
            x_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'x')
            # Applying the binary operator '+' (line 247)
            result_add_1006 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 21), '+', result_mul_1004, x_1005)
            
            # Applying the binary operator '*' (line 247)
            result_mul_1007 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 16), '*', int_1001, result_add_1006)
            
            int_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 34), 'int')
            # Applying the binary operator '+' (line 247)
            result_add_1009 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 16), '+', result_mul_1007, int_1008)
            
            # Storing an element on a container (line 247)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 12), img_1000, (result_add_1009, clamp_call_result_999))
            
            # Assigning a Call to a Subscript (line 248):
            
            # Assigning a Call to a Subscript (line 248):
            
            # Call to clamp(...): (line 248)
            # Processing the call arguments (line 248)
            # Getting the type of 'fb' (line 248)
            fb_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 45), 'fb', False)
            # Getting the type of 'nsubs_nsubs' (line 248)
            nsubs_nsubs_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 50), 'nsubs_nsubs', False)
            # Applying the binary operator 'div' (line 248)
            result_div_1013 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 45), 'div', fb_1011, nsubs_nsubs_1012)
            
            # Processing the call keyword arguments (line 248)
            kwargs_1014 = {}
            # Getting the type of 'clamp' (line 248)
            clamp_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 39), 'clamp', False)
            # Calling clamp(args, kwargs) (line 248)
            clamp_call_result_1015 = invoke(stypy.reporting.localization.Localization(__file__, 248, 39), clamp_1010, *[result_div_1013], **kwargs_1014)
            
            # Getting the type of 'img' (line 248)
            img_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'img')
            int_1017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 16), 'int')
            # Getting the type of 'y' (line 248)
            y_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'y')
            # Getting the type of 'w' (line 248)
            w_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'w')
            # Applying the binary operator '*' (line 248)
            result_mul_1020 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 21), '*', y_1018, w_1019)
            
            # Getting the type of 'x' (line 248)
            x_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 29), 'x')
            # Applying the binary operator '+' (line 248)
            result_add_1022 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 21), '+', result_mul_1020, x_1021)
            
            # Applying the binary operator '*' (line 248)
            result_mul_1023 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), '*', int_1017, result_add_1022)
            
            int_1024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 34), 'int')
            # Applying the binary operator '+' (line 248)
            result_add_1025 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), '+', result_mul_1023, int_1024)
            
            # Storing an element on a container (line 248)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 12), img_1016, (result_add_1025, clamp_call_result_1015))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'img' (line 250)
    img_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'img')
    # Assigning a type to the variable 'stypy_return_type' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type', img_1026)
    
    # ################# End of 'render(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'render' in the type store
    # Getting the type of 'stypy_return_type' (line 210)
    stypy_return_type_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'render'
    return stypy_return_type_1027

# Assigning a type to the variable 'render' (line 210)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'render', render)

@norecursion
def init_scene(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'init_scene'
    module_type_store = module_type_store.open_function_context('init_scene', 253, 0, False)
    
    # Passed parameters checking function
    init_scene.stypy_localization = localization
    init_scene.stypy_type_of_self = None
    init_scene.stypy_type_store = module_type_store
    init_scene.stypy_function_name = 'init_scene'
    init_scene.stypy_param_names_list = []
    init_scene.stypy_varargs_param_name = None
    init_scene.stypy_kwargs_param_name = None
    init_scene.stypy_call_defaults = defaults
    init_scene.stypy_call_varargs = varargs
    init_scene.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'init_scene', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'init_scene', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'init_scene(...)' code ##################

    # Marking variables as global (line 254)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 254, 4), 'sphere1')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 254, 4), 'sphere2')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 254, 4), 'sphere3')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 254, 4), 'plane')
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to Sphere(...): (line 255)
    # Processing the call arguments (line 255)
    
    # Call to Vector(...): (line 255)
    # Processing the call arguments (line 255)
    float_1030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 28), 'float')
    float_1031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 34), 'float')
    float_1032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 39), 'float')
    # Processing the call keyword arguments (line 255)
    kwargs_1033 = {}
    # Getting the type of 'Vector' (line 255)
    Vector_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'Vector', False)
    # Calling Vector(args, kwargs) (line 255)
    Vector_call_result_1034 = invoke(stypy.reporting.localization.Localization(__file__, 255, 21), Vector_1029, *[float_1030, float_1031, float_1032], **kwargs_1033)
    
    float_1035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 46), 'float')
    # Processing the call keyword arguments (line 255)
    kwargs_1036 = {}
    # Getting the type of 'Sphere' (line 255)
    Sphere_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 14), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 255)
    Sphere_call_result_1037 = invoke(stypy.reporting.localization.Localization(__file__, 255, 14), Sphere_1028, *[Vector_call_result_1034, float_1035], **kwargs_1036)
    
    # Assigning a type to the variable 'sphere1' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'sphere1', Sphere_call_result_1037)
    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 256):
    
    # Call to Sphere(...): (line 256)
    # Processing the call arguments (line 256)
    
    # Call to Vector(...): (line 256)
    # Processing the call arguments (line 256)
    float_1040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 28), 'float')
    float_1041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 34), 'float')
    float_1042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 39), 'float')
    # Processing the call keyword arguments (line 256)
    kwargs_1043 = {}
    # Getting the type of 'Vector' (line 256)
    Vector_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 21), 'Vector', False)
    # Calling Vector(args, kwargs) (line 256)
    Vector_call_result_1044 = invoke(stypy.reporting.localization.Localization(__file__, 256, 21), Vector_1039, *[float_1040, float_1041, float_1042], **kwargs_1043)
    
    float_1045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 46), 'float')
    # Processing the call keyword arguments (line 256)
    kwargs_1046 = {}
    # Getting the type of 'Sphere' (line 256)
    Sphere_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 256)
    Sphere_call_result_1047 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), Sphere_1038, *[Vector_call_result_1044, float_1045], **kwargs_1046)
    
    # Assigning a type to the variable 'sphere2' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'sphere2', Sphere_call_result_1047)
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to Sphere(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Call to Vector(...): (line 257)
    # Processing the call arguments (line 257)
    float_1050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'float')
    float_1051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 33), 'float')
    float_1052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 38), 'float')
    # Processing the call keyword arguments (line 257)
    kwargs_1053 = {}
    # Getting the type of 'Vector' (line 257)
    Vector_1049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 21), 'Vector', False)
    # Calling Vector(args, kwargs) (line 257)
    Vector_call_result_1054 = invoke(stypy.reporting.localization.Localization(__file__, 257, 21), Vector_1049, *[float_1050, float_1051, float_1052], **kwargs_1053)
    
    float_1055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 45), 'float')
    # Processing the call keyword arguments (line 257)
    kwargs_1056 = {}
    # Getting the type of 'Sphere' (line 257)
    Sphere_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 14), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 257)
    Sphere_call_result_1057 = invoke(stypy.reporting.localization.Localization(__file__, 257, 14), Sphere_1048, *[Vector_call_result_1054, float_1055], **kwargs_1056)
    
    # Assigning a type to the variable 'sphere3' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'sphere3', Sphere_call_result_1057)
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to Plane(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Call to Vector(...): (line 258)
    # Processing the call arguments (line 258)
    float_1060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 25), 'float')
    float_1061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'float')
    float_1062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 36), 'float')
    # Processing the call keyword arguments (line 258)
    kwargs_1063 = {}
    # Getting the type of 'Vector' (line 258)
    Vector_1059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'Vector', False)
    # Calling Vector(args, kwargs) (line 258)
    Vector_call_result_1064 = invoke(stypy.reporting.localization.Localization(__file__, 258, 18), Vector_1059, *[float_1060, float_1061, float_1062], **kwargs_1063)
    
    
    # Call to Vector(...): (line 258)
    # Processing the call arguments (line 258)
    float_1066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 49), 'float')
    float_1067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 54), 'float')
    float_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 59), 'float')
    # Processing the call keyword arguments (line 258)
    kwargs_1069 = {}
    # Getting the type of 'Vector' (line 258)
    Vector_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 42), 'Vector', False)
    # Calling Vector(args, kwargs) (line 258)
    Vector_call_result_1070 = invoke(stypy.reporting.localization.Localization(__file__, 258, 42), Vector_1065, *[float_1066, float_1067, float_1068], **kwargs_1069)
    
    # Processing the call keyword arguments (line 258)
    kwargs_1071 = {}
    # Getting the type of 'Plane' (line 258)
    Plane_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'Plane', False)
    # Calling Plane(args, kwargs) (line 258)
    Plane_call_result_1072 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), Plane_1058, *[Vector_call_result_1064, Vector_call_result_1070], **kwargs_1071)
    
    # Assigning a type to the variable 'plane' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'plane', Plane_call_result_1072)
    
    # ################# End of 'init_scene(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'init_scene' in the type store
    # Getting the type of 'stypy_return_type' (line 253)
    stypy_return_type_1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1073)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'init_scene'
    return stypy_return_type_1073

# Assigning a type to the variable 'init_scene' (line 253)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'init_scene', init_scene)

@norecursion
def save_ppm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'save_ppm'
    module_type_store = module_type_store.open_function_context('save_ppm', 261, 0, False)
    
    # Passed parameters checking function
    save_ppm.stypy_localization = localization
    save_ppm.stypy_type_of_self = None
    save_ppm.stypy_type_store = module_type_store
    save_ppm.stypy_function_name = 'save_ppm'
    save_ppm.stypy_param_names_list = ['img', 'w', 'h', 'fname']
    save_ppm.stypy_varargs_param_name = None
    save_ppm.stypy_kwargs_param_name = None
    save_ppm.stypy_call_defaults = defaults
    save_ppm.stypy_call_varargs = varargs
    save_ppm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'save_ppm', ['img', 'w', 'h', 'fname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'save_ppm', localization, ['img', 'w', 'h', 'fname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'save_ppm(...)' code ##################

    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to open(...): (line 262)
    # Processing the call arguments (line 262)
    
    # Call to Relative(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'fname' (line 262)
    fname_1076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'fname', False)
    # Processing the call keyword arguments (line 262)
    kwargs_1077 = {}
    # Getting the type of 'Relative' (line 262)
    Relative_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'Relative', False)
    # Calling Relative(args, kwargs) (line 262)
    Relative_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 262, 16), Relative_1075, *[fname_1076], **kwargs_1077)
    
    str_1079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'str', 'wb')
    # Processing the call keyword arguments (line 262)
    kwargs_1080 = {}
    # Getting the type of 'open' (line 262)
    open_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'open', False)
    # Calling open(args, kwargs) (line 262)
    open_call_result_1081 = invoke(stypy.reporting.localization.Localization(__file__, 262, 11), open_1074, *[Relative_call_result_1078, str_1079], **kwargs_1080)
    
    # Assigning a type to the variable 'fout' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'fout', open_call_result_1081)
    
    # Call to tofile(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'fout' (line 266)
    fout_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'fout', False)
    # Processing the call keyword arguments (line 266)
    kwargs_1089 = {}
    
    # Call to array(...): (line 266)
    # Processing the call arguments (line 266)
    str_1083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 10), 'str', 'B')
    # Getting the type of 'img' (line 266)
    img_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'img', False)
    # Processing the call keyword arguments (line 266)
    kwargs_1085 = {}
    # Getting the type of 'array' (line 266)
    array_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'array', False)
    # Calling array(args, kwargs) (line 266)
    array_call_result_1086 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), array_1082, *[str_1083, img_1084], **kwargs_1085)
    
    # Obtaining the member 'tofile' of a type (line 266)
    tofile_1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 4), array_call_result_1086, 'tofile')
    # Calling tofile(args, kwargs) (line 266)
    tofile_call_result_1090 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), tofile_1087, *[fout_1088], **kwargs_1089)
    
    
    # Call to close(...): (line 267)
    # Processing the call keyword arguments (line 267)
    kwargs_1093 = {}
    # Getting the type of 'fout' (line 267)
    fout_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'fout', False)
    # Obtaining the member 'close' of a type (line 267)
    close_1092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 4), fout_1091, 'close')
    # Calling close(args, kwargs) (line 267)
    close_call_result_1094 = invoke(stypy.reporting.localization.Localization(__file__, 267, 4), close_1092, *[], **kwargs_1093)
    
    
    # ################# End of 'save_ppm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save_ppm' in the type store
    # Getting the type of 'stypy_return_type' (line 261)
    stypy_return_type_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1095)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save_ppm'
    return stypy_return_type_1095

# Assigning a type to the variable 'save_ppm' (line 261)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'save_ppm', save_ppm)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 270, 0, False)
    
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

    
    # Call to init_scene(...): (line 271)
    # Processing the call keyword arguments (line 271)
    kwargs_1097 = {}
    # Getting the type of 'init_scene' (line 271)
    init_scene_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'init_scene', False)
    # Calling init_scene(args, kwargs) (line 271)
    init_scene_call_result_1098 = invoke(stypy.reporting.localization.Localization(__file__, 271, 4), init_scene_1096, *[], **kwargs_1097)
    
    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to render(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'WIDTH' (line 272)
    WIDTH_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'WIDTH', False)
    # Getting the type of 'HEIGHT' (line 272)
    HEIGHT_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 24), 'HEIGHT', False)
    # Getting the type of 'NSUBSAMPLES' (line 272)
    NSUBSAMPLES_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'NSUBSAMPLES', False)
    # Processing the call keyword arguments (line 272)
    kwargs_1103 = {}
    # Getting the type of 'render' (line 272)
    render_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 10), 'render', False)
    # Calling render(args, kwargs) (line 272)
    render_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 272, 10), render_1099, *[WIDTH_1100, HEIGHT_1101, NSUBSAMPLES_1102], **kwargs_1103)
    
    # Assigning a type to the variable 'img' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'img', render_call_result_1104)
    
    # Call to save_ppm(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'img' (line 273)
    img_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 13), 'img', False)
    # Getting the type of 'WIDTH' (line 273)
    WIDTH_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'WIDTH', False)
    # Getting the type of 'HEIGHT' (line 273)
    HEIGHT_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'HEIGHT', False)
    str_1109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 33), 'str', 'ao_py.ppm')
    # Processing the call keyword arguments (line 273)
    kwargs_1110 = {}
    # Getting the type of 'save_ppm' (line 273)
    save_ppm_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'save_ppm', False)
    # Calling save_ppm(args, kwargs) (line 273)
    save_ppm_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 273, 4), save_ppm_1105, *[img_1106, WIDTH_1107, HEIGHT_1108, str_1109], **kwargs_1110)
    
    # Getting the type of 'True' (line 274)
    True_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type', True_1112)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 270)
    stypy_return_type_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1113)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1113

# Assigning a type to the variable 'run' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'run', run)

# Call to run(...): (line 277)
# Processing the call keyword arguments (line 277)
kwargs_1115 = {}
# Getting the type of 'run' (line 277)
run_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 0), 'run', False)
# Calling run(args, kwargs) (line 277)
run_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 277, 0), run_1114, *[], **kwargs_1115)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

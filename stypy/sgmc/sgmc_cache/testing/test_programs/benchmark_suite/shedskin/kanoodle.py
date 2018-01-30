
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This will find solutions to the kanoodle puzzle where the "L" piece,
3: which looks like a "*" sign is placed with the center at (Lcol, Lrow)
4: 
5: implements the Dancing Links algorithm, following Knuth's paper
6: 
7: copyright David Austin, license GPL2
8: 
9: '''
10: 
11: import sys
12: 
13: updates = 0
14: udates = [0] * 324
15: nodes = 0
16: 
17: filename = None
18: 
19: 
20: def setfilename(s):
21:     global filename
22:     filename = s
23: 
24: 
25: class Column:
26:     def __init__(self, name=None):
27:         if name is None:
28:             self.up = None
29:             self.down = None
30:             self.right = None
31:             self.left = None
32:             self.column = None
33:         else:
34:             self.size = 0
35:             self.name = name
36:             self.down = self
37:             self.up = self
38:             self.extra = None
39: 
40:     def cover(self):
41:         global updates
42:         updates += 1
43:         udates[level] += 1
44: 
45:         self.right.left = self.left
46:         self.left.right = self.right
47:         i = self.down
48:         while i != self:
49:             j = i.right
50:             while j != i:
51:                 j.down.up = j.up
52:                 j.up.down = j.down
53:                 j.column.size -= 1
54:                 j = j.right
55:             i = i.down
56: 
57:     def uncover(self):
58:         i = self.up
59:         while i != self:
60:             j = i.left
61:             while j != i:
62:                 j.column.size += 1
63:                 j.down.up = j
64:                 j.up.down = j
65:                 j = j.left
66:             i = i.up
67:         self.right.left = self
68:         self.left.right = self
69: 
70:     def __str__(self):
71:         return self.name
72: 
73: 
74: def search(k):
75:     global o, solutions, level, nodes
76:     level = k
77:     if root.right == root:
78:         printsolution(o)
79:         solutions += 1
80:         sys.exit()  # XXX shedskin
81:         return
82: 
83:     nodes += 1
84:     j = root.right
85:     s = j.size
86:     c = j
87:     j = j.right
88:     while j != root:
89:         if j.size < s:
90:             c = j
91:             s = j.size
92:         j = j.right
93: 
94:     ## Don't use S heuristic
95:     #    c = root.right
96: 
97:     c.cover()
98:     r = c.down
99:     while r != c:
100:         o.append(r)
101:         j = r.right
102:         while j != r:
103:             j.column.cover()
104:             j = j.right
105:         search(k + 1)
106:         level = k
107:         r = o.pop(-1)
108:         c = r.column
109:         j = r.left
110:         while j != r:
111:             j.column.uncover()
112:             j = j.left
113:         r = r.down
114:     c.uncover()
115: 
116:     if k == 0:
117:         count = 0
118:         j = root.right
119:         while j != root:
120:             count += 1
121:             j = j.right
122:         # print 'nodes =', nodes
123:         # print 'solutions =', solutions
124: 
125: 
126: def printsolution(o):
127:     # print '### solution!'
128:     for row in o:
129:         r = row
130:         s = r.column.name
131:         r = r.right
132:         while r != row:
133:             s += ' ' + r.column.name
134:             r = r.right
135:         # print s
136: 
137: 
138: def printmatrix(root):
139:     c = root.right
140:     while c != root:
141:         r = c.down
142:         while r != c:
143:             printrow(r)
144:             r = r.down
145:         c = c.right
146: 
147: 
148: def printrow(r):
149:     s = r.column.name
150:     next = r.right
151:     while next != r:
152:         s += ' ' + next.column.name
153:         next = next.right
154:     # print s
155: 
156: 
157: def setroot(r):
158:     global root
159:     root = r
160: 
161: 
162: solutions = 0
163: o = []
164: 
165: # def setprintsolution(f):
166: #    global printsolution
167: #    printsolution = f
168: 
169: Lcol = 5
170: Lrow = 2
171: 
172: 
173: ## some basic matrix operations
174: 
175: def matrixmultiply(m, n):
176:     r = [[0, 0], [0, 0]]
177:     for i in range(2):
178:         for j in range(2):
179:             sum = 0
180:             for k in range(2):
181:                 sum += m[i][k] * n[k][j]
182:             r[i][j] = sum
183:     return r
184: 
185: 
186: def matrixact(m, v):
187:     u = [0, 0]
188:     for i in range(2):
189:         sum = 0
190:         for j in range(2):
191:             sum += m[i][j] * v[j]
192:         u[i] = sum
193:     return u
194: 
195: 
196: ## linear isometries to apply to kanoodle pieces
197: identity = [[1, 0], [0, 1]]
198: r90 = [[0, -1], [1, 0]]
199: r180 = [[-1, 0], [0, -1]]
200: r270 = [[0, 1], [-1, 0]]
201: r1 = [[1, 0], [0, -1]]
202: r2 = matrixmultiply(r1, r90)
203: r3 = matrixmultiply(r1, r180)
204: r4 = matrixmultiply(r1, r270)
205: 
206: ## sets of isometries
207: 
208: symmetries = [identity, r90, r180, r270, r1, r2, r3, r4]
209: rotations = [identity, r90, r180, r270]
210: 
211: 
212: ## classes for each of the pieces
213: 
214: class Omino:
215:     def getorientations(self):
216:         orientations = []
217:         for symmetry in self.cosets:
218:             orientation = []
219:             for cell in self.cells:
220:                 orientation.append(matrixact(symmetry, cell))
221:             orientations.append(orientation)
222:         self.orientations = orientations
223: 
224:     def move(self, v):
225:         newcells = []
226:         for cell in self.cells:
227:             newcells.append([cell[0] + v[0], cell[1] + v[1]])
228:         self.cells = newcells
229: 
230:     def translate(self, v):
231:         r = []
232:         for orientation in self.orientations:
233:             s = []
234:             for cell in orientation:
235:                 s.append([cell[0] + v[0], cell[1] + v[1]])
236:             r.append(s)
237:         return r
238: 
239: 
240: class A(Omino):
241:     def __init__(self):
242:         self.name = 'A'
243:         self.cells = [[0, 0], [1, 0], [1, 1], [1, 2]]
244:         self.cosets = symmetries
245:         self.getorientations()
246: 
247: 
248: class B(Omino):
249:     def __init__(self):
250:         self.name = 'B'
251:         self.cells = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
252:         self.cosets = symmetries
253:         self.getorientations()
254: 
255: 
256: class C(Omino):
257:     def __init__(self):
258:         self.name = 'C'
259:         self.cells = [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3]]
260:         self.cosets = symmetries
261:         self.getorientations()
262: 
263: 
264: class D(Omino):
265:     def __init__(self):
266:         self.name = 'D'
267:         self.cells = [[0, -1], [-1, 0], [0, 0], [0, 1], [0, 2]]
268:         self.cosets = symmetries
269:         self.getorientations()
270: 
271: 
272: class E(Omino):
273:     def __init__(self):
274:         self.name = 'E'
275:         self.cells = [[0, 0], [0, 1], [1, 1], [1, 2], [1, 3]]
276:         self.cosets = symmetries
277:         self.getorientations()
278: 
279: 
280: class F(Omino):
281:     def __init__(self):
282:         self.name = 'F'
283:         self.cells = [[0, 0], [1, 0], [0, 1]]
284:         self.cosets = rotations
285:         self.getorientations()
286: 
287: 
288: class G(Omino):
289:     def __init__(self):
290:         self.name = 'G'
291:         self.cells = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]
292:         self.cosets = rotations
293:         self.getorientations()
294: 
295: 
296: class H(Omino):
297:     def __init__(self):
298:         self.name = 'H'
299:         self.cells = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]
300:         self.cosets = rotations
301:         self.getorientations()
302: 
303: 
304: class I(Omino):
305:     def __init__(self):
306:         self.name = 'I'
307:         self.cells = [[0, 1], [0, 0], [1, 0], [2, 0], [2, 1]]
308:         self.cosets = rotations
309:         self.getorientations()
310: 
311: 
312: class J(Omino):
313:     def __init__(self):
314:         self.name = 'J'
315:         self.cells = [[0, 0], [0, 1], [0, 2], [0, 3]]
316:         self.cosets = [identity, r90]
317:         self.getorientations()
318: 
319: 
320: class K(Omino):
321:     def __init__(self):
322:         self.name = 'K'
323:         self.cells = [[0, 0], [1, 0], [1, 1], [0, 1]]
324:         self.cosets = [identity]
325:         self.getorientations()
326: 
327: 
328: class L(Omino):
329:     def __init__(self):
330:         self.name = 'L'
331:         self.cells = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
332:         self.cosets = [identity]
333:         self.getorientations()
334: 
335: 
336: def set5x11():
337:     global c1, ominos, rows, columns
338:     c1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
339:     ominos = [A(), B(), C(), D(), E(), F(), G(), H(), I(), J(), K(), L()]
340:     rows = 5
341:     columns = 11
342: 
343: 
344: ## set up the 5x11 board
345: set5x11()
346: 
347: ## start building the matrix for the exact cover problem
348: 
349: root = Column('root')
350: root.left = root
351: root.right = root
352: 
353: last = root
354: 
355: ## build the columns
356: 
357: pcolumns = {}
358: for col2 in c1:
359:     c = Column(col2)
360:     last.right = c
361:     c.left = last
362:     c.right = root
363:     root.left = c
364:     last = c
365:     pcolumns[col2] = c
366: 
367: last = root
368: for row in range(rows):
369:     for col in range(columns):
370:         c = Column('[' + str(col) + ',' + str(row) + '] ')
371:         c.extra = [col, row]
372: 
373:         last.right.left = c
374:         c.right = last.right
375:         last.right = c
376:         c.left = last
377:         last = c
378: 
379: 
380: ## check to see if a pieces fits on the board
381: 
382: def validatecell(c):
383:     if c[0] < 0 or c[0] > columns: return False
384:     if c[1] < 0 or c[1] > rows: return False
385:     return True
386: 
387: 
388: def validate(orientation):
389:     for cell in orientation:
390:         if validatecell(cell) == False: return False
391:     return True
392: 
393: 
394: ## construct the rows of the matrix
395: 
396: rownums = 0
397: for tile in ominos:
398:     for col in range(columns):
399:         if tile.name == 'L' and col != Lcol: continue
400:         for row in range(rows):
401:             if tile.name == 'L' and row != Lrow: continue
402:             orientations = tile.translate([col, row])
403:             for orientation in orientations:
404:                 if validate(orientation) == False: continue
405:                 rownums += 1
406:                 element = Column()
407:                 element.right = element
408:                 element.left = element
409: 
410:                 column = pcolumns[tile.name]
411:                 element.column = column
412:                 element.up = column.up
413:                 element.down = column
414:                 column.up.down = element
415:                 column.up = element
416:                 column.size += 1
417:                 rowelement = element
418: 
419:                 column = root.right
420:                 while column.extra != None:
421:                     entry = column.extra
422:                     for cell in orientation:
423:                         if entry[0] == cell[0] and entry[1] == cell[1]:
424:                             element = Column()
425:                             rowelement.right.left = element
426:                             element.right = rowelement.right
427:                             rowelement.right = element
428:                             element.left = rowelement
429: 
430:                             element.column = column
431:                             element.up = column.up
432:                             element.down = column
433:                             column.up.down = element
434:                             column.up = element
435:                             rowelement = element
436:                             column.size += 1
437:                     column = column.right
438: 
439: 
440: ## apply the Dancing Links algorithm to the matrix
441: 
442: def run():
443:     try:
444:         setroot(root)
445:         # print 'begin search'
446:         search(0)
447:         # print 'finished search'
448:     except SystemExit:
449:         pass
450:     return True
451: 
452: 
453: run()
454: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\nThis will find solutions to the kanoodle puzzle where the "L" piece,\nwhich looks like a "*" sign is placed with the center at (Lcol, Lrow)\n\nimplements the Dancing Links algorithm, following Knuth\'s paper\n\ncopyright David Austin, license GPL2\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import sys' statement (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)


# Assigning a Num to a Name (line 13):
int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'int')
# Assigning a type to the variable 'updates' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'updates', int_2)

# Assigning a BinOp to a Name (line 14):

# Obtaining an instance of the builtin type 'list' (line 14)
list_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 9), list_3, int_4)

int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
# Applying the binary operator '*' (line 14)
result_mul_6 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '*', list_3, int_5)

# Assigning a type to the variable 'udates' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'udates', result_mul_6)

# Assigning a Num to a Name (line 15):
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'int')
# Assigning a type to the variable 'nodes' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'nodes', int_7)

# Assigning a Name to a Name (line 17):
# Getting the type of 'None' (line 17)
None_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'None')
# Assigning a type to the variable 'filename' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'filename', None_8)

@norecursion
def setfilename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setfilename'
    module_type_store = module_type_store.open_function_context('setfilename', 20, 0, False)
    
    # Passed parameters checking function
    setfilename.stypy_localization = localization
    setfilename.stypy_type_of_self = None
    setfilename.stypy_type_store = module_type_store
    setfilename.stypy_function_name = 'setfilename'
    setfilename.stypy_param_names_list = ['s']
    setfilename.stypy_varargs_param_name = None
    setfilename.stypy_kwargs_param_name = None
    setfilename.stypy_call_defaults = defaults
    setfilename.stypy_call_varargs = varargs
    setfilename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setfilename', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setfilename', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setfilename(...)' code ##################

    # Marking variables as global (line 21)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 21, 4), 'filename')
    
    # Assigning a Name to a Name (line 22):
    # Getting the type of 's' (line 22)
    s_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 's')
    # Assigning a type to the variable 'filename' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'filename', s_9)
    
    # ################# End of 'setfilename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setfilename' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setfilename'
    return stypy_return_type_10

# Assigning a type to the variable 'setfilename' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'setfilename', setfilename)
# Declaration of the 'Column' class

class Column:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 26)
        None_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'None')
        defaults = [None_11]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Column.__init__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 27)
        # Getting the type of 'name' (line 27)
        name_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'name')
        # Getting the type of 'None' (line 27)
        None_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'None')
        
        (may_be_14, more_types_in_union_15) = may_be_none(name_12, None_13)

        if may_be_14:

            if more_types_in_union_15:
                # Runtime conditional SSA (line 27)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 28):
            # Getting the type of 'None' (line 28)
            None_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'None')
            # Getting the type of 'self' (line 28)
            self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'self')
            # Setting the type of the member 'up' of a type (line 28)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), self_17, 'up', None_16)
            
            # Assigning a Name to a Attribute (line 29):
            # Getting the type of 'None' (line 29)
            None_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'None')
            # Getting the type of 'self' (line 29)
            self_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'self')
            # Setting the type of the member 'down' of a type (line 29)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), self_19, 'down', None_18)
            
            # Assigning a Name to a Attribute (line 30):
            # Getting the type of 'None' (line 30)
            None_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'None')
            # Getting the type of 'self' (line 30)
            self_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'self')
            # Setting the type of the member 'right' of a type (line 30)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), self_21, 'right', None_20)
            
            # Assigning a Name to a Attribute (line 31):
            # Getting the type of 'None' (line 31)
            None_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'None')
            # Getting the type of 'self' (line 31)
            self_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'self')
            # Setting the type of the member 'left' of a type (line 31)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), self_23, 'left', None_22)
            
            # Assigning a Name to a Attribute (line 32):
            # Getting the type of 'None' (line 32)
            None_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'None')
            # Getting the type of 'self' (line 32)
            self_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self')
            # Setting the type of the member 'column' of a type (line 32)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_25, 'column', None_24)

            if more_types_in_union_15:
                # Runtime conditional SSA for else branch (line 27)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_14) or more_types_in_union_15):
            
            # Assigning a Num to a Attribute (line 34):
            int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'int')
            # Getting the type of 'self' (line 34)
            self_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self')
            # Setting the type of the member 'size' of a type (line 34)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_27, 'size', int_26)
            
            # Assigning a Name to a Attribute (line 35):
            # Getting the type of 'name' (line 35)
            name_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'name')
            # Getting the type of 'self' (line 35)
            self_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self')
            # Setting the type of the member 'name' of a type (line 35)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_29, 'name', name_28)
            
            # Assigning a Name to a Attribute (line 36):
            # Getting the type of 'self' (line 36)
            self_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'self')
            # Getting the type of 'self' (line 36)
            self_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
            # Setting the type of the member 'down' of a type (line 36)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_31, 'down', self_30)
            
            # Assigning a Name to a Attribute (line 37):
            # Getting the type of 'self' (line 37)
            self_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'self')
            # Getting the type of 'self' (line 37)
            self_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'self')
            # Setting the type of the member 'up' of a type (line 37)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), self_33, 'up', self_32)
            
            # Assigning a Name to a Attribute (line 38):
            # Getting the type of 'None' (line 38)
            None_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'None')
            # Getting the type of 'self' (line 38)
            self_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'self')
            # Setting the type of the member 'extra' of a type (line 38)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), self_35, 'extra', None_34)

            if (may_be_14 and more_types_in_union_15):
                # SSA join for if statement (line 27)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def cover(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cover'
        module_type_store = module_type_store.open_function_context('cover', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Column.cover.__dict__.__setitem__('stypy_localization', localization)
        Column.cover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Column.cover.__dict__.__setitem__('stypy_type_store', module_type_store)
        Column.cover.__dict__.__setitem__('stypy_function_name', 'Column.cover')
        Column.cover.__dict__.__setitem__('stypy_param_names_list', [])
        Column.cover.__dict__.__setitem__('stypy_varargs_param_name', None)
        Column.cover.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Column.cover.__dict__.__setitem__('stypy_call_defaults', defaults)
        Column.cover.__dict__.__setitem__('stypy_call_varargs', varargs)
        Column.cover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Column.cover.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Column.cover', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cover', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cover(...)' code ##################

        # Marking variables as global (line 41)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 41, 8), 'updates')
        
        # Getting the type of 'updates' (line 42)
        updates_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'updates')
        int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'int')
        # Applying the binary operator '+=' (line 42)
        result_iadd_38 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 8), '+=', updates_36, int_37)
        # Assigning a type to the variable 'updates' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'updates', result_iadd_38)
        
        
        # Getting the type of 'udates' (line 43)
        udates_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'udates')
        
        # Obtaining the type of the subscript
        # Getting the type of 'level' (line 43)
        level_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'level')
        # Getting the type of 'udates' (line 43)
        udates_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'udates')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), udates_41, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), getitem___42, level_40)
        
        int_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'int')
        # Applying the binary operator '+=' (line 43)
        result_iadd_45 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 8), '+=', subscript_call_result_43, int_44)
        # Getting the type of 'udates' (line 43)
        udates_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'udates')
        # Getting the type of 'level' (line 43)
        level_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'level')
        # Storing an element on a container (line 43)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), udates_46, (level_47, result_iadd_45))
        
        
        # Assigning a Attribute to a Attribute (line 45):
        # Getting the type of 'self' (line 45)
        self_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'self')
        # Obtaining the member 'left' of a type (line 45)
        left_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 26), self_48, 'left')
        # Getting the type of 'self' (line 45)
        self_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Obtaining the member 'right' of a type (line 45)
        right_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_50, 'right')
        # Setting the type of the member 'left' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), right_51, 'left', left_49)
        
        # Assigning a Attribute to a Attribute (line 46):
        # Getting the type of 'self' (line 46)
        self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'self')
        # Obtaining the member 'right' of a type (line 46)
        right_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 26), self_52, 'right')
        # Getting the type of 'self' (line 46)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Obtaining the member 'left' of a type (line 46)
        left_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_54, 'left')
        # Setting the type of the member 'right' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), left_55, 'right', right_53)
        
        # Assigning a Attribute to a Name (line 47):
        # Getting the type of 'self' (line 47)
        self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self')
        # Obtaining the member 'down' of a type (line 47)
        down_57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_56, 'down')
        # Assigning a type to the variable 'i' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'i', down_57)
        
        
        # Getting the type of 'i' (line 48)
        i_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'i')
        # Getting the type of 'self' (line 48)
        self_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'self')
        # Applying the binary operator '!=' (line 48)
        result_ne_60 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 14), '!=', i_58, self_59)
        
        # Testing if the while is going to be iterated (line 48)
        # Testing the type of an if condition (line 48)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_ne_60)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 8), result_ne_60):
            # SSA begins for while statement (line 48)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Attribute to a Name (line 49):
            # Getting the type of 'i' (line 49)
            i_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'i')
            # Obtaining the member 'right' of a type (line 49)
            right_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), i_61, 'right')
            # Assigning a type to the variable 'j' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'j', right_62)
            
            
            # Getting the type of 'j' (line 50)
            j_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'j')
            # Getting the type of 'i' (line 50)
            i_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 'i')
            # Applying the binary operator '!=' (line 50)
            result_ne_65 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 18), '!=', j_63, i_64)
            
            # Testing if the while is going to be iterated (line 50)
            # Testing the type of an if condition (line 50)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), result_ne_65)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 12), result_ne_65):
                # SSA begins for while statement (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a Attribute to a Attribute (line 51):
                # Getting the type of 'j' (line 51)
                j_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'j')
                # Obtaining the member 'up' of a type (line 51)
                up_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), j_66, 'up')
                # Getting the type of 'j' (line 51)
                j_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'j')
                # Obtaining the member 'down' of a type (line 51)
                down_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), j_68, 'down')
                # Setting the type of the member 'up' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), down_69, 'up', up_67)
                
                # Assigning a Attribute to a Attribute (line 52):
                # Getting the type of 'j' (line 52)
                j_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'j')
                # Obtaining the member 'down' of a type (line 52)
                down_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), j_70, 'down')
                # Getting the type of 'j' (line 52)
                j_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'j')
                # Obtaining the member 'up' of a type (line 52)
                up_73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), j_72, 'up')
                # Setting the type of the member 'down' of a type (line 52)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), up_73, 'down', down_71)
                
                # Getting the type of 'j' (line 53)
                j_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'j')
                # Obtaining the member 'column' of a type (line 53)
                column_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), j_74, 'column')
                # Obtaining the member 'size' of a type (line 53)
                size_76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), column_75, 'size')
                int_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 33), 'int')
                # Applying the binary operator '-=' (line 53)
                result_isub_78 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 16), '-=', size_76, int_77)
                # Getting the type of 'j' (line 53)
                j_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'j')
                # Obtaining the member 'column' of a type (line 53)
                column_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), j_79, 'column')
                # Setting the type of the member 'size' of a type (line 53)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), column_80, 'size', result_isub_78)
                
                
                # Assigning a Attribute to a Name (line 54):
                # Getting the type of 'j' (line 54)
                j_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'j')
                # Obtaining the member 'right' of a type (line 54)
                right_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 20), j_81, 'right')
                # Assigning a type to the variable 'j' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'j', right_82)
                # SSA join for while statement (line 50)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Attribute to a Name (line 55):
            # Getting the type of 'i' (line 55)
            i_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'i')
            # Obtaining the member 'down' of a type (line 55)
            down_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), i_83, 'down')
            # Assigning a type to the variable 'i' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'i', down_84)
            # SSA join for while statement (line 48)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'cover(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cover' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_85)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cover'
        return stypy_return_type_85


    @norecursion
    def uncover(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'uncover'
        module_type_store = module_type_store.open_function_context('uncover', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Column.uncover.__dict__.__setitem__('stypy_localization', localization)
        Column.uncover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Column.uncover.__dict__.__setitem__('stypy_type_store', module_type_store)
        Column.uncover.__dict__.__setitem__('stypy_function_name', 'Column.uncover')
        Column.uncover.__dict__.__setitem__('stypy_param_names_list', [])
        Column.uncover.__dict__.__setitem__('stypy_varargs_param_name', None)
        Column.uncover.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Column.uncover.__dict__.__setitem__('stypy_call_defaults', defaults)
        Column.uncover.__dict__.__setitem__('stypy_call_varargs', varargs)
        Column.uncover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Column.uncover.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Column.uncover', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'uncover', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'uncover(...)' code ##################

        
        # Assigning a Attribute to a Name (line 58):
        # Getting the type of 'self' (line 58)
        self_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'self')
        # Obtaining the member 'up' of a type (line 58)
        up_87 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), self_86, 'up')
        # Assigning a type to the variable 'i' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'i', up_87)
        
        
        # Getting the type of 'i' (line 59)
        i_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'i')
        # Getting the type of 'self' (line 59)
        self_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'self')
        # Applying the binary operator '!=' (line 59)
        result_ne_90 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 14), '!=', i_88, self_89)
        
        # Testing if the while is going to be iterated (line 59)
        # Testing the type of an if condition (line 59)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_ne_90)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 59, 8), result_ne_90):
            # SSA begins for while statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Attribute to a Name (line 60):
            # Getting the type of 'i' (line 60)
            i_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'i')
            # Obtaining the member 'left' of a type (line 60)
            left_92 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), i_91, 'left')
            # Assigning a type to the variable 'j' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'j', left_92)
            
            
            # Getting the type of 'j' (line 61)
            j_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'j')
            # Getting the type of 'i' (line 61)
            i_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'i')
            # Applying the binary operator '!=' (line 61)
            result_ne_95 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 18), '!=', j_93, i_94)
            
            # Testing if the while is going to be iterated (line 61)
            # Testing the type of an if condition (line 61)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 12), result_ne_95)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 12), result_ne_95):
                # SSA begins for while statement (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Getting the type of 'j' (line 62)
                j_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'j')
                # Obtaining the member 'column' of a type (line 62)
                column_97 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), j_96, 'column')
                # Obtaining the member 'size' of a type (line 62)
                size_98 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), column_97, 'size')
                int_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 33), 'int')
                # Applying the binary operator '+=' (line 62)
                result_iadd_100 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 16), '+=', size_98, int_99)
                # Getting the type of 'j' (line 62)
                j_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'j')
                # Obtaining the member 'column' of a type (line 62)
                column_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), j_101, 'column')
                # Setting the type of the member 'size' of a type (line 62)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), column_102, 'size', result_iadd_100)
                
                
                # Assigning a Name to a Attribute (line 63):
                # Getting the type of 'j' (line 63)
                j_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'j')
                # Getting the type of 'j' (line 63)
                j_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'j')
                # Obtaining the member 'down' of a type (line 63)
                down_105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), j_104, 'down')
                # Setting the type of the member 'up' of a type (line 63)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), down_105, 'up', j_103)
                
                # Assigning a Name to a Attribute (line 64):
                # Getting the type of 'j' (line 64)
                j_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'j')
                # Getting the type of 'j' (line 64)
                j_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'j')
                # Obtaining the member 'up' of a type (line 64)
                up_108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), j_107, 'up')
                # Setting the type of the member 'down' of a type (line 64)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), up_108, 'down', j_106)
                
                # Assigning a Attribute to a Name (line 65):
                # Getting the type of 'j' (line 65)
                j_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'j')
                # Obtaining the member 'left' of a type (line 65)
                left_110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), j_109, 'left')
                # Assigning a type to the variable 'j' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'j', left_110)
                # SSA join for while statement (line 61)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Attribute to a Name (line 66):
            # Getting the type of 'i' (line 66)
            i_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'i')
            # Obtaining the member 'up' of a type (line 66)
            up_112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), i_111, 'up')
            # Assigning a type to the variable 'i' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'i', up_112)
            # SSA join for while statement (line 59)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 67):
        # Getting the type of 'self' (line 67)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'self')
        # Getting the type of 'self' (line 67)
        self_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Obtaining the member 'right' of a type (line 67)
        right_115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_114, 'right')
        # Setting the type of the member 'left' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), right_115, 'left', self_113)
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'self' (line 68)
        self_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'self')
        # Getting the type of 'self' (line 68)
        self_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Obtaining the member 'left' of a type (line 68)
        left_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_117, 'left')
        # Setting the type of the member 'right' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), left_118, 'right', self_116)
        
        # ################# End of 'uncover(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'uncover' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'uncover'
        return stypy_return_type_119


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Column.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Column.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Column.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Column.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Column.stypy__str__')
        Column.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Column.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Column.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Column.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Column.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Column.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Column.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Column.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 71)
        self_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'self')
        # Obtaining the member 'name' of a type (line 71)
        name_121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 15), self_120, 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'stypy_return_type', name_121)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_122


# Assigning a type to the variable 'Column' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'Column', Column)

@norecursion
def search(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'search'
    module_type_store = module_type_store.open_function_context('search', 74, 0, False)
    
    # Passed parameters checking function
    search.stypy_localization = localization
    search.stypy_type_of_self = None
    search.stypy_type_store = module_type_store
    search.stypy_function_name = 'search'
    search.stypy_param_names_list = ['k']
    search.stypy_varargs_param_name = None
    search.stypy_kwargs_param_name = None
    search.stypy_call_defaults = defaults
    search.stypy_call_varargs = varargs
    search.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'search', ['k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'search', localization, ['k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'search(...)' code ##################

    # Marking variables as global (line 75)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 75, 4), 'o')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 75, 4), 'solutions')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 75, 4), 'level')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 75, 4), 'nodes')
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'k' (line 76)
    k_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'k')
    # Assigning a type to the variable 'level' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'level', k_123)
    
    # Getting the type of 'root' (line 77)
    root_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 7), 'root')
    # Obtaining the member 'right' of a type (line 77)
    right_125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 7), root_124, 'right')
    # Getting the type of 'root' (line 77)
    root_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'root')
    # Applying the binary operator '==' (line 77)
    result_eq_127 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), '==', right_125, root_126)
    
    # Testing if the type of an if condition is none (line 77)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 4), result_eq_127):
        pass
    else:
        
        # Testing the type of an if condition (line 77)
        if_condition_128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), result_eq_127)
        # Assigning a type to the variable 'if_condition_128' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_128', if_condition_128)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to printsolution(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'o' (line 78)
        o_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'o', False)
        # Processing the call keyword arguments (line 78)
        kwargs_131 = {}
        # Getting the type of 'printsolution' (line 78)
        printsolution_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'printsolution', False)
        # Calling printsolution(args, kwargs) (line 78)
        printsolution_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), printsolution_129, *[o_130], **kwargs_131)
        
        
        # Getting the type of 'solutions' (line 79)
        solutions_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'solutions')
        int_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'int')
        # Applying the binary operator '+=' (line 79)
        result_iadd_135 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 8), '+=', solutions_133, int_134)
        # Assigning a type to the variable 'solutions' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'solutions', result_iadd_135)
        
        
        # Call to exit(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_138 = {}
        # Getting the type of 'sys' (line 80)
        sys_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'sys', False)
        # Obtaining the member 'exit' of a type (line 80)
        exit_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), sys_136, 'exit')
        # Calling exit(args, kwargs) (line 80)
        exit_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), exit_137, *[], **kwargs_138)
        
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'nodes' (line 83)
    nodes_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'nodes')
    int_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'int')
    # Applying the binary operator '+=' (line 83)
    result_iadd_142 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 4), '+=', nodes_140, int_141)
    # Assigning a type to the variable 'nodes' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'nodes', result_iadd_142)
    
    
    # Assigning a Attribute to a Name (line 84):
    # Getting the type of 'root' (line 84)
    root_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'root')
    # Obtaining the member 'right' of a type (line 84)
    right_144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), root_143, 'right')
    # Assigning a type to the variable 'j' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'j', right_144)
    
    # Assigning a Attribute to a Name (line 85):
    # Getting the type of 'j' (line 85)
    j_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'j')
    # Obtaining the member 'size' of a type (line 85)
    size_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), j_145, 'size')
    # Assigning a type to the variable 's' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 's', size_146)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'j' (line 86)
    j_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'j')
    # Assigning a type to the variable 'c' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'c', j_147)
    
    # Assigning a Attribute to a Name (line 87):
    # Getting the type of 'j' (line 87)
    j_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'j')
    # Obtaining the member 'right' of a type (line 87)
    right_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), j_148, 'right')
    # Assigning a type to the variable 'j' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'j', right_149)
    
    
    # Getting the type of 'j' (line 88)
    j_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 10), 'j')
    # Getting the type of 'root' (line 88)
    root_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'root')
    # Applying the binary operator '!=' (line 88)
    result_ne_152 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 10), '!=', j_150, root_151)
    
    # Testing if the while is going to be iterated (line 88)
    # Testing the type of an if condition (line 88)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 4), result_ne_152)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 4), result_ne_152):
        # SSA begins for while statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'j' (line 89)
        j_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'j')
        # Obtaining the member 'size' of a type (line 89)
        size_154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), j_153, 'size')
        # Getting the type of 's' (line 89)
        s_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 's')
        # Applying the binary operator '<' (line 89)
        result_lt_156 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), '<', size_154, s_155)
        
        # Testing if the type of an if condition is none (line 89)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 8), result_lt_156):
            pass
        else:
            
            # Testing the type of an if condition (line 89)
            if_condition_157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), result_lt_156)
            # Assigning a type to the variable 'if_condition_157' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_157', if_condition_157)
            # SSA begins for if statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 90):
            # Getting the type of 'j' (line 90)
            j_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'j')
            # Assigning a type to the variable 'c' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'c', j_158)
            
            # Assigning a Attribute to a Name (line 91):
            # Getting the type of 'j' (line 91)
            j_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'j')
            # Obtaining the member 'size' of a type (line 91)
            size_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 16), j_159, 'size')
            # Assigning a type to the variable 's' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 's', size_160)
            # SSA join for if statement (line 89)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Attribute to a Name (line 92):
        # Getting the type of 'j' (line 92)
        j_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'j')
        # Obtaining the member 'right' of a type (line 92)
        right_162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), j_161, 'right')
        # Assigning a type to the variable 'j' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'j', right_162)
        # SSA join for while statement (line 88)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to cover(...): (line 97)
    # Processing the call keyword arguments (line 97)
    kwargs_165 = {}
    # Getting the type of 'c' (line 97)
    c_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'c', False)
    # Obtaining the member 'cover' of a type (line 97)
    cover_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), c_163, 'cover')
    # Calling cover(args, kwargs) (line 97)
    cover_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), cover_164, *[], **kwargs_165)
    
    
    # Assigning a Attribute to a Name (line 98):
    # Getting the type of 'c' (line 98)
    c_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'c')
    # Obtaining the member 'down' of a type (line 98)
    down_168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), c_167, 'down')
    # Assigning a type to the variable 'r' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'r', down_168)
    
    
    # Getting the type of 'r' (line 99)
    r_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 10), 'r')
    # Getting the type of 'c' (line 99)
    c_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'c')
    # Applying the binary operator '!=' (line 99)
    result_ne_171 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 10), '!=', r_169, c_170)
    
    # Testing if the while is going to be iterated (line 99)
    # Testing the type of an if condition (line 99)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_ne_171)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 99, 4), result_ne_171):
        # SSA begins for while statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to append(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'r' (line 100)
        r_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'r', False)
        # Processing the call keyword arguments (line 100)
        kwargs_175 = {}
        # Getting the type of 'o' (line 100)
        o_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'o', False)
        # Obtaining the member 'append' of a type (line 100)
        append_173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), o_172, 'append')
        # Calling append(args, kwargs) (line 100)
        append_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), append_173, *[r_174], **kwargs_175)
        
        
        # Assigning a Attribute to a Name (line 101):
        # Getting the type of 'r' (line 101)
        r_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'r')
        # Obtaining the member 'right' of a type (line 101)
        right_178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), r_177, 'right')
        # Assigning a type to the variable 'j' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'j', right_178)
        
        
        # Getting the type of 'j' (line 102)
        j_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'j')
        # Getting the type of 'r' (line 102)
        r_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'r')
        # Applying the binary operator '!=' (line 102)
        result_ne_181 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 14), '!=', j_179, r_180)
        
        # Testing if the while is going to be iterated (line 102)
        # Testing the type of an if condition (line 102)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_ne_181)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 102, 8), result_ne_181):
            # SSA begins for while statement (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Call to cover(...): (line 103)
            # Processing the call keyword arguments (line 103)
            kwargs_185 = {}
            # Getting the type of 'j' (line 103)
            j_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'j', False)
            # Obtaining the member 'column' of a type (line 103)
            column_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), j_182, 'column')
            # Obtaining the member 'cover' of a type (line 103)
            cover_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), column_183, 'cover')
            # Calling cover(args, kwargs) (line 103)
            cover_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), cover_184, *[], **kwargs_185)
            
            
            # Assigning a Attribute to a Name (line 104):
            # Getting the type of 'j' (line 104)
            j_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'j')
            # Obtaining the member 'right' of a type (line 104)
            right_188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), j_187, 'right')
            # Assigning a type to the variable 'j' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'j', right_188)
            # SSA join for while statement (line 102)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to search(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'k' (line 105)
        k_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'k', False)
        int_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'int')
        # Applying the binary operator '+' (line 105)
        result_add_192 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), '+', k_190, int_191)
        
        # Processing the call keyword arguments (line 105)
        kwargs_193 = {}
        # Getting the type of 'search' (line 105)
        search_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'search', False)
        # Calling search(args, kwargs) (line 105)
        search_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), search_189, *[result_add_192], **kwargs_193)
        
        
        # Assigning a Name to a Name (line 106):
        # Getting the type of 'k' (line 106)
        k_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'k')
        # Assigning a type to the variable 'level' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'level', k_195)
        
        # Assigning a Call to a Name (line 107):
        
        # Call to pop(...): (line 107)
        # Processing the call arguments (line 107)
        int_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'int')
        # Processing the call keyword arguments (line 107)
        kwargs_199 = {}
        # Getting the type of 'o' (line 107)
        o_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'o', False)
        # Obtaining the member 'pop' of a type (line 107)
        pop_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), o_196, 'pop')
        # Calling pop(args, kwargs) (line 107)
        pop_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), pop_197, *[int_198], **kwargs_199)
        
        # Assigning a type to the variable 'r' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'r', pop_call_result_200)
        
        # Assigning a Attribute to a Name (line 108):
        # Getting the type of 'r' (line 108)
        r_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'r')
        # Obtaining the member 'column' of a type (line 108)
        column_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), r_201, 'column')
        # Assigning a type to the variable 'c' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'c', column_202)
        
        # Assigning a Attribute to a Name (line 109):
        # Getting the type of 'r' (line 109)
        r_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'r')
        # Obtaining the member 'left' of a type (line 109)
        left_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), r_203, 'left')
        # Assigning a type to the variable 'j' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'j', left_204)
        
        
        # Getting the type of 'j' (line 110)
        j_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'j')
        # Getting the type of 'r' (line 110)
        r_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'r')
        # Applying the binary operator '!=' (line 110)
        result_ne_207 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 14), '!=', j_205, r_206)
        
        # Testing if the while is going to be iterated (line 110)
        # Testing the type of an if condition (line 110)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_ne_207)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 8), result_ne_207):
            # SSA begins for while statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Call to uncover(...): (line 111)
            # Processing the call keyword arguments (line 111)
            kwargs_211 = {}
            # Getting the type of 'j' (line 111)
            j_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'j', False)
            # Obtaining the member 'column' of a type (line 111)
            column_209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), j_208, 'column')
            # Obtaining the member 'uncover' of a type (line 111)
            uncover_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), column_209, 'uncover')
            # Calling uncover(args, kwargs) (line 111)
            uncover_call_result_212 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), uncover_210, *[], **kwargs_211)
            
            
            # Assigning a Attribute to a Name (line 112):
            # Getting the type of 'j' (line 112)
            j_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'j')
            # Obtaining the member 'left' of a type (line 112)
            left_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), j_213, 'left')
            # Assigning a type to the variable 'j' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'j', left_214)
            # SSA join for while statement (line 110)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Attribute to a Name (line 113):
        # Getting the type of 'r' (line 113)
        r_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'r')
        # Obtaining the member 'down' of a type (line 113)
        down_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), r_215, 'down')
        # Assigning a type to the variable 'r' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'r', down_216)
        # SSA join for while statement (line 99)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to uncover(...): (line 114)
    # Processing the call keyword arguments (line 114)
    kwargs_219 = {}
    # Getting the type of 'c' (line 114)
    c_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'c', False)
    # Obtaining the member 'uncover' of a type (line 114)
    uncover_218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 4), c_217, 'uncover')
    # Calling uncover(args, kwargs) (line 114)
    uncover_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), uncover_218, *[], **kwargs_219)
    
    
    # Getting the type of 'k' (line 116)
    k_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'k')
    int_222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 12), 'int')
    # Applying the binary operator '==' (line 116)
    result_eq_223 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), '==', k_221, int_222)
    
    # Testing if the type of an if condition is none (line 116)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 4), result_eq_223):
        pass
    else:
        
        # Testing the type of an if condition (line 116)
        if_condition_224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_eq_223)
        # Assigning a type to the variable 'if_condition_224' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_224', if_condition_224)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 117):
        int_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 16), 'int')
        # Assigning a type to the variable 'count' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'count', int_225)
        
        # Assigning a Attribute to a Name (line 118):
        # Getting the type of 'root' (line 118)
        root_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'root')
        # Obtaining the member 'right' of a type (line 118)
        right_227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), root_226, 'right')
        # Assigning a type to the variable 'j' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'j', right_227)
        
        
        # Getting the type of 'j' (line 119)
        j_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'j')
        # Getting the type of 'root' (line 119)
        root_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'root')
        # Applying the binary operator '!=' (line 119)
        result_ne_230 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 14), '!=', j_228, root_229)
        
        # Testing if the while is going to be iterated (line 119)
        # Testing the type of an if condition (line 119)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_ne_230)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 119, 8), result_ne_230):
            # SSA begins for while statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 'count' (line 120)
            count_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'count')
            int_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'int')
            # Applying the binary operator '+=' (line 120)
            result_iadd_233 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), '+=', count_231, int_232)
            # Assigning a type to the variable 'count' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'count', result_iadd_233)
            
            
            # Assigning a Attribute to a Name (line 121):
            # Getting the type of 'j' (line 121)
            j_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'j')
            # Obtaining the member 'right' of a type (line 121)
            right_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), j_234, 'right')
            # Assigning a type to the variable 'j' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'j', right_235)
            # SSA join for while statement (line 119)
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'search(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'search' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_236)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'search'
    return stypy_return_type_236

# Assigning a type to the variable 'search' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'search', search)

@norecursion
def printsolution(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'printsolution'
    module_type_store = module_type_store.open_function_context('printsolution', 126, 0, False)
    
    # Passed parameters checking function
    printsolution.stypy_localization = localization
    printsolution.stypy_type_of_self = None
    printsolution.stypy_type_store = module_type_store
    printsolution.stypy_function_name = 'printsolution'
    printsolution.stypy_param_names_list = ['o']
    printsolution.stypy_varargs_param_name = None
    printsolution.stypy_kwargs_param_name = None
    printsolution.stypy_call_defaults = defaults
    printsolution.stypy_call_varargs = varargs
    printsolution.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'printsolution', ['o'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'printsolution', localization, ['o'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'printsolution(...)' code ##################

    
    # Getting the type of 'o' (line 128)
    o_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'o')
    # Testing if the for loop is going to be iterated (line 128)
    # Testing the type of a for loop iterable (line 128)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 128, 4), o_237)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 128, 4), o_237):
        # Getting the type of the for loop variable (line 128)
        for_loop_var_238 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 128, 4), o_237)
        # Assigning a type to the variable 'row' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'row', for_loop_var_238)
        # SSA begins for a for statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 129):
        # Getting the type of 'row' (line 129)
        row_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'row')
        # Assigning a type to the variable 'r' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'r', row_239)
        
        # Assigning a Attribute to a Name (line 130):
        # Getting the type of 'r' (line 130)
        r_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'r')
        # Obtaining the member 'column' of a type (line 130)
        column_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), r_240, 'column')
        # Obtaining the member 'name' of a type (line 130)
        name_242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), column_241, 'name')
        # Assigning a type to the variable 's' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 's', name_242)
        
        # Assigning a Attribute to a Name (line 131):
        # Getting the type of 'r' (line 131)
        r_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'r')
        # Obtaining the member 'right' of a type (line 131)
        right_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), r_243, 'right')
        # Assigning a type to the variable 'r' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'r', right_244)
        
        
        # Getting the type of 'r' (line 132)
        r_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'r')
        # Getting the type of 'row' (line 132)
        row_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'row')
        # Applying the binary operator '!=' (line 132)
        result_ne_247 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 14), '!=', r_245, row_246)
        
        # Testing if the while is going to be iterated (line 132)
        # Testing the type of an if condition (line 132)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_ne_247)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 132, 8), result_ne_247):
            # SSA begins for while statement (line 132)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 's' (line 133)
            s_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 's')
            str_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 17), 'str', ' ')
            # Getting the type of 'r' (line 133)
            r_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'r')
            # Obtaining the member 'column' of a type (line 133)
            column_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), r_250, 'column')
            # Obtaining the member 'name' of a type (line 133)
            name_252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), column_251, 'name')
            # Applying the binary operator '+' (line 133)
            result_add_253 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 17), '+', str_249, name_252)
            
            # Applying the binary operator '+=' (line 133)
            result_iadd_254 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), '+=', s_248, result_add_253)
            # Assigning a type to the variable 's' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 's', result_iadd_254)
            
            
            # Assigning a Attribute to a Name (line 134):
            # Getting the type of 'r' (line 134)
            r_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'r')
            # Obtaining the member 'right' of a type (line 134)
            right_256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), r_255, 'right')
            # Assigning a type to the variable 'r' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'r', right_256)
            # SSA join for while statement (line 132)
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'printsolution(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'printsolution' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'printsolution'
    return stypy_return_type_257

# Assigning a type to the variable 'printsolution' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'printsolution', printsolution)

@norecursion
def printmatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'printmatrix'
    module_type_store = module_type_store.open_function_context('printmatrix', 138, 0, False)
    
    # Passed parameters checking function
    printmatrix.stypy_localization = localization
    printmatrix.stypy_type_of_self = None
    printmatrix.stypy_type_store = module_type_store
    printmatrix.stypy_function_name = 'printmatrix'
    printmatrix.stypy_param_names_list = ['root']
    printmatrix.stypy_varargs_param_name = None
    printmatrix.stypy_kwargs_param_name = None
    printmatrix.stypy_call_defaults = defaults
    printmatrix.stypy_call_varargs = varargs
    printmatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'printmatrix', ['root'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'printmatrix', localization, ['root'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'printmatrix(...)' code ##################

    
    # Assigning a Attribute to a Name (line 139):
    # Getting the type of 'root' (line 139)
    root_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'root')
    # Obtaining the member 'right' of a type (line 139)
    right_259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), root_258, 'right')
    # Assigning a type to the variable 'c' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'c', right_259)
    
    
    # Getting the type of 'c' (line 140)
    c_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 10), 'c')
    # Getting the type of 'root' (line 140)
    root_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'root')
    # Applying the binary operator '!=' (line 140)
    result_ne_262 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 10), '!=', c_260, root_261)
    
    # Testing if the while is going to be iterated (line 140)
    # Testing the type of an if condition (line 140)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 4), result_ne_262)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 140, 4), result_ne_262):
        # SSA begins for while statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Attribute to a Name (line 141):
        # Getting the type of 'c' (line 141)
        c_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'c')
        # Obtaining the member 'down' of a type (line 141)
        down_264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), c_263, 'down')
        # Assigning a type to the variable 'r' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'r', down_264)
        
        
        # Getting the type of 'r' (line 142)
        r_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'r')
        # Getting the type of 'c' (line 142)
        c_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'c')
        # Applying the binary operator '!=' (line 142)
        result_ne_267 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 14), '!=', r_265, c_266)
        
        # Testing if the while is going to be iterated (line 142)
        # Testing the type of an if condition (line 142)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), result_ne_267)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 142, 8), result_ne_267):
            # SSA begins for while statement (line 142)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Call to printrow(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'r' (line 143)
            r_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'r', False)
            # Processing the call keyword arguments (line 143)
            kwargs_270 = {}
            # Getting the type of 'printrow' (line 143)
            printrow_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'printrow', False)
            # Calling printrow(args, kwargs) (line 143)
            printrow_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), printrow_268, *[r_269], **kwargs_270)
            
            
            # Assigning a Attribute to a Name (line 144):
            # Getting the type of 'r' (line 144)
            r_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'r')
            # Obtaining the member 'down' of a type (line 144)
            down_273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), r_272, 'down')
            # Assigning a type to the variable 'r' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'r', down_273)
            # SSA join for while statement (line 142)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Attribute to a Name (line 145):
        # Getting the type of 'c' (line 145)
        c_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'c')
        # Obtaining the member 'right' of a type (line 145)
        right_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), c_274, 'right')
        # Assigning a type to the variable 'c' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'c', right_275)
        # SSA join for while statement (line 140)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'printmatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'printmatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_276)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'printmatrix'
    return stypy_return_type_276

# Assigning a type to the variable 'printmatrix' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'printmatrix', printmatrix)

@norecursion
def printrow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'printrow'
    module_type_store = module_type_store.open_function_context('printrow', 148, 0, False)
    
    # Passed parameters checking function
    printrow.stypy_localization = localization
    printrow.stypy_type_of_self = None
    printrow.stypy_type_store = module_type_store
    printrow.stypy_function_name = 'printrow'
    printrow.stypy_param_names_list = ['r']
    printrow.stypy_varargs_param_name = None
    printrow.stypy_kwargs_param_name = None
    printrow.stypy_call_defaults = defaults
    printrow.stypy_call_varargs = varargs
    printrow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'printrow', ['r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'printrow', localization, ['r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'printrow(...)' code ##################

    
    # Assigning a Attribute to a Name (line 149):
    # Getting the type of 'r' (line 149)
    r_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'r')
    # Obtaining the member 'column' of a type (line 149)
    column_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), r_277, 'column')
    # Obtaining the member 'name' of a type (line 149)
    name_279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), column_278, 'name')
    # Assigning a type to the variable 's' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 's', name_279)
    
    # Assigning a Attribute to a Name (line 150):
    # Getting the type of 'r' (line 150)
    r_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'r')
    # Obtaining the member 'right' of a type (line 150)
    right_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), r_280, 'right')
    # Assigning a type to the variable 'next' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'next', right_281)
    
    
    # Getting the type of 'next' (line 151)
    next_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 10), 'next')
    # Getting the type of 'r' (line 151)
    r_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'r')
    # Applying the binary operator '!=' (line 151)
    result_ne_284 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 10), '!=', next_282, r_283)
    
    # Testing if the while is going to be iterated (line 151)
    # Testing the type of an if condition (line 151)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_ne_284)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 151, 4), result_ne_284):
        # SSA begins for while statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 's' (line 152)
        s_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 's')
        str_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 13), 'str', ' ')
        # Getting the type of 'next' (line 152)
        next_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'next')
        # Obtaining the member 'column' of a type (line 152)
        column_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 19), next_287, 'column')
        # Obtaining the member 'name' of a type (line 152)
        name_289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 19), column_288, 'name')
        # Applying the binary operator '+' (line 152)
        result_add_290 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 13), '+', str_286, name_289)
        
        # Applying the binary operator '+=' (line 152)
        result_iadd_291 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 8), '+=', s_285, result_add_290)
        # Assigning a type to the variable 's' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 's', result_iadd_291)
        
        
        # Assigning a Attribute to a Name (line 153):
        # Getting the type of 'next' (line 153)
        next_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'next')
        # Obtaining the member 'right' of a type (line 153)
        right_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), next_292, 'right')
        # Assigning a type to the variable 'next' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'next', right_293)
        # SSA join for while statement (line 151)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'printrow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'printrow' in the type store
    # Getting the type of 'stypy_return_type' (line 148)
    stypy_return_type_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'printrow'
    return stypy_return_type_294

# Assigning a type to the variable 'printrow' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'printrow', printrow)

@norecursion
def setroot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setroot'
    module_type_store = module_type_store.open_function_context('setroot', 157, 0, False)
    
    # Passed parameters checking function
    setroot.stypy_localization = localization
    setroot.stypy_type_of_self = None
    setroot.stypy_type_store = module_type_store
    setroot.stypy_function_name = 'setroot'
    setroot.stypy_param_names_list = ['r']
    setroot.stypy_varargs_param_name = None
    setroot.stypy_kwargs_param_name = None
    setroot.stypy_call_defaults = defaults
    setroot.stypy_call_varargs = varargs
    setroot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setroot', ['r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setroot', localization, ['r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setroot(...)' code ##################

    # Marking variables as global (line 158)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 158, 4), 'root')
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'r' (line 159)
    r_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'r')
    # Assigning a type to the variable 'root' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'root', r_295)
    
    # ################# End of 'setroot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setroot' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setroot'
    return stypy_return_type_296

# Assigning a type to the variable 'setroot' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'setroot', setroot)

# Assigning a Num to a Name (line 162):
int_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'int')
# Assigning a type to the variable 'solutions' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'solutions', int_297)

# Assigning a List to a Name (line 163):

# Obtaining an instance of the builtin type 'list' (line 163)
list_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 163)

# Assigning a type to the variable 'o' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'o', list_298)

# Assigning a Num to a Name (line 169):
int_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 7), 'int')
# Assigning a type to the variable 'Lcol' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'Lcol', int_299)

# Assigning a Num to a Name (line 170):
int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 7), 'int')
# Assigning a type to the variable 'Lrow' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'Lrow', int_300)

@norecursion
def matrixmultiply(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matrixmultiply'
    module_type_store = module_type_store.open_function_context('matrixmultiply', 175, 0, False)
    
    # Passed parameters checking function
    matrixmultiply.stypy_localization = localization
    matrixmultiply.stypy_type_of_self = None
    matrixmultiply.stypy_type_store = module_type_store
    matrixmultiply.stypy_function_name = 'matrixmultiply'
    matrixmultiply.stypy_param_names_list = ['m', 'n']
    matrixmultiply.stypy_varargs_param_name = None
    matrixmultiply.stypy_kwargs_param_name = None
    matrixmultiply.stypy_call_defaults = defaults
    matrixmultiply.stypy_call_varargs = varargs
    matrixmultiply.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matrixmultiply', ['m', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matrixmultiply', localization, ['m', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matrixmultiply(...)' code ##################

    
    # Assigning a List to a Name (line 176):
    
    # Obtaining an instance of the builtin type 'list' (line 176)
    list_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 176)
    # Adding element type (line 176)
    
    # Obtaining an instance of the builtin type 'list' (line 176)
    list_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 176)
    # Adding element type (line 176)
    int_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 9), list_302, int_303)
    # Adding element type (line 176)
    int_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 9), list_302, int_304)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 8), list_301, list_302)
    # Adding element type (line 176)
    
    # Obtaining an instance of the builtin type 'list' (line 176)
    list_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 176)
    # Adding element type (line 176)
    int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 17), list_305, int_306)
    # Adding element type (line 176)
    int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 17), list_305, int_307)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 8), list_301, list_305)
    
    # Assigning a type to the variable 'r' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'r', list_301)
    
    
    # Call to range(...): (line 177)
    # Processing the call arguments (line 177)
    int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 19), 'int')
    # Processing the call keyword arguments (line 177)
    kwargs_310 = {}
    # Getting the type of 'range' (line 177)
    range_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'range', False)
    # Calling range(args, kwargs) (line 177)
    range_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), range_308, *[int_309], **kwargs_310)
    
    # Testing if the for loop is going to be iterated (line 177)
    # Testing the type of a for loop iterable (line 177)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 4), range_call_result_311)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 177, 4), range_call_result_311):
        # Getting the type of the for loop variable (line 177)
        for_loop_var_312 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 4), range_call_result_311)
        # Assigning a type to the variable 'i' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'i', for_loop_var_312)
        # SSA begins for a for statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 178)
        # Processing the call arguments (line 178)
        int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'int')
        # Processing the call keyword arguments (line 178)
        kwargs_315 = {}
        # Getting the type of 'range' (line 178)
        range_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'range', False)
        # Calling range(args, kwargs) (line 178)
        range_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 178, 17), range_313, *[int_314], **kwargs_315)
        
        # Testing if the for loop is going to be iterated (line 178)
        # Testing the type of a for loop iterable (line 178)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 178, 8), range_call_result_316)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 178, 8), range_call_result_316):
            # Getting the type of the for loop variable (line 178)
            for_loop_var_317 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 178, 8), range_call_result_316)
            # Assigning a type to the variable 'j' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'j', for_loop_var_317)
            # SSA begins for a for statement (line 178)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 179):
            int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 18), 'int')
            # Assigning a type to the variable 'sum' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'sum', int_318)
            
            
            # Call to range(...): (line 180)
            # Processing the call arguments (line 180)
            int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 27), 'int')
            # Processing the call keyword arguments (line 180)
            kwargs_321 = {}
            # Getting the type of 'range' (line 180)
            range_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'range', False)
            # Calling range(args, kwargs) (line 180)
            range_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 180, 21), range_319, *[int_320], **kwargs_321)
            
            # Testing if the for loop is going to be iterated (line 180)
            # Testing the type of a for loop iterable (line 180)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 12), range_call_result_322)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 180, 12), range_call_result_322):
                # Getting the type of the for loop variable (line 180)
                for_loop_var_323 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 12), range_call_result_322)
                # Assigning a type to the variable 'k' (line 180)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'k', for_loop_var_323)
                # SSA begins for a for statement (line 180)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'sum' (line 181)
                sum_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'sum')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 181)
                k_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'k')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 181)
                i_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'i')
                # Getting the type of 'm' (line 181)
                m_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'm')
                # Obtaining the member '__getitem__' of a type (line 181)
                getitem___328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 23), m_327, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 181)
                subscript_call_result_329 = invoke(stypy.reporting.localization.Localization(__file__, 181, 23), getitem___328, i_326)
                
                # Obtaining the member '__getitem__' of a type (line 181)
                getitem___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 23), subscript_call_result_329, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 181)
                subscript_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 181, 23), getitem___330, k_325)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 181)
                j_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 38), 'j')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 181)
                k_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 35), 'k')
                # Getting the type of 'n' (line 181)
                n_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'n')
                # Obtaining the member '__getitem__' of a type (line 181)
                getitem___335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 33), n_334, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 181)
                subscript_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 181, 33), getitem___335, k_333)
                
                # Obtaining the member '__getitem__' of a type (line 181)
                getitem___337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 33), subscript_call_result_336, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 181)
                subscript_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 181, 33), getitem___337, j_332)
                
                # Applying the binary operator '*' (line 181)
                result_mul_339 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 23), '*', subscript_call_result_331, subscript_call_result_338)
                
                # Applying the binary operator '+=' (line 181)
                result_iadd_340 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 16), '+=', sum_324, result_mul_339)
                # Assigning a type to the variable 'sum' (line 181)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'sum', result_iadd_340)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Name to a Subscript (line 182):
            # Getting the type of 'sum' (line 182)
            sum_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'sum')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 182)
            i_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'i')
            # Getting the type of 'r' (line 182)
            r_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'r')
            # Obtaining the member '__getitem__' of a type (line 182)
            getitem___344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), r_343, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 182)
            subscript_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), getitem___344, i_342)
            
            # Getting the type of 'j' (line 182)
            j_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'j')
            # Storing an element on a container (line 182)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 12), subscript_call_result_345, (j_346, sum_341))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'r' (line 183)
    r_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', r_347)
    
    # ################# End of 'matrixmultiply(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matrixmultiply' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_348)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matrixmultiply'
    return stypy_return_type_348

# Assigning a type to the variable 'matrixmultiply' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'matrixmultiply', matrixmultiply)

@norecursion
def matrixact(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matrixact'
    module_type_store = module_type_store.open_function_context('matrixact', 186, 0, False)
    
    # Passed parameters checking function
    matrixact.stypy_localization = localization
    matrixact.stypy_type_of_self = None
    matrixact.stypy_type_store = module_type_store
    matrixact.stypy_function_name = 'matrixact'
    matrixact.stypy_param_names_list = ['m', 'v']
    matrixact.stypy_varargs_param_name = None
    matrixact.stypy_kwargs_param_name = None
    matrixact.stypy_call_defaults = defaults
    matrixact.stypy_call_varargs = varargs
    matrixact.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matrixact', ['m', 'v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matrixact', localization, ['m', 'v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matrixact(...)' code ##################

    
    # Assigning a List to a Name (line 187):
    
    # Obtaining an instance of the builtin type 'list' (line 187)
    list_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 187)
    # Adding element type (line 187)
    int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 8), list_349, int_350)
    # Adding element type (line 187)
    int_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 8), list_349, int_351)
    
    # Assigning a type to the variable 'u' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'u', list_349)
    
    
    # Call to range(...): (line 188)
    # Processing the call arguments (line 188)
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 19), 'int')
    # Processing the call keyword arguments (line 188)
    kwargs_354 = {}
    # Getting the type of 'range' (line 188)
    range_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'range', False)
    # Calling range(args, kwargs) (line 188)
    range_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 188, 13), range_352, *[int_353], **kwargs_354)
    
    # Testing if the for loop is going to be iterated (line 188)
    # Testing the type of a for loop iterable (line 188)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 4), range_call_result_355)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 188, 4), range_call_result_355):
        # Getting the type of the for loop variable (line 188)
        for_loop_var_356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 4), range_call_result_355)
        # Assigning a type to the variable 'i' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'i', for_loop_var_356)
        # SSA begins for a for statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Name (line 189):
        int_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 14), 'int')
        # Assigning a type to the variable 'sum' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'sum', int_357)
        
        
        # Call to range(...): (line 190)
        # Processing the call arguments (line 190)
        int_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 23), 'int')
        # Processing the call keyword arguments (line 190)
        kwargs_360 = {}
        # Getting the type of 'range' (line 190)
        range_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'range', False)
        # Calling range(args, kwargs) (line 190)
        range_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), range_358, *[int_359], **kwargs_360)
        
        # Testing if the for loop is going to be iterated (line 190)
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_361)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_361):
            # Getting the type of the for loop variable (line 190)
            for_loop_var_362 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_361)
            # Assigning a type to the variable 'j' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'j', for_loop_var_362)
            # SSA begins for a for statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'sum' (line 191)
            sum_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'sum')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 191)
            j_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'j')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 191)
            i_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'i')
            # Getting the type of 'm' (line 191)
            m_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'm')
            # Obtaining the member '__getitem__' of a type (line 191)
            getitem___367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 19), m_366, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 191)
            subscript_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), getitem___367, i_365)
            
            # Obtaining the member '__getitem__' of a type (line 191)
            getitem___369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 19), subscript_call_result_368, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 191)
            subscript_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), getitem___369, j_364)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 191)
            j_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 31), 'j')
            # Getting the type of 'v' (line 191)
            v_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 29), 'v')
            # Obtaining the member '__getitem__' of a type (line 191)
            getitem___373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 29), v_372, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 191)
            subscript_call_result_374 = invoke(stypy.reporting.localization.Localization(__file__, 191, 29), getitem___373, j_371)
            
            # Applying the binary operator '*' (line 191)
            result_mul_375 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 19), '*', subscript_call_result_370, subscript_call_result_374)
            
            # Applying the binary operator '+=' (line 191)
            result_iadd_376 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 12), '+=', sum_363, result_mul_375)
            # Assigning a type to the variable 'sum' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'sum', result_iadd_376)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Subscript (line 192):
        # Getting the type of 'sum' (line 192)
        sum_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'sum')
        # Getting the type of 'u' (line 192)
        u_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'u')
        # Getting the type of 'i' (line 192)
        i_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 10), 'i')
        # Storing an element on a container (line 192)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 8), u_378, (i_379, sum_377))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'u' (line 193)
    u_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'u')
    # Assigning a type to the variable 'stypy_return_type' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type', u_380)
    
    # ################# End of 'matrixact(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matrixact' in the type store
    # Getting the type of 'stypy_return_type' (line 186)
    stypy_return_type_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matrixact'
    return stypy_return_type_381

# Assigning a type to the variable 'matrixact' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'matrixact', matrixact)

# Assigning a List to a Name (line 197):

# Obtaining an instance of the builtin type 'list' (line 197)
list_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 197)
# Adding element type (line 197)

# Obtaining an instance of the builtin type 'list' (line 197)
list_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 197)
# Adding element type (line 197)
int_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 12), list_383, int_384)
# Adding element type (line 197)
int_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 12), list_383, int_385)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 11), list_382, list_383)
# Adding element type (line 197)

# Obtaining an instance of the builtin type 'list' (line 197)
list_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 197)
# Adding element type (line 197)
int_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 20), list_386, int_387)
# Adding element type (line 197)
int_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 20), list_386, int_388)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 11), list_382, list_386)

# Assigning a type to the variable 'identity' (line 197)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'identity', list_382)

# Assigning a List to a Name (line 198):

# Obtaining an instance of the builtin type 'list' (line 198)
list_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 198)
# Adding element type (line 198)

# Obtaining an instance of the builtin type 'list' (line 198)
list_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 198)
# Adding element type (line 198)
int_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 7), list_390, int_391)
# Adding element type (line 198)
int_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 7), list_390, int_392)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 6), list_389, list_390)
# Adding element type (line 198)

# Obtaining an instance of the builtin type 'list' (line 198)
list_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 198)
# Adding element type (line 198)
int_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 16), list_393, int_394)
# Adding element type (line 198)
int_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 16), list_393, int_395)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 6), list_389, list_393)

# Assigning a type to the variable 'r90' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'r90', list_389)

# Assigning a List to a Name (line 199):

# Obtaining an instance of the builtin type 'list' (line 199)
list_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 199)
# Adding element type (line 199)

# Obtaining an instance of the builtin type 'list' (line 199)
list_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 199)
# Adding element type (line 199)
int_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 8), list_397, int_398)
# Adding element type (line 199)
int_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 8), list_397, int_399)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 7), list_396, list_397)
# Adding element type (line 199)

# Obtaining an instance of the builtin type 'list' (line 199)
list_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 199)
# Adding element type (line 199)
int_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 17), list_400, int_401)
# Adding element type (line 199)
int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 17), list_400, int_402)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 7), list_396, list_400)

# Assigning a type to the variable 'r180' (line 199)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 0), 'r180', list_396)

# Assigning a List to a Name (line 200):

# Obtaining an instance of the builtin type 'list' (line 200)
list_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 200)
# Adding element type (line 200)

# Obtaining an instance of the builtin type 'list' (line 200)
list_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 200)
# Adding element type (line 200)
int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 8), list_404, int_405)
# Adding element type (line 200)
int_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 8), list_404, int_406)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 7), list_403, list_404)
# Adding element type (line 200)

# Obtaining an instance of the builtin type 'list' (line 200)
list_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 200)
# Adding element type (line 200)
int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), list_407, int_408)
# Adding element type (line 200)
int_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), list_407, int_409)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 7), list_403, list_407)

# Assigning a type to the variable 'r270' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'r270', list_403)

# Assigning a List to a Name (line 201):

# Obtaining an instance of the builtin type 'list' (line 201)
list_410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 201)
# Adding element type (line 201)

# Obtaining an instance of the builtin type 'list' (line 201)
list_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 201)
# Adding element type (line 201)
int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 6), list_411, int_412)
# Adding element type (line 201)
int_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 6), list_411, int_413)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), list_410, list_411)
# Adding element type (line 201)

# Obtaining an instance of the builtin type 'list' (line 201)
list_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 201)
# Adding element type (line 201)
int_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 14), list_414, int_415)
# Adding element type (line 201)
int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 14), list_414, int_416)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), list_410, list_414)

# Assigning a type to the variable 'r1' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'r1', list_410)

# Assigning a Call to a Name (line 202):

# Call to matrixmultiply(...): (line 202)
# Processing the call arguments (line 202)
# Getting the type of 'r1' (line 202)
r1_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'r1', False)
# Getting the type of 'r90' (line 202)
r90_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'r90', False)
# Processing the call keyword arguments (line 202)
kwargs_420 = {}
# Getting the type of 'matrixmultiply' (line 202)
matrixmultiply_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 5), 'matrixmultiply', False)
# Calling matrixmultiply(args, kwargs) (line 202)
matrixmultiply_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 202, 5), matrixmultiply_417, *[r1_418, r90_419], **kwargs_420)

# Assigning a type to the variable 'r2' (line 202)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'r2', matrixmultiply_call_result_421)

# Assigning a Call to a Name (line 203):

# Call to matrixmultiply(...): (line 203)
# Processing the call arguments (line 203)
# Getting the type of 'r1' (line 203)
r1_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'r1', False)
# Getting the type of 'r180' (line 203)
r180_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'r180', False)
# Processing the call keyword arguments (line 203)
kwargs_425 = {}
# Getting the type of 'matrixmultiply' (line 203)
matrixmultiply_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 5), 'matrixmultiply', False)
# Calling matrixmultiply(args, kwargs) (line 203)
matrixmultiply_call_result_426 = invoke(stypy.reporting.localization.Localization(__file__, 203, 5), matrixmultiply_422, *[r1_423, r180_424], **kwargs_425)

# Assigning a type to the variable 'r3' (line 203)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'r3', matrixmultiply_call_result_426)

# Assigning a Call to a Name (line 204):

# Call to matrixmultiply(...): (line 204)
# Processing the call arguments (line 204)
# Getting the type of 'r1' (line 204)
r1_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'r1', False)
# Getting the type of 'r270' (line 204)
r270_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'r270', False)
# Processing the call keyword arguments (line 204)
kwargs_430 = {}
# Getting the type of 'matrixmultiply' (line 204)
matrixmultiply_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 5), 'matrixmultiply', False)
# Calling matrixmultiply(args, kwargs) (line 204)
matrixmultiply_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 204, 5), matrixmultiply_427, *[r1_428, r270_429], **kwargs_430)

# Assigning a type to the variable 'r4' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'r4', matrixmultiply_call_result_431)

# Assigning a List to a Name (line 208):

# Obtaining an instance of the builtin type 'list' (line 208)
list_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 208)
# Adding element type (line 208)
# Getting the type of 'identity' (line 208)
identity_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 14), 'identity')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, identity_433)
# Adding element type (line 208)
# Getting the type of 'r90' (line 208)
r90_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'r90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, r90_434)
# Adding element type (line 208)
# Getting the type of 'r180' (line 208)
r180_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'r180')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, r180_435)
# Adding element type (line 208)
# Getting the type of 'r270' (line 208)
r270_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 35), 'r270')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, r270_436)
# Adding element type (line 208)
# Getting the type of 'r1' (line 208)
r1_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 41), 'r1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, r1_437)
# Adding element type (line 208)
# Getting the type of 'r2' (line 208)
r2_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 45), 'r2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, r2_438)
# Adding element type (line 208)
# Getting the type of 'r3' (line 208)
r3_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 49), 'r3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, r3_439)
# Adding element type (line 208)
# Getting the type of 'r4' (line 208)
r4_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 53), 'r4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), list_432, r4_440)

# Assigning a type to the variable 'symmetries' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'symmetries', list_432)

# Assigning a List to a Name (line 209):

# Obtaining an instance of the builtin type 'list' (line 209)
list_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 209)
# Adding element type (line 209)
# Getting the type of 'identity' (line 209)
identity_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'identity')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_441, identity_442)
# Adding element type (line 209)
# Getting the type of 'r90' (line 209)
r90_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'r90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_441, r90_443)
# Adding element type (line 209)
# Getting the type of 'r180' (line 209)
r180_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 28), 'r180')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_441, r180_444)
# Adding element type (line 209)
# Getting the type of 'r270' (line 209)
r270_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'r270')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_441, r270_445)

# Assigning a type to the variable 'rotations' (line 209)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'rotations', list_441)
# Declaration of the 'Omino' class

class Omino:

    @norecursion
    def getorientations(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getorientations'
        module_type_store = module_type_store.open_function_context('getorientations', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Omino.getorientations.__dict__.__setitem__('stypy_localization', localization)
        Omino.getorientations.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Omino.getorientations.__dict__.__setitem__('stypy_type_store', module_type_store)
        Omino.getorientations.__dict__.__setitem__('stypy_function_name', 'Omino.getorientations')
        Omino.getorientations.__dict__.__setitem__('stypy_param_names_list', [])
        Omino.getorientations.__dict__.__setitem__('stypy_varargs_param_name', None)
        Omino.getorientations.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Omino.getorientations.__dict__.__setitem__('stypy_call_defaults', defaults)
        Omino.getorientations.__dict__.__setitem__('stypy_call_varargs', varargs)
        Omino.getorientations.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Omino.getorientations.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Omino.getorientations', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getorientations', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getorientations(...)' code ##################

        
        # Assigning a List to a Name (line 216):
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        
        # Assigning a type to the variable 'orientations' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'orientations', list_446)
        
        # Getting the type of 'self' (line 217)
        self_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'self')
        # Obtaining the member 'cosets' of a type (line 217)
        cosets_448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 24), self_447, 'cosets')
        # Testing if the for loop is going to be iterated (line 217)
        # Testing the type of a for loop iterable (line 217)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 217, 8), cosets_448)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 217, 8), cosets_448):
            # Getting the type of the for loop variable (line 217)
            for_loop_var_449 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 217, 8), cosets_448)
            # Assigning a type to the variable 'symmetry' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'symmetry', for_loop_var_449)
            # SSA begins for a for statement (line 217)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Name (line 218):
            
            # Obtaining an instance of the builtin type 'list' (line 218)
            list_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 218)
            
            # Assigning a type to the variable 'orientation' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'orientation', list_450)
            
            # Getting the type of 'self' (line 219)
            self_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'self')
            # Obtaining the member 'cells' of a type (line 219)
            cells_452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 24), self_451, 'cells')
            # Testing if the for loop is going to be iterated (line 219)
            # Testing the type of a for loop iterable (line 219)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 12), cells_452)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 219, 12), cells_452):
                # Getting the type of the for loop variable (line 219)
                for_loop_var_453 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 12), cells_452)
                # Assigning a type to the variable 'cell' (line 219)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'cell', for_loop_var_453)
                # SSA begins for a for statement (line 219)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 220)
                # Processing the call arguments (line 220)
                
                # Call to matrixact(...): (line 220)
                # Processing the call arguments (line 220)
                # Getting the type of 'symmetry' (line 220)
                symmetry_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 45), 'symmetry', False)
                # Getting the type of 'cell' (line 220)
                cell_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 55), 'cell', False)
                # Processing the call keyword arguments (line 220)
                kwargs_459 = {}
                # Getting the type of 'matrixact' (line 220)
                matrixact_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 35), 'matrixact', False)
                # Calling matrixact(args, kwargs) (line 220)
                matrixact_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 220, 35), matrixact_456, *[symmetry_457, cell_458], **kwargs_459)
                
                # Processing the call keyword arguments (line 220)
                kwargs_461 = {}
                # Getting the type of 'orientation' (line 220)
                orientation_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'orientation', False)
                # Obtaining the member 'append' of a type (line 220)
                append_455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), orientation_454, 'append')
                # Calling append(args, kwargs) (line 220)
                append_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), append_455, *[matrixact_call_result_460], **kwargs_461)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to append(...): (line 221)
            # Processing the call arguments (line 221)
            # Getting the type of 'orientation' (line 221)
            orientation_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'orientation', False)
            # Processing the call keyword arguments (line 221)
            kwargs_466 = {}
            # Getting the type of 'orientations' (line 221)
            orientations_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'orientations', False)
            # Obtaining the member 'append' of a type (line 221)
            append_464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), orientations_463, 'append')
            # Calling append(args, kwargs) (line 221)
            append_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), append_464, *[orientation_465], **kwargs_466)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 222):
        # Getting the type of 'orientations' (line 222)
        orientations_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'orientations')
        # Getting the type of 'self' (line 222)
        self_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self')
        # Setting the type of the member 'orientations' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_469, 'orientations', orientations_468)
        
        # ################# End of 'getorientations(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getorientations' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getorientations'
        return stypy_return_type_470


    @norecursion
    def move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'move'
        module_type_store = module_type_store.open_function_context('move', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Omino.move.__dict__.__setitem__('stypy_localization', localization)
        Omino.move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Omino.move.__dict__.__setitem__('stypy_type_store', module_type_store)
        Omino.move.__dict__.__setitem__('stypy_function_name', 'Omino.move')
        Omino.move.__dict__.__setitem__('stypy_param_names_list', ['v'])
        Omino.move.__dict__.__setitem__('stypy_varargs_param_name', None)
        Omino.move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Omino.move.__dict__.__setitem__('stypy_call_defaults', defaults)
        Omino.move.__dict__.__setitem__('stypy_call_varargs', varargs)
        Omino.move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Omino.move.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Omino.move', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'move', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'move(...)' code ##################

        
        # Assigning a List to a Name (line 225):
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        
        # Assigning a type to the variable 'newcells' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'newcells', list_471)
        
        # Getting the type of 'self' (line 226)
        self_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'self')
        # Obtaining the member 'cells' of a type (line 226)
        cells_473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 20), self_472, 'cells')
        # Testing if the for loop is going to be iterated (line 226)
        # Testing the type of a for loop iterable (line 226)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), cells_473)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 226, 8), cells_473):
            # Getting the type of the for loop variable (line 226)
            for_loop_var_474 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), cells_473)
            # Assigning a type to the variable 'cell' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'cell', for_loop_var_474)
            # SSA begins for a for statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 227)
            # Processing the call arguments (line 227)
            
            # Obtaining an instance of the builtin type 'list' (line 227)
            list_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 227)
            # Adding element type (line 227)
            
            # Obtaining the type of the subscript
            int_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 34), 'int')
            # Getting the type of 'cell' (line 227)
            cell_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'cell', False)
            # Obtaining the member '__getitem__' of a type (line 227)
            getitem___480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 29), cell_479, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 227)
            subscript_call_result_481 = invoke(stypy.reporting.localization.Localization(__file__, 227, 29), getitem___480, int_478)
            
            
            # Obtaining the type of the subscript
            int_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 41), 'int')
            # Getting the type of 'v' (line 227)
            v_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 39), 'v', False)
            # Obtaining the member '__getitem__' of a type (line 227)
            getitem___484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 39), v_483, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 227)
            subscript_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 227, 39), getitem___484, int_482)
            
            # Applying the binary operator '+' (line 227)
            result_add_486 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 29), '+', subscript_call_result_481, subscript_call_result_485)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 28), list_477, result_add_486)
            # Adding element type (line 227)
            
            # Obtaining the type of the subscript
            int_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 50), 'int')
            # Getting the type of 'cell' (line 227)
            cell_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 45), 'cell', False)
            # Obtaining the member '__getitem__' of a type (line 227)
            getitem___489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 45), cell_488, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 227)
            subscript_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 227, 45), getitem___489, int_487)
            
            
            # Obtaining the type of the subscript
            int_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 57), 'int')
            # Getting the type of 'v' (line 227)
            v_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 55), 'v', False)
            # Obtaining the member '__getitem__' of a type (line 227)
            getitem___493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 55), v_492, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 227)
            subscript_call_result_494 = invoke(stypy.reporting.localization.Localization(__file__, 227, 55), getitem___493, int_491)
            
            # Applying the binary operator '+' (line 227)
            result_add_495 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 45), '+', subscript_call_result_490, subscript_call_result_494)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 28), list_477, result_add_495)
            
            # Processing the call keyword arguments (line 227)
            kwargs_496 = {}
            # Getting the type of 'newcells' (line 227)
            newcells_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'newcells', False)
            # Obtaining the member 'append' of a type (line 227)
            append_476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), newcells_475, 'append')
            # Calling append(args, kwargs) (line 227)
            append_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), append_476, *[list_477], **kwargs_496)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 228):
        # Getting the type of 'newcells' (line 228)
        newcells_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'newcells')
        # Getting the type of 'self' (line 228)
        self_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_499, 'cells', newcells_498)
        
        # ################# End of 'move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move' in the type store
        # Getting the type of 'stypy_return_type' (line 224)
        stypy_return_type_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move'
        return stypy_return_type_500


    @norecursion
    def translate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'translate'
        module_type_store = module_type_store.open_function_context('translate', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Omino.translate.__dict__.__setitem__('stypy_localization', localization)
        Omino.translate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Omino.translate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Omino.translate.__dict__.__setitem__('stypy_function_name', 'Omino.translate')
        Omino.translate.__dict__.__setitem__('stypy_param_names_list', ['v'])
        Omino.translate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Omino.translate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Omino.translate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Omino.translate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Omino.translate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Omino.translate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Omino.translate', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'translate', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'translate(...)' code ##################

        
        # Assigning a List to a Name (line 231):
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        
        # Assigning a type to the variable 'r' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'r', list_501)
        
        # Getting the type of 'self' (line 232)
        self_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'self')
        # Obtaining the member 'orientations' of a type (line 232)
        orientations_503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 27), self_502, 'orientations')
        # Testing if the for loop is going to be iterated (line 232)
        # Testing the type of a for loop iterable (line 232)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 232, 8), orientations_503)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 232, 8), orientations_503):
            # Getting the type of the for loop variable (line 232)
            for_loop_var_504 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 232, 8), orientations_503)
            # Assigning a type to the variable 'orientation' (line 232)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'orientation', for_loop_var_504)
            # SSA begins for a for statement (line 232)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Name (line 233):
            
            # Obtaining an instance of the builtin type 'list' (line 233)
            list_505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'list')
            # Adding type elements to the builtin type 'list' instance (line 233)
            
            # Assigning a type to the variable 's' (line 233)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 's', list_505)
            
            # Getting the type of 'orientation' (line 234)
            orientation_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'orientation')
            # Testing if the for loop is going to be iterated (line 234)
            # Testing the type of a for loop iterable (line 234)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 234, 12), orientation_506)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 234, 12), orientation_506):
                # Getting the type of the for loop variable (line 234)
                for_loop_var_507 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 234, 12), orientation_506)
                # Assigning a type to the variable 'cell' (line 234)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'cell', for_loop_var_507)
                # SSA begins for a for statement (line 234)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 235)
                # Processing the call arguments (line 235)
                
                # Obtaining an instance of the builtin type 'list' (line 235)
                list_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 25), 'list')
                # Adding type elements to the builtin type 'list' instance (line 235)
                # Adding element type (line 235)
                
                # Obtaining the type of the subscript
                int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 31), 'int')
                # Getting the type of 'cell' (line 235)
                cell_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'cell', False)
                # Obtaining the member '__getitem__' of a type (line 235)
                getitem___513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 26), cell_512, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 235)
                subscript_call_result_514 = invoke(stypy.reporting.localization.Localization(__file__, 235, 26), getitem___513, int_511)
                
                
                # Obtaining the type of the subscript
                int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 38), 'int')
                # Getting the type of 'v' (line 235)
                v_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 36), 'v', False)
                # Obtaining the member '__getitem__' of a type (line 235)
                getitem___517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 36), v_516, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 235)
                subscript_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 235, 36), getitem___517, int_515)
                
                # Applying the binary operator '+' (line 235)
                result_add_519 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 26), '+', subscript_call_result_514, subscript_call_result_518)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 25), list_510, result_add_519)
                # Adding element type (line 235)
                
                # Obtaining the type of the subscript
                int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 47), 'int')
                # Getting the type of 'cell' (line 235)
                cell_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 42), 'cell', False)
                # Obtaining the member '__getitem__' of a type (line 235)
                getitem___522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 42), cell_521, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 235)
                subscript_call_result_523 = invoke(stypy.reporting.localization.Localization(__file__, 235, 42), getitem___522, int_520)
                
                
                # Obtaining the type of the subscript
                int_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 54), 'int')
                # Getting the type of 'v' (line 235)
                v_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'v', False)
                # Obtaining the member '__getitem__' of a type (line 235)
                getitem___526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 52), v_525, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 235)
                subscript_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 235, 52), getitem___526, int_524)
                
                # Applying the binary operator '+' (line 235)
                result_add_528 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 42), '+', subscript_call_result_523, subscript_call_result_527)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 25), list_510, result_add_528)
                
                # Processing the call keyword arguments (line 235)
                kwargs_529 = {}
                # Getting the type of 's' (line 235)
                s_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 's', False)
                # Obtaining the member 'append' of a type (line 235)
                append_509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), s_508, 'append')
                # Calling append(args, kwargs) (line 235)
                append_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), append_509, *[list_510], **kwargs_529)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to append(...): (line 236)
            # Processing the call arguments (line 236)
            # Getting the type of 's' (line 236)
            s_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 21), 's', False)
            # Processing the call keyword arguments (line 236)
            kwargs_534 = {}
            # Getting the type of 'r' (line 236)
            r_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'r', False)
            # Obtaining the member 'append' of a type (line 236)
            append_532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), r_531, 'append')
            # Calling append(args, kwargs) (line 236)
            append_call_result_535 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), append_532, *[s_533], **kwargs_534)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'r' (line 237)
        r_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type', r_536)
        
        # ################# End of 'translate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'translate' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'translate'
        return stypy_return_type_537


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 214, 0, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Omino.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Omino' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'Omino', Omino)
# Declaration of the 'A' class
# Getting the type of 'Omino' (line 240)
Omino_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'Omino')

class A(Omino_538, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'A.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 242):
        str_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 20), 'str', 'A')
        # Getting the type of 'self' (line 242)
        self_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self')
        # Setting the type of the member 'name' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_540, 'name', str_539)
        
        # Assigning a List to a Attribute (line 243):
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        int_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 22), list_542, int_543)
        # Adding element type (line 243)
        int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 22), list_542, int_544)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 21), list_541, list_542)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        int_546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 30), list_545, int_546)
        # Adding element type (line 243)
        int_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 30), list_545, int_547)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 21), list_541, list_545)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        int_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 38), list_548, int_549)
        # Adding element type (line 243)
        int_550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 38), list_548, int_550)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 21), list_541, list_548)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        int_552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 46), list_551, int_552)
        # Adding element type (line 243)
        int_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 46), list_551, int_553)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 21), list_541, list_551)
        
        # Getting the type of 'self' (line 243)
        self_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), self_554, 'cells', list_541)
        
        # Assigning a Name to a Attribute (line 244):
        # Getting the type of 'symmetries' (line 244)
        symmetries_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 22), 'symmetries')
        # Getting the type of 'self' (line 244)
        self_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_556, 'cosets', symmetries_555)
        
        # Call to getorientations(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_559 = {}
        # Getting the type of 'self' (line 245)
        self_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 245)
        getorientations_558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_557, 'getorientations')
        # Calling getorientations(args, kwargs) (line 245)
        getorientations_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), getorientations_558, *[], **kwargs_559)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'A' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'A', A)
# Declaration of the 'B' class
# Getting the type of 'Omino' (line 248)
Omino_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'Omino')

class B(Omino_561, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 249, 4, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'B.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 250):
        str_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'str', 'B')
        # Getting the type of 'self' (line 250)
        self_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'name' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_563, 'name', str_562)
        
        # Assigning a List to a Attribute (line 251):
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        int_566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 22), list_565, int_566)
        # Adding element type (line 251)
        int_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 22), list_565, int_567)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 21), list_564, list_565)
        # Adding element type (line 251)
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        int_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 30), list_568, int_569)
        # Adding element type (line 251)
        int_570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 30), list_568, int_570)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 21), list_564, list_568)
        # Adding element type (line 251)
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        int_572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 38), list_571, int_572)
        # Adding element type (line 251)
        int_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 38), list_571, int_573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 21), list_564, list_571)
        # Adding element type (line 251)
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        int_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 46), list_574, int_575)
        # Adding element type (line 251)
        int_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 46), list_574, int_576)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 21), list_564, list_574)
        # Adding element type (line 251)
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        int_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 54), list_577, int_578)
        # Adding element type (line 251)
        int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 54), list_577, int_579)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 21), list_564, list_577)
        
        # Getting the type of 'self' (line 251)
        self_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_580, 'cells', list_564)
        
        # Assigning a Name to a Attribute (line 252):
        # Getting the type of 'symmetries' (line 252)
        symmetries_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 22), 'symmetries')
        # Getting the type of 'self' (line 252)
        self_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_582, 'cosets', symmetries_581)
        
        # Call to getorientations(...): (line 253)
        # Processing the call keyword arguments (line 253)
        kwargs_585 = {}
        # Getting the type of 'self' (line 253)
        self_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 253)
        getorientations_584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_583, 'getorientations')
        # Calling getorientations(args, kwargs) (line 253)
        getorientations_call_result_586 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), getorientations_584, *[], **kwargs_585)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'B' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'B', B)
# Declaration of the 'C' class
# Getting the type of 'Omino' (line 256)
Omino_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'Omino')

class C(Omino_587, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 258):
        str_588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 20), 'str', 'C')
        # Getting the type of 'self' (line 258)
        self_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self')
        # Setting the type of the member 'name' of a type (line 258)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_589, 'name', str_588)
        
        # Assigning a List to a Attribute (line 259):
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        int_592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 22), list_591, int_592)
        # Adding element type (line 259)
        int_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 22), list_591, int_593)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), list_590, list_591)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        int_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 30), list_594, int_595)
        # Adding element type (line 259)
        int_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 30), list_594, int_596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), list_590, list_594)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        int_598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 38), list_597, int_598)
        # Adding element type (line 259)
        int_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 38), list_597, int_599)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), list_590, list_597)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        int_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 46), list_600, int_601)
        # Adding element type (line 259)
        int_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 46), list_600, int_602)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), list_590, list_600)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        int_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 54), list_603, int_604)
        # Adding element type (line 259)
        int_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 54), list_603, int_605)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), list_590, list_603)
        
        # Getting the type of 'self' (line 259)
        self_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 259)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_606, 'cells', list_590)
        
        # Assigning a Name to a Attribute (line 260):
        # Getting the type of 'symmetries' (line 260)
        symmetries_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'symmetries')
        # Getting the type of 'self' (line 260)
        self_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 260)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), self_608, 'cosets', symmetries_607)
        
        # Call to getorientations(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_611 = {}
        # Getting the type of 'self' (line 261)
        self_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 261)
        getorientations_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_609, 'getorientations')
        # Calling getorientations(args, kwargs) (line 261)
        getorientations_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), getorientations_610, *[], **kwargs_611)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'C' (line 256)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'C', C)
# Declaration of the 'D' class
# Getting the type of 'Omino' (line 264)
Omino_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'Omino')

class D(Omino_613, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'D.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 266):
        str_614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 20), 'str', 'D')
        # Getting the type of 'self' (line 266)
        self_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self')
        # Setting the type of the member 'name' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_615, 'name', str_614)
        
        # Assigning a List to a Attribute (line 267):
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 22), list_617, int_618)
        # Adding element type (line 267)
        int_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 22), list_617, int_619)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_616, list_617)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 31), list_620, int_621)
        # Adding element type (line 267)
        int_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 31), list_620, int_622)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_616, list_620)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 40), list_623, int_624)
        # Adding element type (line 267)
        int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 40), list_623, int_625)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_616, list_623)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 48), list_626, int_627)
        # Adding element type (line 267)
        int_628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 48), list_626, int_628)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_616, list_626)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 56), list_629, int_630)
        # Adding element type (line 267)
        int_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 56), list_629, int_631)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_616, list_629)
        
        # Getting the type of 'self' (line 267)
        self_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 267)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_632, 'cells', list_616)
        
        # Assigning a Name to a Attribute (line 268):
        # Getting the type of 'symmetries' (line 268)
        symmetries_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'symmetries')
        # Getting the type of 'self' (line 268)
        self_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 268)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_634, 'cosets', symmetries_633)
        
        # Call to getorientations(...): (line 269)
        # Processing the call keyword arguments (line 269)
        kwargs_637 = {}
        # Getting the type of 'self' (line 269)
        self_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 269)
        getorientations_636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_635, 'getorientations')
        # Calling getorientations(args, kwargs) (line 269)
        getorientations_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), getorientations_636, *[], **kwargs_637)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'D' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'D', D)
# Declaration of the 'E' class
# Getting the type of 'Omino' (line 272)
Omino_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'Omino')

class E(Omino_639, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'E.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 274):
        str_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 20), 'str', 'E')
        # Getting the type of 'self' (line 274)
        self_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self')
        # Setting the type of the member 'name' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_641, 'name', str_640)
        
        # Assigning a List to a Attribute (line 275):
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 22), list_643, int_644)
        # Adding element type (line 275)
        int_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 22), list_643, int_645)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 21), list_642, list_643)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_646, int_647)
        # Adding element type (line 275)
        int_648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_646, int_648)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 21), list_642, list_646)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 38), list_649, int_650)
        # Adding element type (line 275)
        int_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 38), list_649, int_651)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 21), list_642, list_649)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 46), list_652, int_653)
        # Adding element type (line 275)
        int_654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 46), list_652, int_654)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 21), list_642, list_652)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 54), list_655, int_656)
        # Adding element type (line 275)
        int_657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 54), list_655, int_657)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 21), list_642, list_655)
        
        # Getting the type of 'self' (line 275)
        self_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_658, 'cells', list_642)
        
        # Assigning a Name to a Attribute (line 276):
        # Getting the type of 'symmetries' (line 276)
        symmetries_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'symmetries')
        # Getting the type of 'self' (line 276)
        self_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 276)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_660, 'cosets', symmetries_659)
        
        # Call to getorientations(...): (line 277)
        # Processing the call keyword arguments (line 277)
        kwargs_663 = {}
        # Getting the type of 'self' (line 277)
        self_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 277)
        getorientations_662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), self_661, 'getorientations')
        # Calling getorientations(args, kwargs) (line 277)
        getorientations_call_result_664 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), getorientations_662, *[], **kwargs_663)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'E' (line 272)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 0), 'E', E)
# Declaration of the 'F' class
# Getting the type of 'Omino' (line 280)
Omino_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'Omino')

class F(Omino_665, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'F.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 282):
        str_666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 20), 'str', 'F')
        # Getting the type of 'self' (line 282)
        self_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'self')
        # Setting the type of the member 'name' of a type (line 282)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), self_667, 'name', str_666)
        
        # Assigning a List to a Attribute (line 283):
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        int_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 22), list_669, int_670)
        # Adding element type (line 283)
        int_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 22), list_669, int_671)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 21), list_668, list_669)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        int_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 30), list_672, int_673)
        # Adding element type (line 283)
        int_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 30), list_672, int_674)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 21), list_668, list_672)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        int_676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 38), list_675, int_676)
        # Adding element type (line 283)
        int_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 38), list_675, int_677)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 21), list_668, list_675)
        
        # Getting the type of 'self' (line 283)
        self_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 283)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_678, 'cells', list_668)
        
        # Assigning a Name to a Attribute (line 284):
        # Getting the type of 'rotations' (line 284)
        rotations_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 22), 'rotations')
        # Getting the type of 'self' (line 284)
        self_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_680, 'cosets', rotations_679)
        
        # Call to getorientations(...): (line 285)
        # Processing the call keyword arguments (line 285)
        kwargs_683 = {}
        # Getting the type of 'self' (line 285)
        self_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 285)
        getorientations_682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_681, 'getorientations')
        # Calling getorientations(args, kwargs) (line 285)
        getorientations_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), getorientations_682, *[], **kwargs_683)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'F' (line 280)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'F', F)
# Declaration of the 'G' class
# Getting the type of 'Omino' (line 288)
Omino_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'Omino')

class G(Omino_685, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'G.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 290):
        str_686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 20), 'str', 'G')
        # Getting the type of 'self' (line 290)
        self_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'self')
        # Setting the type of the member 'name' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), self_687, 'name', str_686)
        
        # Assigning a List to a Attribute (line 291):
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 22), list_689, int_690)
        # Adding element type (line 291)
        int_691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 22), list_689, int_691)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 21), list_688, list_689)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 30), list_692, int_693)
        # Adding element type (line 291)
        int_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 30), list_692, int_694)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 21), list_688, list_692)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 38), list_695, int_696)
        # Adding element type (line 291)
        int_697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 38), list_695, int_697)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 21), list_688, list_695)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 46), list_698, int_699)
        # Adding element type (line 291)
        int_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 46), list_698, int_700)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 21), list_688, list_698)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 54), list_701, int_702)
        # Adding element type (line 291)
        int_703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 54), list_701, int_703)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 21), list_688, list_701)
        
        # Getting the type of 'self' (line 291)
        self_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 291)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_704, 'cells', list_688)
        
        # Assigning a Name to a Attribute (line 292):
        # Getting the type of 'rotations' (line 292)
        rotations_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 22), 'rotations')
        # Getting the type of 'self' (line 292)
        self_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 292)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_706, 'cosets', rotations_705)
        
        # Call to getorientations(...): (line 293)
        # Processing the call keyword arguments (line 293)
        kwargs_709 = {}
        # Getting the type of 'self' (line 293)
        self_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 293)
        getorientations_708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), self_707, 'getorientations')
        # Calling getorientations(args, kwargs) (line 293)
        getorientations_call_result_710 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), getorientations_708, *[], **kwargs_709)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'G' (line 288)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'G', G)
# Declaration of the 'H' class
# Getting the type of 'Omino' (line 296)
Omino_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'Omino')

class H(Omino_711, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 297, 4, False)
        # Assigning a type to the variable 'self' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'H.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 298):
        str_712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 20), 'str', 'H')
        # Getting the type of 'self' (line 298)
        self_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'self')
        # Setting the type of the member 'name' of a type (line 298)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), self_713, 'name', str_712)
        
        # Assigning a List to a Attribute (line 299):
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        int_716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 22), list_715, int_716)
        # Adding element type (line 299)
        int_717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 22), list_715, int_717)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 21), list_714, list_715)
        # Adding element type (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        int_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 30), list_718, int_719)
        # Adding element type (line 299)
        int_720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 30), list_718, int_720)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 21), list_714, list_718)
        # Adding element type (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        int_722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 38), list_721, int_722)
        # Adding element type (line 299)
        int_723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 38), list_721, int_723)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 21), list_714, list_721)
        # Adding element type (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 46), list_724, int_725)
        # Adding element type (line 299)
        int_726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 46), list_724, int_726)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 21), list_714, list_724)
        # Adding element type (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        int_728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 54), list_727, int_728)
        # Adding element type (line 299)
        int_729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 54), list_727, int_729)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 21), list_714, list_727)
        
        # Getting the type of 'self' (line 299)
        self_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 299)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), self_730, 'cells', list_714)
        
        # Assigning a Name to a Attribute (line 300):
        # Getting the type of 'rotations' (line 300)
        rotations_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 22), 'rotations')
        # Getting the type of 'self' (line 300)
        self_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 300)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), self_732, 'cosets', rotations_731)
        
        # Call to getorientations(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_735 = {}
        # Getting the type of 'self' (line 301)
        self_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 301)
        getorientations_734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_733, 'getorientations')
        # Calling getorientations(args, kwargs) (line 301)
        getorientations_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), getorientations_734, *[], **kwargs_735)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'H' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'H', H)
# Declaration of the 'I' class
# Getting the type of 'Omino' (line 304)
Omino_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'Omino')

class I(Omino_737, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'I.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 306):
        str_738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 20), 'str', 'I')
        # Getting the type of 'self' (line 306)
        self_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self')
        # Setting the type of the member 'name' of a type (line 306)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_739, 'name', str_738)
        
        # Assigning a List to a Attribute (line 307):
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 22), list_741, int_742)
        # Adding element type (line 307)
        int_743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 22), list_741, int_743)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 21), list_740, list_741)
        # Adding element type (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 30), list_744, int_745)
        # Adding element type (line 307)
        int_746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 30), list_744, int_746)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 21), list_740, list_744)
        # Adding element type (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 38), list_747, int_748)
        # Adding element type (line 307)
        int_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 38), list_747, int_749)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 21), list_740, list_747)
        # Adding element type (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 46), list_750, int_751)
        # Adding element type (line 307)
        int_752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 46), list_750, int_752)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 21), list_740, list_750)
        # Adding element type (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 54), list_753, int_754)
        # Adding element type (line 307)
        int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 54), list_753, int_755)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 21), list_740, list_753)
        
        # Getting the type of 'self' (line 307)
        self_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_756, 'cells', list_740)
        
        # Assigning a Name to a Attribute (line 308):
        # Getting the type of 'rotations' (line 308)
        rotations_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 22), 'rotations')
        # Getting the type of 'self' (line 308)
        self_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 308)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_758, 'cosets', rotations_757)
        
        # Call to getorientations(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_761 = {}
        # Getting the type of 'self' (line 309)
        self_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 309)
        getorientations_760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_759, 'getorientations')
        # Calling getorientations(args, kwargs) (line 309)
        getorientations_call_result_762 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), getorientations_760, *[], **kwargs_761)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'I' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'I', I)
# Declaration of the 'J' class
# Getting the type of 'Omino' (line 312)
Omino_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'Omino')

class J(Omino_763, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 313, 4, False)
        # Assigning a type to the variable 'self' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'J.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 314):
        str_764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 20), 'str', 'J')
        # Getting the type of 'self' (line 314)
        self_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self')
        # Setting the type of the member 'name' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_765, 'name', str_764)
        
        # Assigning a List to a Attribute (line 315):
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        int_768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 22), list_767, int_768)
        # Adding element type (line 315)
        int_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 22), list_767, int_769)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), list_766, list_767)
        # Adding element type (line 315)
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        int_771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 30), list_770, int_771)
        # Adding element type (line 315)
        int_772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 30), list_770, int_772)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), list_766, list_770)
        # Adding element type (line 315)
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        int_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 38), list_773, int_774)
        # Adding element type (line 315)
        int_775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 38), list_773, int_775)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), list_766, list_773)
        # Adding element type (line 315)
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        int_777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 46), list_776, int_777)
        # Adding element type (line 315)
        int_778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 46), list_776, int_778)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), list_766, list_776)
        
        # Getting the type of 'self' (line 315)
        self_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_779, 'cells', list_766)
        
        # Assigning a List to a Attribute (line 316):
        
        # Obtaining an instance of the builtin type 'list' (line 316)
        list_780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 316)
        # Adding element type (line 316)
        # Getting the type of 'identity' (line 316)
        identity_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 22), list_780, identity_781)
        # Adding element type (line 316)
        # Getting the type of 'r90' (line 316)
        r90_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 33), 'r90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 22), list_780, r90_782)
        
        # Getting the type of 'self' (line 316)
        self_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 316)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_783, 'cosets', list_780)
        
        # Call to getorientations(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_786 = {}
        # Getting the type of 'self' (line 317)
        self_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 317)
        getorientations_785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_784, 'getorientations')
        # Calling getorientations(args, kwargs) (line 317)
        getorientations_call_result_787 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), getorientations_785, *[], **kwargs_786)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'J' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'J', J)
# Declaration of the 'K' class
# Getting the type of 'Omino' (line 320)
Omino_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'Omino')

class K(Omino_788, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'K.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 322):
        str_789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 20), 'str', 'K')
        # Getting the type of 'self' (line 322)
        self_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self')
        # Setting the type of the member 'name' of a type (line 322)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_790, 'name', str_789)
        
        # Assigning a List to a Attribute (line 323):
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 22), list_792, int_793)
        # Adding element type (line 323)
        int_794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 22), list_792, int_794)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 21), list_791, list_792)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 30), list_795, int_796)
        # Adding element type (line 323)
        int_797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 30), list_795, int_797)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 21), list_791, list_795)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 38), list_798, int_799)
        # Adding element type (line 323)
        int_800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 38), list_798, int_800)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 21), list_791, list_798)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 46), list_801, int_802)
        # Adding element type (line 323)
        int_803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 46), list_801, int_803)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 21), list_791, list_801)
        
        # Getting the type of 'self' (line 323)
        self_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 323)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_804, 'cells', list_791)
        
        # Assigning a List to a Attribute (line 324):
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        # Getting the type of 'identity' (line 324)
        identity_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 22), list_805, identity_806)
        
        # Getting the type of 'self' (line 324)
        self_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 324)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), self_807, 'cosets', list_805)
        
        # Call to getorientations(...): (line 325)
        # Processing the call keyword arguments (line 325)
        kwargs_810 = {}
        # Getting the type of 'self' (line 325)
        self_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 325)
        getorientations_809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), self_808, 'getorientations')
        # Calling getorientations(args, kwargs) (line 325)
        getorientations_call_result_811 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), getorientations_809, *[], **kwargs_810)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'K' (line 320)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'K', K)
# Declaration of the 'L' class
# Getting the type of 'Omino' (line 328)
Omino_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'Omino')

class L(Omino_812, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'L.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 330):
        str_813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 20), 'str', 'L')
        # Getting the type of 'self' (line 330)
        self_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'self')
        # Setting the type of the member 'name' of a type (line 330)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), self_814, 'name', str_813)
        
        # Assigning a List to a Attribute (line 331):
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 22), list_816, int_817)
        # Adding element type (line 331)
        int_818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 22), list_816, int_818)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 21), list_815, list_816)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        int_820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 30), list_819, int_820)
        # Adding element type (line 331)
        int_821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 30), list_819, int_821)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 21), list_815, list_819)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        int_823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), list_822, int_823)
        # Adding element type (line 331)
        int_824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), list_822, int_824)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 21), list_815, list_822)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        int_826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 47), list_825, int_826)
        # Adding element type (line 331)
        int_827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 47), list_825, int_827)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 21), list_815, list_825)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        int_829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 56), list_828, int_829)
        # Adding element type (line 331)
        int_830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 56), list_828, int_830)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 21), list_815, list_828)
        
        # Getting the type of 'self' (line 331)
        self_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self')
        # Setting the type of the member 'cells' of a type (line 331)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_831, 'cells', list_815)
        
        # Assigning a List to a Attribute (line 332):
        
        # Obtaining an instance of the builtin type 'list' (line 332)
        list_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 332)
        # Adding element type (line 332)
        # Getting the type of 'identity' (line 332)
        identity_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 23), 'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 22), list_832, identity_833)
        
        # Getting the type of 'self' (line 332)
        self_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self')
        # Setting the type of the member 'cosets' of a type (line 332)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_834, 'cosets', list_832)
        
        # Call to getorientations(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_837 = {}
        # Getting the type of 'self' (line 333)
        self_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self', False)
        # Obtaining the member 'getorientations' of a type (line 333)
        getorientations_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_835, 'getorientations')
        # Calling getorientations(args, kwargs) (line 333)
        getorientations_call_result_838 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), getorientations_836, *[], **kwargs_837)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'L' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'L', L)

@norecursion
def set5x11(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set5x11'
    module_type_store = module_type_store.open_function_context('set5x11', 336, 0, False)
    
    # Passed parameters checking function
    set5x11.stypy_localization = localization
    set5x11.stypy_type_of_self = None
    set5x11.stypy_type_store = module_type_store
    set5x11.stypy_function_name = 'set5x11'
    set5x11.stypy_param_names_list = []
    set5x11.stypy_varargs_param_name = None
    set5x11.stypy_kwargs_param_name = None
    set5x11.stypy_call_defaults = defaults
    set5x11.stypy_call_varargs = varargs
    set5x11.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set5x11', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set5x11', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set5x11(...)' code ##################

    # Marking variables as global (line 337)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 337, 4), 'c1')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 337, 4), 'ominos')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 337, 4), 'rows')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 337, 4), 'columns')
    
    # Assigning a List to a Name (line 338):
    
    # Obtaining an instance of the builtin type 'list' (line 338)
    list_839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 338)
    # Adding element type (line 338)
    str_840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 10), 'str', 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_840)
    # Adding element type (line 338)
    str_841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 15), 'str', 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_841)
    # Adding element type (line 338)
    str_842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 20), 'str', 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_842)
    # Adding element type (line 338)
    str_843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 25), 'str', 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_843)
    # Adding element type (line 338)
    str_844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 30), 'str', 'E')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_844)
    # Adding element type (line 338)
    str_845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 35), 'str', 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_845)
    # Adding element type (line 338)
    str_846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 40), 'str', 'G')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_846)
    # Adding element type (line 338)
    str_847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 45), 'str', 'H')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_847)
    # Adding element type (line 338)
    str_848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 50), 'str', 'I')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_848)
    # Adding element type (line 338)
    str_849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 55), 'str', 'J')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_849)
    # Adding element type (line 338)
    str_850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 60), 'str', 'K')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_850)
    # Adding element type (line 338)
    str_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 65), 'str', 'L')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 9), list_839, str_851)
    
    # Assigning a type to the variable 'c1' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'c1', list_839)
    
    # Assigning a List to a Name (line 339):
    
    # Obtaining an instance of the builtin type 'list' (line 339)
    list_852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 339)
    # Adding element type (line 339)
    
    # Call to A(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_854 = {}
    # Getting the type of 'A' (line 339)
    A_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 14), 'A', False)
    # Calling A(args, kwargs) (line 339)
    A_call_result_855 = invoke(stypy.reporting.localization.Localization(__file__, 339, 14), A_853, *[], **kwargs_854)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, A_call_result_855)
    # Adding element type (line 339)
    
    # Call to B(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_857 = {}
    # Getting the type of 'B' (line 339)
    B_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'B', False)
    # Calling B(args, kwargs) (line 339)
    B_call_result_858 = invoke(stypy.reporting.localization.Localization(__file__, 339, 19), B_856, *[], **kwargs_857)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, B_call_result_858)
    # Adding element type (line 339)
    
    # Call to C(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_860 = {}
    # Getting the type of 'C' (line 339)
    C_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'C', False)
    # Calling C(args, kwargs) (line 339)
    C_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 339, 24), C_859, *[], **kwargs_860)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, C_call_result_861)
    # Adding element type (line 339)
    
    # Call to D(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_863 = {}
    # Getting the type of 'D' (line 339)
    D_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 29), 'D', False)
    # Calling D(args, kwargs) (line 339)
    D_call_result_864 = invoke(stypy.reporting.localization.Localization(__file__, 339, 29), D_862, *[], **kwargs_863)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, D_call_result_864)
    # Adding element type (line 339)
    
    # Call to E(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_866 = {}
    # Getting the type of 'E' (line 339)
    E_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 34), 'E', False)
    # Calling E(args, kwargs) (line 339)
    E_call_result_867 = invoke(stypy.reporting.localization.Localization(__file__, 339, 34), E_865, *[], **kwargs_866)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, E_call_result_867)
    # Adding element type (line 339)
    
    # Call to F(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_869 = {}
    # Getting the type of 'F' (line 339)
    F_868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 39), 'F', False)
    # Calling F(args, kwargs) (line 339)
    F_call_result_870 = invoke(stypy.reporting.localization.Localization(__file__, 339, 39), F_868, *[], **kwargs_869)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, F_call_result_870)
    # Adding element type (line 339)
    
    # Call to G(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_872 = {}
    # Getting the type of 'G' (line 339)
    G_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 44), 'G', False)
    # Calling G(args, kwargs) (line 339)
    G_call_result_873 = invoke(stypy.reporting.localization.Localization(__file__, 339, 44), G_871, *[], **kwargs_872)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, G_call_result_873)
    # Adding element type (line 339)
    
    # Call to H(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_875 = {}
    # Getting the type of 'H' (line 339)
    H_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 49), 'H', False)
    # Calling H(args, kwargs) (line 339)
    H_call_result_876 = invoke(stypy.reporting.localization.Localization(__file__, 339, 49), H_874, *[], **kwargs_875)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, H_call_result_876)
    # Adding element type (line 339)
    
    # Call to I(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_878 = {}
    # Getting the type of 'I' (line 339)
    I_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), 'I', False)
    # Calling I(args, kwargs) (line 339)
    I_call_result_879 = invoke(stypy.reporting.localization.Localization(__file__, 339, 54), I_877, *[], **kwargs_878)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, I_call_result_879)
    # Adding element type (line 339)
    
    # Call to J(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_881 = {}
    # Getting the type of 'J' (line 339)
    J_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 59), 'J', False)
    # Calling J(args, kwargs) (line 339)
    J_call_result_882 = invoke(stypy.reporting.localization.Localization(__file__, 339, 59), J_880, *[], **kwargs_881)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, J_call_result_882)
    # Adding element type (line 339)
    
    # Call to K(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_884 = {}
    # Getting the type of 'K' (line 339)
    K_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 64), 'K', False)
    # Calling K(args, kwargs) (line 339)
    K_call_result_885 = invoke(stypy.reporting.localization.Localization(__file__, 339, 64), K_883, *[], **kwargs_884)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, K_call_result_885)
    # Adding element type (line 339)
    
    # Call to L(...): (line 339)
    # Processing the call keyword arguments (line 339)
    kwargs_887 = {}
    # Getting the type of 'L' (line 339)
    L_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 69), 'L', False)
    # Calling L(args, kwargs) (line 339)
    L_call_result_888 = invoke(stypy.reporting.localization.Localization(__file__, 339, 69), L_886, *[], **kwargs_887)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 13), list_852, L_call_result_888)
    
    # Assigning a type to the variable 'ominos' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'ominos', list_852)
    
    # Assigning a Num to a Name (line 340):
    int_889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 11), 'int')
    # Assigning a type to the variable 'rows' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'rows', int_889)
    
    # Assigning a Num to a Name (line 341):
    int_890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 14), 'int')
    # Assigning a type to the variable 'columns' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'columns', int_890)
    
    # ################# End of 'set5x11(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set5x11' in the type store
    # Getting the type of 'stypy_return_type' (line 336)
    stypy_return_type_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_891)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set5x11'
    return stypy_return_type_891

# Assigning a type to the variable 'set5x11' (line 336)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 0), 'set5x11', set5x11)

# Call to set5x11(...): (line 345)
# Processing the call keyword arguments (line 345)
kwargs_893 = {}
# Getting the type of 'set5x11' (line 345)
set5x11_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'set5x11', False)
# Calling set5x11(args, kwargs) (line 345)
set5x11_call_result_894 = invoke(stypy.reporting.localization.Localization(__file__, 345, 0), set5x11_892, *[], **kwargs_893)


# Assigning a Call to a Name (line 349):

# Call to Column(...): (line 349)
# Processing the call arguments (line 349)
str_896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 14), 'str', 'root')
# Processing the call keyword arguments (line 349)
kwargs_897 = {}
# Getting the type of 'Column' (line 349)
Column_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 7), 'Column', False)
# Calling Column(args, kwargs) (line 349)
Column_call_result_898 = invoke(stypy.reporting.localization.Localization(__file__, 349, 7), Column_895, *[str_896], **kwargs_897)

# Assigning a type to the variable 'root' (line 349)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 0), 'root', Column_call_result_898)

# Assigning a Name to a Attribute (line 350):
# Getting the type of 'root' (line 350)
root_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'root')
# Getting the type of 'root' (line 350)
root_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'root')
# Setting the type of the member 'left' of a type (line 350)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 0), root_900, 'left', root_899)

# Assigning a Name to a Attribute (line 351):
# Getting the type of 'root' (line 351)
root_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'root')
# Getting the type of 'root' (line 351)
root_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 0), 'root')
# Setting the type of the member 'right' of a type (line 351)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 0), root_902, 'right', root_901)

# Assigning a Name to a Name (line 353):
# Getting the type of 'root' (line 353)
root_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 7), 'root')
# Assigning a type to the variable 'last' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'last', root_903)

# Assigning a Dict to a Name (line 357):

# Obtaining an instance of the builtin type 'dict' (line 357)
dict_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 357)

# Assigning a type to the variable 'pcolumns' (line 357)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'pcolumns', dict_904)

# Getting the type of 'c1' (line 358)
c1_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'c1')
# Testing if the for loop is going to be iterated (line 358)
# Testing the type of a for loop iterable (line 358)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 358, 0), c1_905)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 358, 0), c1_905):
    # Getting the type of the for loop variable (line 358)
    for_loop_var_906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 358, 0), c1_905)
    # Assigning a type to the variable 'col2' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'col2', for_loop_var_906)
    # SSA begins for a for statement (line 358)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 359):
    
    # Call to Column(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'col2' (line 359)
    col2_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'col2', False)
    # Processing the call keyword arguments (line 359)
    kwargs_909 = {}
    # Getting the type of 'Column' (line 359)
    Column_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'Column', False)
    # Calling Column(args, kwargs) (line 359)
    Column_call_result_910 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), Column_907, *[col2_908], **kwargs_909)
    
    # Assigning a type to the variable 'c' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'c', Column_call_result_910)
    
    # Assigning a Name to a Attribute (line 360):
    # Getting the type of 'c' (line 360)
    c_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'c')
    # Getting the type of 'last' (line 360)
    last_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'last')
    # Setting the type of the member 'right' of a type (line 360)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 4), last_912, 'right', c_911)
    
    # Assigning a Name to a Attribute (line 361):
    # Getting the type of 'last' (line 361)
    last_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 13), 'last')
    # Getting the type of 'c' (line 361)
    c_914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'c')
    # Setting the type of the member 'left' of a type (line 361)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 4), c_914, 'left', last_913)
    
    # Assigning a Name to a Attribute (line 362):
    # Getting the type of 'root' (line 362)
    root_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 14), 'root')
    # Getting the type of 'c' (line 362)
    c_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'c')
    # Setting the type of the member 'right' of a type (line 362)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 4), c_916, 'right', root_915)
    
    # Assigning a Name to a Attribute (line 363):
    # Getting the type of 'c' (line 363)
    c_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'c')
    # Getting the type of 'root' (line 363)
    root_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'root')
    # Setting the type of the member 'left' of a type (line 363)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 4), root_918, 'left', c_917)
    
    # Assigning a Name to a Name (line 364):
    # Getting the type of 'c' (line 364)
    c_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 11), 'c')
    # Assigning a type to the variable 'last' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'last', c_919)
    
    # Assigning a Name to a Subscript (line 365):
    # Getting the type of 'c' (line 365)
    c_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'c')
    # Getting the type of 'pcolumns' (line 365)
    pcolumns_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'pcolumns')
    # Getting the type of 'col2' (line 365)
    col2_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 13), 'col2')
    # Storing an element on a container (line 365)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 4), pcolumns_921, (col2_922, c_920))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()



# Assigning a Name to a Name (line 367):
# Getting the type of 'root' (line 367)
root_923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 7), 'root')
# Assigning a type to the variable 'last' (line 367)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), 'last', root_923)


# Call to range(...): (line 368)
# Processing the call arguments (line 368)
# Getting the type of 'rows' (line 368)
rows_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), 'rows', False)
# Processing the call keyword arguments (line 368)
kwargs_926 = {}
# Getting the type of 'range' (line 368)
range_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 'range', False)
# Calling range(args, kwargs) (line 368)
range_call_result_927 = invoke(stypy.reporting.localization.Localization(__file__, 368, 11), range_924, *[rows_925], **kwargs_926)

# Testing if the for loop is going to be iterated (line 368)
# Testing the type of a for loop iterable (line 368)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 368, 0), range_call_result_927)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 368, 0), range_call_result_927):
    # Getting the type of the for loop variable (line 368)
    for_loop_var_928 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 368, 0), range_call_result_927)
    # Assigning a type to the variable 'row' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'row', for_loop_var_928)
    # SSA begins for a for statement (line 368)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'columns' (line 369)
    columns_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'columns', False)
    # Processing the call keyword arguments (line 369)
    kwargs_931 = {}
    # Getting the type of 'range' (line 369)
    range_929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'range', False)
    # Calling range(args, kwargs) (line 369)
    range_call_result_932 = invoke(stypy.reporting.localization.Localization(__file__, 369, 15), range_929, *[columns_930], **kwargs_931)
    
    # Testing if the for loop is going to be iterated (line 369)
    # Testing the type of a for loop iterable (line 369)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 369, 4), range_call_result_932)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 369, 4), range_call_result_932):
        # Getting the type of the for loop variable (line 369)
        for_loop_var_933 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 369, 4), range_call_result_932)
        # Assigning a type to the variable 'col' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'col', for_loop_var_933)
        # SSA begins for a for statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 370):
        
        # Call to Column(...): (line 370)
        # Processing the call arguments (line 370)
        str_935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 19), 'str', '[')
        
        # Call to str(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'col' (line 370)
        col_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 29), 'col', False)
        # Processing the call keyword arguments (line 370)
        kwargs_938 = {}
        # Getting the type of 'str' (line 370)
        str_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 25), 'str', False)
        # Calling str(args, kwargs) (line 370)
        str_call_result_939 = invoke(stypy.reporting.localization.Localization(__file__, 370, 25), str_936, *[col_937], **kwargs_938)
        
        # Applying the binary operator '+' (line 370)
        result_add_940 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 19), '+', str_935, str_call_result_939)
        
        str_941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 36), 'str', ',')
        # Applying the binary operator '+' (line 370)
        result_add_942 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 34), '+', result_add_940, str_941)
        
        
        # Call to str(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'row' (line 370)
        row_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 46), 'row', False)
        # Processing the call keyword arguments (line 370)
        kwargs_945 = {}
        # Getting the type of 'str' (line 370)
        str_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 42), 'str', False)
        # Calling str(args, kwargs) (line 370)
        str_call_result_946 = invoke(stypy.reporting.localization.Localization(__file__, 370, 42), str_943, *[row_944], **kwargs_945)
        
        # Applying the binary operator '+' (line 370)
        result_add_947 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 40), '+', result_add_942, str_call_result_946)
        
        str_948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 53), 'str', '] ')
        # Applying the binary operator '+' (line 370)
        result_add_949 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 51), '+', result_add_947, str_948)
        
        # Processing the call keyword arguments (line 370)
        kwargs_950 = {}
        # Getting the type of 'Column' (line 370)
        Column_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'Column', False)
        # Calling Column(args, kwargs) (line 370)
        Column_call_result_951 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), Column_934, *[result_add_949], **kwargs_950)
        
        # Assigning a type to the variable 'c' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'c', Column_call_result_951)
        
        # Assigning a List to a Attribute (line 371):
        
        # Obtaining an instance of the builtin type 'list' (line 371)
        list_952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 371)
        # Adding element type (line 371)
        # Getting the type of 'col' (line 371)
        col_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 18), list_952, col_953)
        # Adding element type (line 371)
        # Getting the type of 'row' (line 371)
        row_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 18), list_952, row_954)
        
        # Getting the type of 'c' (line 371)
        c_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'c')
        # Setting the type of the member 'extra' of a type (line 371)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), c_955, 'extra', list_952)
        
        # Assigning a Name to a Attribute (line 373):
        # Getting the type of 'c' (line 373)
        c_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'c')
        # Getting the type of 'last' (line 373)
        last_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'last')
        # Obtaining the member 'right' of a type (line 373)
        right_958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), last_957, 'right')
        # Setting the type of the member 'left' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), right_958, 'left', c_956)
        
        # Assigning a Attribute to a Attribute (line 374):
        # Getting the type of 'last' (line 374)
        last_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 18), 'last')
        # Obtaining the member 'right' of a type (line 374)
        right_960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 18), last_959, 'right')
        # Getting the type of 'c' (line 374)
        c_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'c')
        # Setting the type of the member 'right' of a type (line 374)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), c_961, 'right', right_960)
        
        # Assigning a Name to a Attribute (line 375):
        # Getting the type of 'c' (line 375)
        c_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 21), 'c')
        # Getting the type of 'last' (line 375)
        last_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'last')
        # Setting the type of the member 'right' of a type (line 375)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), last_963, 'right', c_962)
        
        # Assigning a Name to a Attribute (line 376):
        # Getting the type of 'last' (line 376)
        last_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 17), 'last')
        # Getting the type of 'c' (line 376)
        c_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'c')
        # Setting the type of the member 'left' of a type (line 376)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), c_965, 'left', last_964)
        
        # Assigning a Name to a Name (line 377):
        # Getting the type of 'c' (line 377)
        c_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'c')
        # Assigning a type to the variable 'last' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'last', c_966)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()



@norecursion
def validatecell(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'validatecell'
    module_type_store = module_type_store.open_function_context('validatecell', 382, 0, False)
    
    # Passed parameters checking function
    validatecell.stypy_localization = localization
    validatecell.stypy_type_of_self = None
    validatecell.stypy_type_store = module_type_store
    validatecell.stypy_function_name = 'validatecell'
    validatecell.stypy_param_names_list = ['c']
    validatecell.stypy_varargs_param_name = None
    validatecell.stypy_kwargs_param_name = None
    validatecell.stypy_call_defaults = defaults
    validatecell.stypy_call_varargs = varargs
    validatecell.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'validatecell', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'validatecell', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'validatecell(...)' code ##################

    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 9), 'int')
    # Getting the type of 'c' (line 383)
    c_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 7), 'c')
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 7), c_968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_970 = invoke(stypy.reporting.localization.Localization(__file__, 383, 7), getitem___969, int_967)
    
    int_971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 14), 'int')
    # Applying the binary operator '<' (line 383)
    result_lt_972 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 7), '<', subscript_call_result_970, int_971)
    
    
    
    # Obtaining the type of the subscript
    int_973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 21), 'int')
    # Getting the type of 'c' (line 383)
    c_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'c')
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), c_974, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_976 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), getitem___975, int_973)
    
    # Getting the type of 'columns' (line 383)
    columns_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 26), 'columns')
    # Applying the binary operator '>' (line 383)
    result_gt_978 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 19), '>', subscript_call_result_976, columns_977)
    
    # Applying the binary operator 'or' (line 383)
    result_or_keyword_979 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 7), 'or', result_lt_972, result_gt_978)
    
    # Testing if the type of an if condition is none (line 383)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 383, 4), result_or_keyword_979):
        pass
    else:
        
        # Testing the type of an if condition (line 383)
        if_condition_980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 4), result_or_keyword_979)
        # Assigning a type to the variable 'if_condition_980' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'if_condition_980', if_condition_980)
        # SSA begins for if statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 383)
        False_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 42), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 35), 'stypy_return_type', False_981)
        # SSA join for if statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 9), 'int')
    # Getting the type of 'c' (line 384)
    c_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 7), 'c')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 7), c_983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_985 = invoke(stypy.reporting.localization.Localization(__file__, 384, 7), getitem___984, int_982)
    
    int_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 14), 'int')
    # Applying the binary operator '<' (line 384)
    result_lt_987 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 7), '<', subscript_call_result_985, int_986)
    
    
    
    # Obtaining the type of the subscript
    int_988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 21), 'int')
    # Getting the type of 'c' (line 384)
    c_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'c')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 19), c_989, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_991 = invoke(stypy.reporting.localization.Localization(__file__, 384, 19), getitem___990, int_988)
    
    # Getting the type of 'rows' (line 384)
    rows_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 26), 'rows')
    # Applying the binary operator '>' (line 384)
    result_gt_993 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 19), '>', subscript_call_result_991, rows_992)
    
    # Applying the binary operator 'or' (line 384)
    result_or_keyword_994 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 7), 'or', result_lt_987, result_gt_993)
    
    # Testing if the type of an if condition is none (line 384)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 384, 4), result_or_keyword_994):
        pass
    else:
        
        # Testing the type of an if condition (line 384)
        if_condition_995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 4), result_or_keyword_994)
        # Assigning a type to the variable 'if_condition_995' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'if_condition_995', if_condition_995)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 384)
        False_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 39), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 32), 'stypy_return_type', False_996)
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'True' (line 385)
    True_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type', True_997)
    
    # ################# End of 'validatecell(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'validatecell' in the type store
    # Getting the type of 'stypy_return_type' (line 382)
    stypy_return_type_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_998)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'validatecell'
    return stypy_return_type_998

# Assigning a type to the variable 'validatecell' (line 382)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 0), 'validatecell', validatecell)

@norecursion
def validate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'validate'
    module_type_store = module_type_store.open_function_context('validate', 388, 0, False)
    
    # Passed parameters checking function
    validate.stypy_localization = localization
    validate.stypy_type_of_self = None
    validate.stypy_type_store = module_type_store
    validate.stypy_function_name = 'validate'
    validate.stypy_param_names_list = ['orientation']
    validate.stypy_varargs_param_name = None
    validate.stypy_kwargs_param_name = None
    validate.stypy_call_defaults = defaults
    validate.stypy_call_varargs = varargs
    validate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'validate', ['orientation'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'validate', localization, ['orientation'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'validate(...)' code ##################

    
    # Getting the type of 'orientation' (line 389)
    orientation_999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'orientation')
    # Testing if the for loop is going to be iterated (line 389)
    # Testing the type of a for loop iterable (line 389)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 389, 4), orientation_999)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 389, 4), orientation_999):
        # Getting the type of the for loop variable (line 389)
        for_loop_var_1000 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 389, 4), orientation_999)
        # Assigning a type to the variable 'cell' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'cell', for_loop_var_1000)
        # SSA begins for a for statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to validatecell(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'cell' (line 390)
        cell_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'cell', False)
        # Processing the call keyword arguments (line 390)
        kwargs_1003 = {}
        # Getting the type of 'validatecell' (line 390)
        validatecell_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 'validatecell', False)
        # Calling validatecell(args, kwargs) (line 390)
        validatecell_call_result_1004 = invoke(stypy.reporting.localization.Localization(__file__, 390, 11), validatecell_1001, *[cell_1002], **kwargs_1003)
        
        # Getting the type of 'False' (line 390)
        False_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 33), 'False')
        # Applying the binary operator '==' (line 390)
        result_eq_1006 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 11), '==', validatecell_call_result_1004, False_1005)
        
        # Testing if the type of an if condition is none (line 390)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 390, 8), result_eq_1006):
            pass
        else:
            
            # Testing the type of an if condition (line 390)
            if_condition_1007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 8), result_eq_1006)
            # Assigning a type to the variable 'if_condition_1007' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'if_condition_1007', if_condition_1007)
            # SSA begins for if statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 390)
            False_1008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 47), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 40), 'stypy_return_type', False_1008)
            # SSA join for if statement (line 390)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 391)
    True_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type', True_1009)
    
    # ################# End of 'validate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'validate' in the type store
    # Getting the type of 'stypy_return_type' (line 388)
    stypy_return_type_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1010)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'validate'
    return stypy_return_type_1010

# Assigning a type to the variable 'validate' (line 388)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), 'validate', validate)

# Assigning a Num to a Name (line 396):
int_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 10), 'int')
# Assigning a type to the variable 'rownums' (line 396)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 0), 'rownums', int_1011)

# Getting the type of 'ominos' (line 397)
ominos_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'ominos')
# Testing if the for loop is going to be iterated (line 397)
# Testing the type of a for loop iterable (line 397)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 397, 0), ominos_1012)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 397, 0), ominos_1012):
    # Getting the type of the for loop variable (line 397)
    for_loop_var_1013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 397, 0), ominos_1012)
    # Assigning a type to the variable 'tile' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), 'tile', for_loop_var_1013)
    # SSA begins for a for statement (line 397)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'columns' (line 398)
    columns_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'columns', False)
    # Processing the call keyword arguments (line 398)
    kwargs_1016 = {}
    # Getting the type of 'range' (line 398)
    range_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'range', False)
    # Calling range(args, kwargs) (line 398)
    range_call_result_1017 = invoke(stypy.reporting.localization.Localization(__file__, 398, 15), range_1014, *[columns_1015], **kwargs_1016)
    
    # Testing if the for loop is going to be iterated (line 398)
    # Testing the type of a for loop iterable (line 398)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 398, 4), range_call_result_1017)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 398, 4), range_call_result_1017):
        # Getting the type of the for loop variable (line 398)
        for_loop_var_1018 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 398, 4), range_call_result_1017)
        # Assigning a type to the variable 'col' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'col', for_loop_var_1018)
        # SSA begins for a for statement (line 398)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'tile' (line 399)
        tile_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'tile')
        # Obtaining the member 'name' of a type (line 399)
        name_1020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 11), tile_1019, 'name')
        str_1021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 24), 'str', 'L')
        # Applying the binary operator '==' (line 399)
        result_eq_1022 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 11), '==', name_1020, str_1021)
        
        
        # Getting the type of 'col' (line 399)
        col_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 32), 'col')
        # Getting the type of 'Lcol' (line 399)
        Lcol_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 39), 'Lcol')
        # Applying the binary operator '!=' (line 399)
        result_ne_1025 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 32), '!=', col_1023, Lcol_1024)
        
        # Applying the binary operator 'and' (line 399)
        result_and_keyword_1026 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 11), 'and', result_eq_1022, result_ne_1025)
        
        # Testing if the type of an if condition is none (line 399)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 399, 8), result_and_keyword_1026):
            pass
        else:
            
            # Testing the type of an if condition (line 399)
            if_condition_1027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 8), result_and_keyword_1026)
            # Assigning a type to the variable 'if_condition_1027' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'if_condition_1027', if_condition_1027)
            # SSA begins for if statement (line 399)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 399)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to range(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'rows' (line 400)
        rows_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 25), 'rows', False)
        # Processing the call keyword arguments (line 400)
        kwargs_1030 = {}
        # Getting the type of 'range' (line 400)
        range_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 19), 'range', False)
        # Calling range(args, kwargs) (line 400)
        range_call_result_1031 = invoke(stypy.reporting.localization.Localization(__file__, 400, 19), range_1028, *[rows_1029], **kwargs_1030)
        
        # Testing if the for loop is going to be iterated (line 400)
        # Testing the type of a for loop iterable (line 400)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 400, 8), range_call_result_1031)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 400, 8), range_call_result_1031):
            # Getting the type of the for loop variable (line 400)
            for_loop_var_1032 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 400, 8), range_call_result_1031)
            # Assigning a type to the variable 'row' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'row', for_loop_var_1032)
            # SSA begins for a for statement (line 400)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'tile' (line 401)
            tile_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'tile')
            # Obtaining the member 'name' of a type (line 401)
            name_1034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 15), tile_1033, 'name')
            str_1035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 28), 'str', 'L')
            # Applying the binary operator '==' (line 401)
            result_eq_1036 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 15), '==', name_1034, str_1035)
            
            
            # Getting the type of 'row' (line 401)
            row_1037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 36), 'row')
            # Getting the type of 'Lrow' (line 401)
            Lrow_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 43), 'Lrow')
            # Applying the binary operator '!=' (line 401)
            result_ne_1039 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 36), '!=', row_1037, Lrow_1038)
            
            # Applying the binary operator 'and' (line 401)
            result_and_keyword_1040 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 15), 'and', result_eq_1036, result_ne_1039)
            
            # Testing if the type of an if condition is none (line 401)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 401, 12), result_and_keyword_1040):
                pass
            else:
                
                # Testing the type of an if condition (line 401)
                if_condition_1041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 12), result_and_keyword_1040)
                # Assigning a type to the variable 'if_condition_1041' (line 401)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'if_condition_1041', if_condition_1041)
                # SSA begins for if statement (line 401)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 401)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 402):
            
            # Call to translate(...): (line 402)
            # Processing the call arguments (line 402)
            
            # Obtaining an instance of the builtin type 'list' (line 402)
            list_1044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 402)
            # Adding element type (line 402)
            # Getting the type of 'col' (line 402)
            col_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 43), 'col', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 42), list_1044, col_1045)
            # Adding element type (line 402)
            # Getting the type of 'row' (line 402)
            row_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 48), 'row', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 42), list_1044, row_1046)
            
            # Processing the call keyword arguments (line 402)
            kwargs_1047 = {}
            # Getting the type of 'tile' (line 402)
            tile_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 27), 'tile', False)
            # Obtaining the member 'translate' of a type (line 402)
            translate_1043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 27), tile_1042, 'translate')
            # Calling translate(args, kwargs) (line 402)
            translate_call_result_1048 = invoke(stypy.reporting.localization.Localization(__file__, 402, 27), translate_1043, *[list_1044], **kwargs_1047)
            
            # Assigning a type to the variable 'orientations' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'orientations', translate_call_result_1048)
            
            # Getting the type of 'orientations' (line 403)
            orientations_1049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 31), 'orientations')
            # Testing if the for loop is going to be iterated (line 403)
            # Testing the type of a for loop iterable (line 403)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 403, 12), orientations_1049)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 403, 12), orientations_1049):
                # Getting the type of the for loop variable (line 403)
                for_loop_var_1050 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 403, 12), orientations_1049)
                # Assigning a type to the variable 'orientation' (line 403)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'orientation', for_loop_var_1050)
                # SSA begins for a for statement (line 403)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to validate(...): (line 404)
                # Processing the call arguments (line 404)
                # Getting the type of 'orientation' (line 404)
                orientation_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 28), 'orientation', False)
                # Processing the call keyword arguments (line 404)
                kwargs_1053 = {}
                # Getting the type of 'validate' (line 404)
                validate_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'validate', False)
                # Calling validate(args, kwargs) (line 404)
                validate_call_result_1054 = invoke(stypy.reporting.localization.Localization(__file__, 404, 19), validate_1051, *[orientation_1052], **kwargs_1053)
                
                # Getting the type of 'False' (line 404)
                False_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 44), 'False')
                # Applying the binary operator '==' (line 404)
                result_eq_1056 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 19), '==', validate_call_result_1054, False_1055)
                
                # Testing if the type of an if condition is none (line 404)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 404, 16), result_eq_1056):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 404)
                    if_condition_1057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 16), result_eq_1056)
                    # Assigning a type to the variable 'if_condition_1057' (line 404)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 16), 'if_condition_1057', if_condition_1057)
                    # SSA begins for if statement (line 404)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 404)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'rownums' (line 405)
                rownums_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'rownums')
                int_1059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 27), 'int')
                # Applying the binary operator '+=' (line 405)
                result_iadd_1060 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 16), '+=', rownums_1058, int_1059)
                # Assigning a type to the variable 'rownums' (line 405)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'rownums', result_iadd_1060)
                
                
                # Assigning a Call to a Name (line 406):
                
                # Call to Column(...): (line 406)
                # Processing the call keyword arguments (line 406)
                kwargs_1062 = {}
                # Getting the type of 'Column' (line 406)
                Column_1061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'Column', False)
                # Calling Column(args, kwargs) (line 406)
                Column_call_result_1063 = invoke(stypy.reporting.localization.Localization(__file__, 406, 26), Column_1061, *[], **kwargs_1062)
                
                # Assigning a type to the variable 'element' (line 406)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'element', Column_call_result_1063)
                
                # Assigning a Name to a Attribute (line 407):
                # Getting the type of 'element' (line 407)
                element_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 32), 'element')
                # Getting the type of 'element' (line 407)
                element_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'element')
                # Setting the type of the member 'right' of a type (line 407)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), element_1065, 'right', element_1064)
                
                # Assigning a Name to a Attribute (line 408):
                # Getting the type of 'element' (line 408)
                element_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 31), 'element')
                # Getting the type of 'element' (line 408)
                element_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'element')
                # Setting the type of the member 'left' of a type (line 408)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 16), element_1067, 'left', element_1066)
                
                # Assigning a Subscript to a Name (line 410):
                
                # Obtaining the type of the subscript
                # Getting the type of 'tile' (line 410)
                tile_1068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 34), 'tile')
                # Obtaining the member 'name' of a type (line 410)
                name_1069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 34), tile_1068, 'name')
                # Getting the type of 'pcolumns' (line 410)
                pcolumns_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'pcolumns')
                # Obtaining the member '__getitem__' of a type (line 410)
                getitem___1071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 25), pcolumns_1070, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 410)
                subscript_call_result_1072 = invoke(stypy.reporting.localization.Localization(__file__, 410, 25), getitem___1071, name_1069)
                
                # Assigning a type to the variable 'column' (line 410)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 16), 'column', subscript_call_result_1072)
                
                # Assigning a Name to a Attribute (line 411):
                # Getting the type of 'column' (line 411)
                column_1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 33), 'column')
                # Getting the type of 'element' (line 411)
                element_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'element')
                # Setting the type of the member 'column' of a type (line 411)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 16), element_1074, 'column', column_1073)
                
                # Assigning a Attribute to a Attribute (line 412):
                # Getting the type of 'column' (line 412)
                column_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 29), 'column')
                # Obtaining the member 'up' of a type (line 412)
                up_1076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 29), column_1075, 'up')
                # Getting the type of 'element' (line 412)
                element_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'element')
                # Setting the type of the member 'up' of a type (line 412)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 16), element_1077, 'up', up_1076)
                
                # Assigning a Name to a Attribute (line 413):
                # Getting the type of 'column' (line 413)
                column_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 31), 'column')
                # Getting the type of 'element' (line 413)
                element_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'element')
                # Setting the type of the member 'down' of a type (line 413)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), element_1079, 'down', column_1078)
                
                # Assigning a Name to a Attribute (line 414):
                # Getting the type of 'element' (line 414)
                element_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 33), 'element')
                # Getting the type of 'column' (line 414)
                column_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'column')
                # Obtaining the member 'up' of a type (line 414)
                up_1082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 16), column_1081, 'up')
                # Setting the type of the member 'down' of a type (line 414)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 16), up_1082, 'down', element_1080)
                
                # Assigning a Name to a Attribute (line 415):
                # Getting the type of 'element' (line 415)
                element_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 28), 'element')
                # Getting the type of 'column' (line 415)
                column_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'column')
                # Setting the type of the member 'up' of a type (line 415)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 16), column_1084, 'up', element_1083)
                
                # Getting the type of 'column' (line 416)
                column_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'column')
                # Obtaining the member 'size' of a type (line 416)
                size_1086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 16), column_1085, 'size')
                int_1087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 31), 'int')
                # Applying the binary operator '+=' (line 416)
                result_iadd_1088 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 16), '+=', size_1086, int_1087)
                # Getting the type of 'column' (line 416)
                column_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'column')
                # Setting the type of the member 'size' of a type (line 416)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 16), column_1089, 'size', result_iadd_1088)
                
                
                # Assigning a Name to a Name (line 417):
                # Getting the type of 'element' (line 417)
                element_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 29), 'element')
                # Assigning a type to the variable 'rowelement' (line 417)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'rowelement', element_1090)
                
                # Assigning a Attribute to a Name (line 419):
                # Getting the type of 'root' (line 419)
                root_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 25), 'root')
                # Obtaining the member 'right' of a type (line 419)
                right_1092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 25), root_1091, 'right')
                # Assigning a type to the variable 'column' (line 419)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'column', right_1092)
                
                
                # Getting the type of 'column' (line 420)
                column_1093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 22), 'column')
                # Obtaining the member 'extra' of a type (line 420)
                extra_1094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 22), column_1093, 'extra')
                # Getting the type of 'None' (line 420)
                None_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 38), 'None')
                # Applying the binary operator '!=' (line 420)
                result_ne_1096 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 22), '!=', extra_1094, None_1095)
                
                # Testing if the while is going to be iterated (line 420)
                # Testing the type of an if condition (line 420)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 16), result_ne_1096)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 420, 16), result_ne_1096):
                    # SSA begins for while statement (line 420)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                    
                    # Assigning a Attribute to a Name (line 421):
                    # Getting the type of 'column' (line 421)
                    column_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 28), 'column')
                    # Obtaining the member 'extra' of a type (line 421)
                    extra_1098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 28), column_1097, 'extra')
                    # Assigning a type to the variable 'entry' (line 421)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 20), 'entry', extra_1098)
                    
                    # Getting the type of 'orientation' (line 422)
                    orientation_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 32), 'orientation')
                    # Testing if the for loop is going to be iterated (line 422)
                    # Testing the type of a for loop iterable (line 422)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 422, 20), orientation_1099)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 422, 20), orientation_1099):
                        # Getting the type of the for loop variable (line 422)
                        for_loop_var_1100 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 422, 20), orientation_1099)
                        # Assigning a type to the variable 'cell' (line 422)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'cell', for_loop_var_1100)
                        # SSA begins for a for statement (line 422)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Evaluating a boolean operation
                        
                        
                        # Obtaining the type of the subscript
                        int_1101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 33), 'int')
                        # Getting the type of 'entry' (line 423)
                        entry_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 27), 'entry')
                        # Obtaining the member '__getitem__' of a type (line 423)
                        getitem___1103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 27), entry_1102, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
                        subscript_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 423, 27), getitem___1103, int_1101)
                        
                        
                        # Obtaining the type of the subscript
                        int_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 44), 'int')
                        # Getting the type of 'cell' (line 423)
                        cell_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 39), 'cell')
                        # Obtaining the member '__getitem__' of a type (line 423)
                        getitem___1107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 39), cell_1106, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
                        subscript_call_result_1108 = invoke(stypy.reporting.localization.Localization(__file__, 423, 39), getitem___1107, int_1105)
                        
                        # Applying the binary operator '==' (line 423)
                        result_eq_1109 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 27), '==', subscript_call_result_1104, subscript_call_result_1108)
                        
                        
                        
                        # Obtaining the type of the subscript
                        int_1110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 57), 'int')
                        # Getting the type of 'entry' (line 423)
                        entry_1111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 51), 'entry')
                        # Obtaining the member '__getitem__' of a type (line 423)
                        getitem___1112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 51), entry_1111, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
                        subscript_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 423, 51), getitem___1112, int_1110)
                        
                        
                        # Obtaining the type of the subscript
                        int_1114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 68), 'int')
                        # Getting the type of 'cell' (line 423)
                        cell_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 63), 'cell')
                        # Obtaining the member '__getitem__' of a type (line 423)
                        getitem___1116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 63), cell_1115, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
                        subscript_call_result_1117 = invoke(stypy.reporting.localization.Localization(__file__, 423, 63), getitem___1116, int_1114)
                        
                        # Applying the binary operator '==' (line 423)
                        result_eq_1118 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 51), '==', subscript_call_result_1113, subscript_call_result_1117)
                        
                        # Applying the binary operator 'and' (line 423)
                        result_and_keyword_1119 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 27), 'and', result_eq_1109, result_eq_1118)
                        
                        # Testing if the type of an if condition is none (line 423)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 423, 24), result_and_keyword_1119):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 423)
                            if_condition_1120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 24), result_and_keyword_1119)
                            # Assigning a type to the variable 'if_condition_1120' (line 423)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 24), 'if_condition_1120', if_condition_1120)
                            # SSA begins for if statement (line 423)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Call to a Name (line 424):
                            
                            # Call to Column(...): (line 424)
                            # Processing the call keyword arguments (line 424)
                            kwargs_1122 = {}
                            # Getting the type of 'Column' (line 424)
                            Column_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 38), 'Column', False)
                            # Calling Column(args, kwargs) (line 424)
                            Column_call_result_1123 = invoke(stypy.reporting.localization.Localization(__file__, 424, 38), Column_1121, *[], **kwargs_1122)
                            
                            # Assigning a type to the variable 'element' (line 424)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 28), 'element', Column_call_result_1123)
                            
                            # Assigning a Name to a Attribute (line 425):
                            # Getting the type of 'element' (line 425)
                            element_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 52), 'element')
                            # Getting the type of 'rowelement' (line 425)
                            rowelement_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 28), 'rowelement')
                            # Obtaining the member 'right' of a type (line 425)
                            right_1126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 28), rowelement_1125, 'right')
                            # Setting the type of the member 'left' of a type (line 425)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 28), right_1126, 'left', element_1124)
                            
                            # Assigning a Attribute to a Attribute (line 426):
                            # Getting the type of 'rowelement' (line 426)
                            rowelement_1127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 44), 'rowelement')
                            # Obtaining the member 'right' of a type (line 426)
                            right_1128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 44), rowelement_1127, 'right')
                            # Getting the type of 'element' (line 426)
                            element_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 28), 'element')
                            # Setting the type of the member 'right' of a type (line 426)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 28), element_1129, 'right', right_1128)
                            
                            # Assigning a Name to a Attribute (line 427):
                            # Getting the type of 'element' (line 427)
                            element_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 47), 'element')
                            # Getting the type of 'rowelement' (line 427)
                            rowelement_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 28), 'rowelement')
                            # Setting the type of the member 'right' of a type (line 427)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 28), rowelement_1131, 'right', element_1130)
                            
                            # Assigning a Name to a Attribute (line 428):
                            # Getting the type of 'rowelement' (line 428)
                            rowelement_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 43), 'rowelement')
                            # Getting the type of 'element' (line 428)
                            element_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 28), 'element')
                            # Setting the type of the member 'left' of a type (line 428)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 28), element_1133, 'left', rowelement_1132)
                            
                            # Assigning a Name to a Attribute (line 430):
                            # Getting the type of 'column' (line 430)
                            column_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 45), 'column')
                            # Getting the type of 'element' (line 430)
                            element_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 28), 'element')
                            # Setting the type of the member 'column' of a type (line 430)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 28), element_1135, 'column', column_1134)
                            
                            # Assigning a Attribute to a Attribute (line 431):
                            # Getting the type of 'column' (line 431)
                            column_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 41), 'column')
                            # Obtaining the member 'up' of a type (line 431)
                            up_1137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 41), column_1136, 'up')
                            # Getting the type of 'element' (line 431)
                            element_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 28), 'element')
                            # Setting the type of the member 'up' of a type (line 431)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 28), element_1138, 'up', up_1137)
                            
                            # Assigning a Name to a Attribute (line 432):
                            # Getting the type of 'column' (line 432)
                            column_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 43), 'column')
                            # Getting the type of 'element' (line 432)
                            element_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 28), 'element')
                            # Setting the type of the member 'down' of a type (line 432)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 28), element_1140, 'down', column_1139)
                            
                            # Assigning a Name to a Attribute (line 433):
                            # Getting the type of 'element' (line 433)
                            element_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 45), 'element')
                            # Getting the type of 'column' (line 433)
                            column_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), 'column')
                            # Obtaining the member 'up' of a type (line 433)
                            up_1143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 28), column_1142, 'up')
                            # Setting the type of the member 'down' of a type (line 433)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 28), up_1143, 'down', element_1141)
                            
                            # Assigning a Name to a Attribute (line 434):
                            # Getting the type of 'element' (line 434)
                            element_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 40), 'element')
                            # Getting the type of 'column' (line 434)
                            column_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'column')
                            # Setting the type of the member 'up' of a type (line 434)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 28), column_1145, 'up', element_1144)
                            
                            # Assigning a Name to a Name (line 435):
                            # Getting the type of 'element' (line 435)
                            element_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 41), 'element')
                            # Assigning a type to the variable 'rowelement' (line 435)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 28), 'rowelement', element_1146)
                            
                            # Getting the type of 'column' (line 436)
                            column_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 28), 'column')
                            # Obtaining the member 'size' of a type (line 436)
                            size_1148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 28), column_1147, 'size')
                            int_1149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 43), 'int')
                            # Applying the binary operator '+=' (line 436)
                            result_iadd_1150 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 28), '+=', size_1148, int_1149)
                            # Getting the type of 'column' (line 436)
                            column_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 28), 'column')
                            # Setting the type of the member 'size' of a type (line 436)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 28), column_1151, 'size', result_iadd_1150)
                            
                            # SSA join for if statement (line 423)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    # Assigning a Attribute to a Name (line 437):
                    # Getting the type of 'column' (line 437)
                    column_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 29), 'column')
                    # Obtaining the member 'right' of a type (line 437)
                    right_1153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 29), column_1152, 'right')
                    # Assigning a type to the variable 'column' (line 437)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'column', right_1153)
                    # SSA join for while statement (line 420)
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()



@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 442, 0, False)
    
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

    
    
    # SSA begins for try-except statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to setroot(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'root' (line 444)
    root_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'root', False)
    # Processing the call keyword arguments (line 444)
    kwargs_1156 = {}
    # Getting the type of 'setroot' (line 444)
    setroot_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'setroot', False)
    # Calling setroot(args, kwargs) (line 444)
    setroot_call_result_1157 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), setroot_1154, *[root_1155], **kwargs_1156)
    
    
    # Call to search(...): (line 446)
    # Processing the call arguments (line 446)
    int_1159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 15), 'int')
    # Processing the call keyword arguments (line 446)
    kwargs_1160 = {}
    # Getting the type of 'search' (line 446)
    search_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'search', False)
    # Calling search(args, kwargs) (line 446)
    search_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), search_1158, *[int_1159], **kwargs_1160)
    
    # SSA branch for the except part of a try statement (line 443)
    # SSA branch for the except 'SystemExit' branch of a try statement (line 443)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 450)
    True_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'stypy_return_type', True_1162)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 442)
    stypy_return_type_1163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1163)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1163

# Assigning a type to the variable 'run' (line 442)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 0), 'run', run)

# Call to run(...): (line 453)
# Processing the call keyword arguments (line 453)
kwargs_1165 = {}
# Getting the type of 'run' (line 453)
run_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), 'run', False)
# Calling run(args, kwargs) (line 453)
run_call_result_1166 = invoke(stypy.reporting.localization.Localization(__file__, 453, 0), run_1164, *[], **kwargs_1165)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

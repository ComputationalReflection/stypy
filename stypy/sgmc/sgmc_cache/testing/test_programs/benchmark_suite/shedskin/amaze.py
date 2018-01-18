
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Amaze - A completely object-oriented Pythonic maze generator/solver.
3: This can generate random mazes and solve them. It should be
4: able to solve any kind of maze and inform you in case a maze is
5: unsolveable.
6: 
7: This uses a very simple representation of a mze. A maze is
8: represented as an mxn matrix with each point value being either
9: 0 or 1. Points with value 0 represent paths and those with
10: value 1 represent blocks. The problem is to find a path from
11: point A to point B in the matrix.
12: 
13: The matrix is represented internally as a list of lists.
14: 
15: Have fun :-)
16: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/496884
17: '''
18: import sys
19: import random
20: import os
21: 
22: def Relative(path):
23:     return os.path.join(os.path.dirname(__file__), path)
24: 
25: class MazeReaderException(Exception):
26:     pass
27: 
28: STDIN = 0
29: FILE_ = 1
30: SOCKET = 2
31: 
32: PATH = -1
33: START = -2
34: EXIT = -3
35: 
36: class MazeReader(object):
37: 
38:     def __init__(self):
39:         self.maze_rows = []
40:         pass
41: 
42:     def readStdin(self):
43:         #print 'Enter a maze'
44:         #print 'You can enter a maze row by row'
45:         #print
46: 
47:         data = raw_input('Enter the dimension of the maze as Width X Height: ')
48:         w1, h1 = data.split() # XXX SS 
49:         w, h  = int(w1), int(h1)
50: 
51:         for x in range(h):
52:             row = ''
53:             while not row:
54:                 row = raw_input('Enter row number %d: ' % (x+1))
55:             rowsplit = [int(y) for y in row.split()] # XXX SS
56:             if len(rowsplit) != w:
57:                 raise MazeReaderException('invalid size of maze row')
58:             self.maze_rows.append(rowsplit)
59: 
60:     def readFile(self):
61:         fname = Relative('testdata/maze.txt') #raw_input('Enter maze filename: ')
62:         try:
63:             f = open(fname)
64:             lines = f.readlines()
65:             f.close()
66:             lines = [ line for line in lines if line.strip() ]
67:             w = len(lines[0].split())
68:             for line in lines:
69:                 row = [int(y) for y in line.split()]
70:                 if len(row) != w:
71:                     raise MazeReaderException('Invalid maze file - error in maze dimensions')
72:                 else:
73:                     self.maze_rows.append(row)
74:         except (IOError, OSError), e:
75:             raise MazeReaderException(str(e))
76: 
77:     def getData(self):
78:         return self.maze_rows
79: 
80:     def readMaze(self, source=STDIN):
81:         if source==STDIN:
82:             self.readStdin()
83:         elif source == FILE_:
84:             self.readFile()
85: 
86:         return self.getData()
87: 
88: class MazeFactory(object):
89:     def makeMaze(self, source=STDIN):
90:         reader = MazeReader()
91:         return Maze(reader.readMaze(source))
92: 
93: class MazeError(Exception):
94:     pass
95: 
96: class Maze(object):
97:     def __init__(self, rows=[[]]):
98:         self._rows = rows
99:         self.__validate()
100:         self.__normalize()
101: 
102:     def __str__(self):
103:         s = '\n'
104:         for row in self._rows:
105:             for item in row:
106:                 if item == PATH: sitem = '*'
107:                 elif item == START: sitem = 'S'
108:                 elif item == EXIT: sitem = 'E'
109:                 else: sitem = str(item)
110: 
111:                 s = ''.join((s,'  ',sitem,'   '))
112:             s = ''.join((s,'\n\n'))
113: 
114:         return s
115: 
116:     def __validate(self):
117:         width = len(self._rows[0])
118:         widths = [len(row) for row in self._rows]
119:         if widths.count(width) != len(widths):
120:             raise MazeError('Invalid maze!')
121: 
122:         self._height = len(self._rows)
123:         self._width = width
124: 
125:     def __normalize(self):
126:         for x in range(len(self._rows)):
127:             row = self._rows[x]
128:             row = [min(int(y), 1) for y in row] #map(lambda x: min(int(x), 1), row) # SS
129:             self._rows[x] = row
130: 
131:     def validatePoint(self, pt):
132:         x,y = pt
133:         w = self._width
134:         h = self._height
135: 
136:         # Don't support Pythonic negative indices
137:         if x > w - 1 or x<0:
138:             raise MazeError('x co-ordinate out of range!')
139: 
140:         if y > h - 1 or y<0:
141:             raise MazeError('y co-ordinate out of range!')
142: 
143:         pass # SS
144: 
145:     def getItem(self, x, y):
146:         self.validatePoint((x,y))
147: 
148:         w = self._width
149:         h = self._height
150: 
151:         row = self._rows[h-y-1]
152:         return row[x]
153: 
154:     def setItem(self, x, y, value):
155:         h = self._height
156: 
157:         self.validatePoint((x,y))
158:         row = self._rows[h-y-1]
159:         row[x] = value
160: 
161:     def getNeighBours(self, pt):
162:         self.validatePoint(pt)
163: 
164:         x,y = pt
165: 
166:         h = self._height
167:         w = self._width
168: 
169:         poss_nbors = (x-1,y),(x-1,y+1),(x,y+1),(x+1,y+1),(x+1,y),(x+1,y-1),(x,y-1),(x-1,y-1)
170: 
171:         nbors = []
172:         for xx,yy in poss_nbors:
173:             if (xx>=0 and xx<=w-1) and (yy>=0 and yy<=h-1):
174:                 nbors.append((xx,yy))
175: 
176:         return nbors
177: 
178:     def getExitPoints(self, pt):
179:         exits = []
180:         for xx,yy in self.getNeighBours(pt):
181:             if self.getItem(xx,yy)==0: # SS
182:                 exits.append((xx,yy))
183: 
184:         return exits
185: 
186:     def calcDistance(self, pt1, pt2):
187:         self.validatePoint(pt1)
188:         self.validatePoint(pt2)
189: 
190:         x1,y1 = pt1
191:         x2,y2 = pt2
192: 
193:         return pow( (pow((x1-x2), 2) + pow((y1-y2),2)), 0.5)
194: 
195: class MazeSolver(object):
196:     def __init__(self, maze):
197:         self.maze = maze
198:         self._start = (0,0)
199:         self._end = (0,0)
200:         self._current = (0,0)
201:         self._steps = 0
202:         self._path = []
203:         self._tryalternate = False
204:         self._trynextbest = False
205:         self._disputed = (0,0)
206:         self._loops = 0
207:         self._retrace = False
208:         self._numretraces = 0
209: 
210:     def setStartPoint(self, pt):
211:         self.maze.validatePoint(pt)
212:         self._start = pt
213: 
214:     def setEndPoint(self, pt):
215:         self.maze.validatePoint(pt)
216:         self._end = pt
217: 
218:     def boundaryCheck(self):
219:         exits1 = self.maze.getExitPoints(self._start)
220:         exits2 = self.maze.getExitPoints(self._end)
221: 
222:         if len(exits1)==0 or len(exits2)==0:
223:             return False
224: 
225:         return True
226: 
227:     def setCurrentPoint(self, point):
228:         self._current = point
229:         self._path.append(point)
230: 
231:     def isSolved(self):
232:         return (self._current == self._end)
233: 
234:     def getNextPoint(self):
235:         points = self.maze.getExitPoints(self._current)
236: 
237:         point = self.getBestPoint(points)
238: 
239:         while self.checkClosedLoop(point):
240: 
241:             if self.endlessLoop():
242:                 #print self._loops
243:                 point = None
244:                 break
245: 
246:             point2 = point
247:             if point==self._start and len(self._path)>2:
248:                 self._tryalternate = True
249:                 break
250:             else:
251:                 point = self.getNextClosestPointNotInPath(points, point2)
252:                 if not point:
253:                     self.retracePath()
254:                     self._tryalternate = True
255:                     point = self._start
256:                     break
257: 
258:         return point
259: 
260:     def retracePath(self):
261:         #print 'Retracing...'
262:         self._retrace = True
263: 
264:         path2 = self._path[:]
265:         path2.reverse()
266: 
267:         idx = path2.index(self._start)
268:         self._path += self._path[-2:idx:-1]
269:         self._numretraces += 1
270: 
271:     def endlessLoop(self):
272:         if self._loops>100:
273:             #print 'Seems to be hitting an endless loop.'
274:             return True
275:         elif self._numretraces>8:
276:             #print 'Seem to be retracing loop.'
277:             return True
278: 
279:         return False
280: 
281:     def checkClosedLoop(self, point):
282:         l = range(0, len(self._path)-1, 2)
283:         l.reverse()
284: 
285:         for x in l:
286:             if self._path[x] == point:
287:                 self._loops += 1
288:                 return True
289: 
290:         return False
291: 
292:     def getBestPoint(self, points):
293:         point = self.getClosestPoint(points)
294:         point2 = point
295:         altpoint = point
296: 
297:         if point2 in self._path:
298:             point = self.getNextClosestPointNotInPath(points, point2)
299:             if not point:
300:                 point = point2
301: 
302:         if self._tryalternate:
303:             point = self.getAlternatePoint(points, altpoint)
304:             #print 'Trying alternate...',self._current, point
305: 
306:         self._trynextbest = False
307:         self._tryalternate = False
308:         self._retrace = False
309: 
310:         return point
311: 
312:     def sortPoints(self, points):
313:         distances = [self.maze.calcDistance(point, self._end) for point in points]
314:         distances2 = distances[:]
315: 
316:         distances.sort()
317: 
318:         points2 = [()]*len(points) # SS
319:         count = 0
320: 
321:         for dist in distances:
322:             idx = distances2.index(dist)
323:             point = points[idx]
324: 
325:             while point in points2:
326:                 idx = distances2.index(dist, idx+1)
327:                 point = points[idx]
328: 
329:             points2[count] = point
330:             count += 1
331: 
332:         return points2
333: 
334:     def getClosestPoint(self, points):
335:         points2 = self.sortPoints(points)
336: 
337:         closest = points2[0]
338:         return closest
339: 
340:     def getAlternatePoint(self, points, point):
341:         points2 = points[:]
342:         #print points2, point
343: 
344:         points2.remove(point)
345:         if points2:
346:             return random.choice(points2)
347: 
348:         return None
349: 
350:     def getNextClosestPoint(self, points, point):
351:         points2 = self.sortPoints(points)
352:         idx = points2.index(point)
353: 
354:         try:
355:             return points2[idx+1]
356:         except:
357:             return None 
358: 
359:     def getNextClosestPointNotInPath(self, points, point):
360: 
361: 
362:         point2 = self.getNextClosestPoint(points, point)
363:         while point2 in self._path:
364:             point2 = self.getNextClosestPoint(points, point2)
365: 
366:         return point2
367: 
368:     def solve(self):
369:         #print 'Starting point is', self._start
370:         #print 'Ending point is', self._end
371: 
372:         # First check if both start and end are same
373:         if self._start == self._end:
374:             #print 'Start/end points are the same. Trivial maze.'
375:             #print [self._start, self._end]
376:             return None
377: 
378:         # Check boundary conditions
379:         if not self.boundaryCheck():
380:             #print 'Either start/end point are unreachable. Maze cannot be solved.'
381:             return None
382: 
383:         # Proper maze
384:         #print 'Maze is a proper maze.'
385: 
386:         # Initialize solver
387:         self.setCurrentPoint(self._start)
388: 
389:         unsolvable = False
390: 
391:         while not self.isSolved():
392:             self._steps += 1
393:             pt = self.getNextPoint()
394: 
395:             if pt:
396:                 self.setCurrentPoint(pt)
397:             else:
398:                 #print 'Dead-lock - maze unsolvable'
399:                 unsolvable = True
400:                 break
401: 
402:         if not unsolvable:
403:             pass #print 'Solution path is',self._path
404:         else:
405:             pass#print 'Path till deadlock is',self._path
406: 
407:         self.printResult()
408: 
409:     def printResult(self):
410:         ''' Print the maze showing the path '''
411: 
412:         for x,y in self._path:
413:             self.maze.setItem(x,y,PATH)
414: 
415:         self.maze.setItem(self._start[0], self._start[1], START)
416:         self.maze.setItem(self._end[0], self._end[1], EXIT)
417: 
418:         #print 'Maze with solution path'
419:         #print self.maze
420: 
421: 
422: class MazeGame(object):
423:     def __init__(self):
424:         self._start = (0,0)
425:         self._end = (0,0)
426: 
427:     #def createMaze(self):
428:     #    return None
429: #
430: #    def getStartEndPoints(self, maze):
431: #        return None
432: 
433:     def runGame(self):
434:         maze = self.createMaze()
435:         if not maze:
436:             return None
437: 
438:         #print maze
439:         self.getStartEndPoints(maze)
440: 
441:         #open('maze.txt','w').write(str(maze))
442: 
443:         solver = MazeSolver(maze)
444: 
445:         #open ('maze_pts.txt','w').write(str(self._start) + ' ' + str(self._end) + '\n')
446:         solver.setStartPoint(self._start)
447:         solver.setEndPoint(self._end)
448:         solver.solve()
449: 
450: class FilebasedMazeGame(MazeGame):
451: 
452:     def createMaze(self):
453:         f = MazeFactory()
454:         m = f.makeMaze(FILE_)
455:         #print m
456:         return m
457: 
458:     def getStartEndPoints(self, maze):
459: 
460:         while True:
461:             try:
462:                 #pt1 = raw_input('Enter starting point: ')
463:                 pt1 = '0 4'
464:                 x,y = pt1.split()
465:                 self._start = (int(x), int(y))
466:                 maze.validatePoint(self._start)
467:                 break
468:             except:
469:                 pass
470: 
471:         while True:
472:             try:
473:                 pt2 = '5 4' #pt2 = raw_input('Enter ending point: ')
474:                 x,y = pt2.split()
475:                 self._end = (int(x), int(y))
476:                 maze.validatePoint(self._end)
477:                 break
478:             except:
479:                 pass
480: 
481: def run():
482:     game = FilebasedMazeGame()
483:     for x in range(10000):
484:         game.runGame()
485:     return True
486: 
487: run()
488: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nAmaze - A completely object-oriented Pythonic maze generator/solver.\nThis can generate random mazes and solve them. It should be\nable to solve any kind of maze and inform you in case a maze is\nunsolveable.\n\nThis uses a very simple representation of a mze. A maze is\nrepresented as an mxn matrix with each point value being either\n0 or 1. Points with value 0 represent paths and those with\nvalue 1 represent blocks. The problem is to find a path from\npoint A to point B in the matrix.\n\nThe matrix is represented internally as a list of lists.\n\nHave fun :-)\nhttp://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/496884\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import sys' statement (line 18)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import random' statement (line 19)
import random

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import os' statement (line 20)
import os

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 22, 0, False)
    
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

    
    # Call to join(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to dirname(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of '__file__' (line 23)
    file___27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 40), '__file__', False)
    # Processing the call keyword arguments (line 23)
    kwargs_28 = {}
    # Getting the type of 'os' (line 23)
    os_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 23)
    path_25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 24), os_24, 'path')
    # Obtaining the member 'dirname' of a type (line 23)
    dirname_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 24), path_25, 'dirname')
    # Calling dirname(args, kwargs) (line 23)
    dirname_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 23, 24), dirname_26, *[file___27], **kwargs_28)
    
    # Getting the type of 'path' (line 23)
    path_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 51), 'path', False)
    # Processing the call keyword arguments (line 23)
    kwargs_31 = {}
    # Getting the type of 'os' (line 23)
    os_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 23)
    path_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), os_21, 'path')
    # Obtaining the member 'join' of a type (line 23)
    join_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), path_22, 'join')
    # Calling join(args, kwargs) (line 23)
    join_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), join_23, *[dirname_call_result_29, path_30], **kwargs_31)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', join_call_result_32)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_33

# Assigning a type to the variable 'Relative' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'Relative', Relative)
# Declaration of the 'MazeReaderException' class
# Getting the type of 'Exception' (line 25)
Exception_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'Exception')

class MazeReaderException(Exception_34, ):
    pass

# Assigning a type to the variable 'MazeReaderException' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'MazeReaderException', MazeReaderException)

# Assigning a Num to a Name (line 28):

# Assigning a Num to a Name (line 28):
int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
# Assigning a type to the variable 'STDIN' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'STDIN', int_35)

# Assigning a Num to a Name (line 29):

# Assigning a Num to a Name (line 29):
int_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'int')
# Assigning a type to the variable 'FILE_' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'FILE_', int_36)

# Assigning a Num to a Name (line 30):

# Assigning a Num to a Name (line 30):
int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'int')
# Assigning a type to the variable 'SOCKET' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'SOCKET', int_37)

# Assigning a Num to a Name (line 32):

# Assigning a Num to a Name (line 32):
int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 7), 'int')
# Assigning a type to the variable 'PATH' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'PATH', int_38)

# Assigning a Num to a Name (line 33):

# Assigning a Num to a Name (line 33):
int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'int')
# Assigning a type to the variable 'START' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'START', int_39)

# Assigning a Num to a Name (line 34):

# Assigning a Num to a Name (line 34):
int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 7), 'int')
# Assigning a type to the variable 'EXIT' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'EXIT', int_40)
# Declaration of the 'MazeReader' class

class MazeReader(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeReader.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 39):
        
        # Assigning a List to a Attribute (line 39):
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        
        # Getting the type of 'self' (line 39)
        self_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'maze_rows' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_42, 'maze_rows', list_41)
        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def readStdin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'readStdin'
        module_type_store = module_type_store.open_function_context('readStdin', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeReader.readStdin.__dict__.__setitem__('stypy_localization', localization)
        MazeReader.readStdin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeReader.readStdin.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeReader.readStdin.__dict__.__setitem__('stypy_function_name', 'MazeReader.readStdin')
        MazeReader.readStdin.__dict__.__setitem__('stypy_param_names_list', [])
        MazeReader.readStdin.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeReader.readStdin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeReader.readStdin.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeReader.readStdin.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeReader.readStdin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeReader.readStdin.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeReader.readStdin', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'readStdin', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'readStdin(...)' code ##################

        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to raw_input(...): (line 47)
        # Processing the call arguments (line 47)
        str_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', 'Enter the dimension of the maze as Width X Height: ')
        # Processing the call keyword arguments (line 47)
        kwargs_45 = {}
        # Getting the type of 'raw_input' (line 47)
        raw_input_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 47)
        raw_input_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), raw_input_43, *[str_44], **kwargs_45)
        
        # Assigning a type to the variable 'data' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'data', raw_input_call_result_46)
        
        # Assigning a Call to a Tuple (line 48):
        
        # Assigning a Call to a Name:
        
        # Call to split(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_49 = {}
        # Getting the type of 'data' (line 48)
        data_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'data', False)
        # Obtaining the member 'split' of a type (line 48)
        split_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 17), data_47, 'split')
        # Calling split(args, kwargs) (line 48)
        split_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 48, 17), split_48, *[], **kwargs_49)
        
        # Assigning a type to the variable 'call_assignment_1' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'call_assignment_1', split_call_result_50)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        # Processing the call keyword arguments
        kwargs_54 = {}
        # Getting the type of 'call_assignment_1' (line 48)
        call_assignment_1_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), call_assignment_1_51, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___52, *[int_53], **kwargs_54)
        
        # Assigning a type to the variable 'call_assignment_2' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'call_assignment_2', getitem___call_result_55)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'call_assignment_2' (line 48)
        call_assignment_2_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'call_assignment_2')
        # Assigning a type to the variable 'w1' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'w1', call_assignment_2_56)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        # Processing the call keyword arguments
        kwargs_60 = {}
        # Getting the type of 'call_assignment_1' (line 48)
        call_assignment_1_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), call_assignment_1_57, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___58, *[int_59], **kwargs_60)
        
        # Assigning a type to the variable 'call_assignment_3' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'call_assignment_3', getitem___call_result_61)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'call_assignment_3' (line 48)
        call_assignment_3_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'call_assignment_3')
        # Assigning a type to the variable 'h1' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'h1', call_assignment_3_62)
        
        # Assigning a Tuple to a Tuple (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to int(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'w1' (line 49)
        w1_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'w1', False)
        # Processing the call keyword arguments (line 49)
        kwargs_65 = {}
        # Getting the type of 'int' (line 49)
        int_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'int', False)
        # Calling int(args, kwargs) (line 49)
        int_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), int_63, *[w1_64], **kwargs_65)
        
        # Assigning a type to the variable 'tuple_assignment_4' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_4', int_call_result_66)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to int(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'h1' (line 49)
        h1_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'h1', False)
        # Processing the call keyword arguments (line 49)
        kwargs_69 = {}
        # Getting the type of 'int' (line 49)
        int_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'int', False)
        # Calling int(args, kwargs) (line 49)
        int_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 49, 25), int_67, *[h1_68], **kwargs_69)
        
        # Assigning a type to the variable 'tuple_assignment_5' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_5', int_call_result_70)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_assignment_4' (line 49)
        tuple_assignment_4_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_4')
        # Assigning a type to the variable 'w' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'w', tuple_assignment_4_71)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_assignment_5' (line 49)
        tuple_assignment_5_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_5')
        # Assigning a type to the variable 'h' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'h', tuple_assignment_5_72)
        
        
        # Call to range(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'h' (line 51)
        h_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'h', False)
        # Processing the call keyword arguments (line 51)
        kwargs_75 = {}
        # Getting the type of 'range' (line 51)
        range_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'range', False)
        # Calling range(args, kwargs) (line 51)
        range_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), range_73, *[h_74], **kwargs_75)
        
        # Testing the type of a for loop iterable (line 51)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 8), range_call_result_76)
        # Getting the type of the for loop variable (line 51)
        for_loop_var_77 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 8), range_call_result_76)
        # Assigning a type to the variable 'x' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'x', for_loop_var_77)
        # SSA begins for a for statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Str to a Name (line 52):
        
        # Assigning a Str to a Name (line 52):
        str_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'str', '')
        # Assigning a type to the variable 'row' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'row', str_78)
        
        
        # Getting the type of 'row' (line 53)
        row_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'row')
        # Applying the 'not' unary operator (line 53)
        result_not__80 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 18), 'not', row_79)
        
        # Testing the type of an if condition (line 53)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 12), result_not__80)
        # SSA begins for while statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to raw_input(...): (line 54)
        # Processing the call arguments (line 54)
        str_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'str', 'Enter row number %d: ')
        # Getting the type of 'x' (line 54)
        x_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 59), 'x', False)
        int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 61), 'int')
        # Applying the binary operator '+' (line 54)
        result_add_85 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 59), '+', x_83, int_84)
        
        # Applying the binary operator '%' (line 54)
        result_mod_86 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 32), '%', str_82, result_add_85)
        
        # Processing the call keyword arguments (line 54)
        kwargs_87 = {}
        # Getting the type of 'raw_input' (line 54)
        raw_input_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 54)
        raw_input_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 54, 22), raw_input_81, *[result_mod_86], **kwargs_87)
        
        # Assigning a type to the variable 'row' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'row', raw_input_call_result_88)
        # SSA join for while statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Name (line 55):
        
        # Assigning a ListComp to a Name (line 55):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_95 = {}
        # Getting the type of 'row' (line 55)
        row_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'row', False)
        # Obtaining the member 'split' of a type (line 55)
        split_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 40), row_93, 'split')
        # Calling split(args, kwargs) (line 55)
        split_call_result_96 = invoke(stypy.reporting.localization.Localization(__file__, 55, 40), split_94, *[], **kwargs_95)
        
        comprehension_97 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 24), split_call_result_96)
        # Assigning a type to the variable 'y' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'y', comprehension_97)
        
        # Call to int(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'y' (line 55)
        y_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'y', False)
        # Processing the call keyword arguments (line 55)
        kwargs_91 = {}
        # Getting the type of 'int' (line 55)
        int_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'int', False)
        # Calling int(args, kwargs) (line 55)
        int_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), int_89, *[y_90], **kwargs_91)
        
        list_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 24), list_98, int_call_result_92)
        # Assigning a type to the variable 'rowsplit' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'rowsplit', list_98)
        
        
        
        # Call to len(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'rowsplit' (line 56)
        rowsplit_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'rowsplit', False)
        # Processing the call keyword arguments (line 56)
        kwargs_101 = {}
        # Getting the type of 'len' (line 56)
        len_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'len', False)
        # Calling len(args, kwargs) (line 56)
        len_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), len_99, *[rowsplit_100], **kwargs_101)
        
        # Getting the type of 'w' (line 56)
        w_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'w')
        # Applying the binary operator '!=' (line 56)
        result_ne_104 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), '!=', len_call_result_102, w_103)
        
        # Testing the type of an if condition (line 56)
        if_condition_105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), result_ne_104)
        # Assigning a type to the variable 'if_condition_105' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_105', if_condition_105)
        # SSA begins for if statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to MazeReaderException(...): (line 57)
        # Processing the call arguments (line 57)
        str_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 42), 'str', 'invalid size of maze row')
        # Processing the call keyword arguments (line 57)
        kwargs_108 = {}
        # Getting the type of 'MazeReaderException' (line 57)
        MazeReaderException_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'MazeReaderException', False)
        # Calling MazeReaderException(args, kwargs) (line 57)
        MazeReaderException_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 57, 22), MazeReaderException_106, *[str_107], **kwargs_108)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 57, 16), MazeReaderException_call_result_109, 'raise parameter', BaseException)
        # SSA join for if statement (line 56)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'rowsplit' (line 58)
        rowsplit_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'rowsplit', False)
        # Processing the call keyword arguments (line 58)
        kwargs_114 = {}
        # Getting the type of 'self' (line 58)
        self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'self', False)
        # Obtaining the member 'maze_rows' of a type (line 58)
        maze_rows_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), self_110, 'maze_rows')
        # Obtaining the member 'append' of a type (line 58)
        append_112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), maze_rows_111, 'append')
        # Calling append(args, kwargs) (line 58)
        append_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), append_112, *[rowsplit_113], **kwargs_114)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'readStdin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'readStdin' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'readStdin'
        return stypy_return_type_116


    @norecursion
    def readFile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'readFile'
        module_type_store = module_type_store.open_function_context('readFile', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeReader.readFile.__dict__.__setitem__('stypy_localization', localization)
        MazeReader.readFile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeReader.readFile.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeReader.readFile.__dict__.__setitem__('stypy_function_name', 'MazeReader.readFile')
        MazeReader.readFile.__dict__.__setitem__('stypy_param_names_list', [])
        MazeReader.readFile.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeReader.readFile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeReader.readFile.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeReader.readFile.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeReader.readFile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeReader.readFile.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeReader.readFile', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'readFile', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'readFile(...)' code ##################

        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to Relative(...): (line 61)
        # Processing the call arguments (line 61)
        str_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'str', 'testdata/maze.txt')
        # Processing the call keyword arguments (line 61)
        kwargs_119 = {}
        # Getting the type of 'Relative' (line 61)
        Relative_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'Relative', False)
        # Calling Relative(args, kwargs) (line 61)
        Relative_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), Relative_117, *[str_118], **kwargs_119)
        
        # Assigning a type to the variable 'fname' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'fname', Relative_call_result_120)
        
        
        # SSA begins for try-except statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to open(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'fname' (line 63)
        fname_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'fname', False)
        # Processing the call keyword arguments (line 63)
        kwargs_123 = {}
        # Getting the type of 'open' (line 63)
        open_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'open', False)
        # Calling open(args, kwargs) (line 63)
        open_call_result_124 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), open_121, *[fname_122], **kwargs_123)
        
        # Assigning a type to the variable 'f' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'f', open_call_result_124)
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to readlines(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_127 = {}
        # Getting the type of 'f' (line 64)
        f_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'f', False)
        # Obtaining the member 'readlines' of a type (line 64)
        readlines_126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 20), f_125, 'readlines')
        # Calling readlines(args, kwargs) (line 64)
        readlines_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), readlines_126, *[], **kwargs_127)
        
        # Assigning a type to the variable 'lines' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'lines', readlines_call_result_128)
        
        # Call to close(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_131 = {}
        # Getting the type of 'f' (line 65)
        f_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 65)
        close_130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), f_129, 'close')
        # Calling close(args, kwargs) (line 65)
        close_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), close_130, *[], **kwargs_131)
        
        
        # Assigning a ListComp to a Name (line 66):
        
        # Assigning a ListComp to a Name (line 66):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'lines' (line 66)
        lines_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 39), 'lines')
        comprehension_139 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), lines_138)
        # Assigning a type to the variable 'line' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'line', comprehension_139)
        
        # Call to strip(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_136 = {}
        # Getting the type of 'line' (line 66)
        line_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 48), 'line', False)
        # Obtaining the member 'strip' of a type (line 66)
        strip_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 48), line_134, 'strip')
        # Calling strip(args, kwargs) (line 66)
        strip_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 66, 48), strip_135, *[], **kwargs_136)
        
        # Getting the type of 'line' (line 66)
        line_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'line')
        list_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), list_140, line_133)
        # Assigning a type to the variable 'lines' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'lines', list_140)
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to len(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to split(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_147 = {}
        
        # Obtaining the type of the subscript
        int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'int')
        # Getting the type of 'lines' (line 67)
        lines_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'lines', False)
        # Obtaining the member '__getitem__' of a type (line 67)
        getitem___144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), lines_143, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 67)
        subscript_call_result_145 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), getitem___144, int_142)
        
        # Obtaining the member 'split' of a type (line 67)
        split_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), subscript_call_result_145, 'split')
        # Calling split(args, kwargs) (line 67)
        split_call_result_148 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), split_146, *[], **kwargs_147)
        
        # Processing the call keyword arguments (line 67)
        kwargs_149 = {}
        # Getting the type of 'len' (line 67)
        len_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'len', False)
        # Calling len(args, kwargs) (line 67)
        len_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), len_141, *[split_call_result_148], **kwargs_149)
        
        # Assigning a type to the variable 'w' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'w', len_call_result_150)
        
        # Getting the type of 'lines' (line 68)
        lines_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'lines')
        # Testing the type of a for loop iterable (line 68)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 12), lines_151)
        # Getting the type of the for loop variable (line 68)
        for_loop_var_152 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 12), lines_151)
        # Assigning a type to the variable 'line' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'line', for_loop_var_152)
        # SSA begins for a for statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 69):
        
        # Assigning a ListComp to a Name (line 69):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_159 = {}
        # Getting the type of 'line' (line 69)
        line_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 39), 'line', False)
        # Obtaining the member 'split' of a type (line 69)
        split_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 39), line_157, 'split')
        # Calling split(args, kwargs) (line 69)
        split_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 69, 39), split_158, *[], **kwargs_159)
        
        comprehension_161 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 23), split_call_result_160)
        # Assigning a type to the variable 'y' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'y', comprehension_161)
        
        # Call to int(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'y' (line 69)
        y_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'y', False)
        # Processing the call keyword arguments (line 69)
        kwargs_155 = {}
        # Getting the type of 'int' (line 69)
        int_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'int', False)
        # Calling int(args, kwargs) (line 69)
        int_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), int_153, *[y_154], **kwargs_155)
        
        list_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 23), list_162, int_call_result_156)
        # Assigning a type to the variable 'row' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'row', list_162)
        
        
        
        # Call to len(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'row' (line 70)
        row_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'row', False)
        # Processing the call keyword arguments (line 70)
        kwargs_165 = {}
        # Getting the type of 'len' (line 70)
        len_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'len', False)
        # Calling len(args, kwargs) (line 70)
        len_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), len_163, *[row_164], **kwargs_165)
        
        # Getting the type of 'w' (line 70)
        w_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'w')
        # Applying the binary operator '!=' (line 70)
        result_ne_168 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '!=', len_call_result_166, w_167)
        
        # Testing the type of an if condition (line 70)
        if_condition_169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 16), result_ne_168)
        # Assigning a type to the variable 'if_condition_169' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'if_condition_169', if_condition_169)
        # SSA begins for if statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to MazeReaderException(...): (line 71)
        # Processing the call arguments (line 71)
        str_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'str', 'Invalid maze file - error in maze dimensions')
        # Processing the call keyword arguments (line 71)
        kwargs_172 = {}
        # Getting the type of 'MazeReaderException' (line 71)
        MazeReaderException_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'MazeReaderException', False)
        # Calling MazeReaderException(args, kwargs) (line 71)
        MazeReaderException_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), MazeReaderException_170, *[str_171], **kwargs_172)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 20), MazeReaderException_call_result_173, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 70)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'row' (line 73)
        row_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'row', False)
        # Processing the call keyword arguments (line 73)
        kwargs_178 = {}
        # Getting the type of 'self' (line 73)
        self_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'self', False)
        # Obtaining the member 'maze_rows' of a type (line 73)
        maze_rows_175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), self_174, 'maze_rows')
        # Obtaining the member 'append' of a type (line 73)
        append_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), maze_rows_175, 'append')
        # Calling append(args, kwargs) (line 73)
        append_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), append_176, *[row_177], **kwargs_178)
        
        # SSA join for if statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 62)
        # SSA branch for the except 'Tuple' branch of a try statement (line 62)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        # Getting the type of 'IOError' (line 74)
        IOError_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'IOError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 16), tuple_180, IOError_181)
        # Adding element type (line 74)
        # Getting the type of 'OSError' (line 74)
        OSError_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'OSError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 16), tuple_180, OSError_182)
        
        # Assigning a type to the variable 'e' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'e', tuple_180)
        
        # Call to MazeReaderException(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to str(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'e' (line 75)
        e_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 42), 'e', False)
        # Processing the call keyword arguments (line 75)
        kwargs_186 = {}
        # Getting the type of 'str' (line 75)
        str_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'str', False)
        # Calling str(args, kwargs) (line 75)
        str_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 75, 38), str_184, *[e_185], **kwargs_186)
        
        # Processing the call keyword arguments (line 75)
        kwargs_188 = {}
        # Getting the type of 'MazeReaderException' (line 75)
        MazeReaderException_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'MazeReaderException', False)
        # Calling MazeReaderException(args, kwargs) (line 75)
        MazeReaderException_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 75, 18), MazeReaderException_183, *[str_call_result_187], **kwargs_188)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 75, 12), MazeReaderException_call_result_189, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 62)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'readFile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'readFile' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'readFile'
        return stypy_return_type_190


    @norecursion
    def getData(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getData'
        module_type_store = module_type_store.open_function_context('getData', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeReader.getData.__dict__.__setitem__('stypy_localization', localization)
        MazeReader.getData.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeReader.getData.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeReader.getData.__dict__.__setitem__('stypy_function_name', 'MazeReader.getData')
        MazeReader.getData.__dict__.__setitem__('stypy_param_names_list', [])
        MazeReader.getData.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeReader.getData.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeReader.getData.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeReader.getData.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeReader.getData.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeReader.getData.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeReader.getData', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getData', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getData(...)' code ##################

        # Getting the type of 'self' (line 78)
        self_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'self')
        # Obtaining the member 'maze_rows' of a type (line 78)
        maze_rows_192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), self_191, 'maze_rows')
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', maze_rows_192)
        
        # ################# End of 'getData(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getData' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getData'
        return stypy_return_type_193


    @norecursion
    def readMaze(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'STDIN' (line 80)
        STDIN_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'STDIN')
        defaults = [STDIN_194]
        # Create a new context for function 'readMaze'
        module_type_store = module_type_store.open_function_context('readMaze', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeReader.readMaze.__dict__.__setitem__('stypy_localization', localization)
        MazeReader.readMaze.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeReader.readMaze.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeReader.readMaze.__dict__.__setitem__('stypy_function_name', 'MazeReader.readMaze')
        MazeReader.readMaze.__dict__.__setitem__('stypy_param_names_list', ['source'])
        MazeReader.readMaze.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeReader.readMaze.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeReader.readMaze.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeReader.readMaze.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeReader.readMaze.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeReader.readMaze.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeReader.readMaze', ['source'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'readMaze', localization, ['source'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'readMaze(...)' code ##################

        
        
        # Getting the type of 'source' (line 81)
        source_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'source')
        # Getting the type of 'STDIN' (line 81)
        STDIN_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'STDIN')
        # Applying the binary operator '==' (line 81)
        result_eq_197 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), '==', source_195, STDIN_196)
        
        # Testing the type of an if condition (line 81)
        if_condition_198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_eq_197)
        # Assigning a type to the variable 'if_condition_198' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_198', if_condition_198)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to readStdin(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_201 = {}
        # Getting the type of 'self' (line 82)
        self_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'self', False)
        # Obtaining the member 'readStdin' of a type (line 82)
        readStdin_200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), self_199, 'readStdin')
        # Calling readStdin(args, kwargs) (line 82)
        readStdin_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), readStdin_200, *[], **kwargs_201)
        
        # SSA branch for the else part of an if statement (line 81)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'source' (line 83)
        source_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'source')
        # Getting the type of 'FILE_' (line 83)
        FILE__204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'FILE_')
        # Applying the binary operator '==' (line 83)
        result_eq_205 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 13), '==', source_203, FILE__204)
        
        # Testing the type of an if condition (line 83)
        if_condition_206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 13), result_eq_205)
        # Assigning a type to the variable 'if_condition_206' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'if_condition_206', if_condition_206)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to readFile(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_209 = {}
        # Getting the type of 'self' (line 84)
        self_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self', False)
        # Obtaining the member 'readFile' of a type (line 84)
        readFile_208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_207, 'readFile')
        # Calling readFile(args, kwargs) (line 84)
        readFile_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), readFile_208, *[], **kwargs_209)
        
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to getData(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_213 = {}
        # Getting the type of 'self' (line 86)
        self_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self', False)
        # Obtaining the member 'getData' of a type (line 86)
        getData_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_211, 'getData')
        # Calling getData(args, kwargs) (line 86)
        getData_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), getData_212, *[], **kwargs_213)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', getData_call_result_214)
        
        # ################# End of 'readMaze(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'readMaze' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'readMaze'
        return stypy_return_type_215


# Assigning a type to the variable 'MazeReader' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'MazeReader', MazeReader)
# Declaration of the 'MazeFactory' class

class MazeFactory(object, ):

    @norecursion
    def makeMaze(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'STDIN' (line 89)
        STDIN_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'STDIN')
        defaults = [STDIN_216]
        # Create a new context for function 'makeMaze'
        module_type_store = module_type_store.open_function_context('makeMaze', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_localization', localization)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_function_name', 'MazeFactory.makeMaze')
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_param_names_list', ['source'])
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeFactory.makeMaze.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeFactory.makeMaze', ['source'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'makeMaze', localization, ['source'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'makeMaze(...)' code ##################

        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to MazeReader(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_218 = {}
        # Getting the type of 'MazeReader' (line 90)
        MazeReader_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'MazeReader', False)
        # Calling MazeReader(args, kwargs) (line 90)
        MazeReader_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 90, 17), MazeReader_217, *[], **kwargs_218)
        
        # Assigning a type to the variable 'reader' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'reader', MazeReader_call_result_219)
        
        # Call to Maze(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to readMaze(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'source' (line 91)
        source_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 36), 'source', False)
        # Processing the call keyword arguments (line 91)
        kwargs_224 = {}
        # Getting the type of 'reader' (line 91)
        reader_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'reader', False)
        # Obtaining the member 'readMaze' of a type (line 91)
        readMaze_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), reader_221, 'readMaze')
        # Calling readMaze(args, kwargs) (line 91)
        readMaze_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), readMaze_222, *[source_223], **kwargs_224)
        
        # Processing the call keyword arguments (line 91)
        kwargs_226 = {}
        # Getting the type of 'Maze' (line 91)
        Maze_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'Maze', False)
        # Calling Maze(args, kwargs) (line 91)
        Maze_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), Maze_220, *[readMaze_call_result_225], **kwargs_226)
        
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', Maze_call_result_227)
        
        # ################# End of 'makeMaze(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'makeMaze' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'makeMaze'
        return stypy_return_type_228


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 88, 0, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeFactory.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MazeFactory' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'MazeFactory', MazeFactory)
# Declaration of the 'MazeError' class
# Getting the type of 'Exception' (line 93)
Exception_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'Exception')

class MazeError(Exception_229, ):
    pass

# Assigning a type to the variable 'MazeError' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'MazeError', MazeError)
# Declaration of the 'Maze' class

class Maze(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_230, list_231)
        
        defaults = [list_230]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.__init__', ['rows'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['rows'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 98):
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'rows' (line 98)
        rows_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'rows')
        # Getting the type of 'self' (line 98)
        self_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member '_rows' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_233, '_rows', rows_232)
        
        # Call to __validate(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_236 = {}
        # Getting the type of 'self' (line 99)
        self_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self', False)
        # Obtaining the member '__validate' of a type (line 99)
        validate_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_234, '__validate')
        # Calling __validate(args, kwargs) (line 99)
        validate_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), validate_235, *[], **kwargs_236)
        
        
        # Call to __normalize(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_240 = {}
        # Getting the type of 'self' (line 100)
        self_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self', False)
        # Obtaining the member '__normalize' of a type (line 100)
        normalize_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_238, '__normalize')
        # Calling __normalize(args, kwargs) (line 100)
        normalize_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), normalize_239, *[], **kwargs_240)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Maze.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Maze.stypy__str__')
        Maze.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Maze.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 103):
        
        # Assigning a Str to a Name (line 103):
        str_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 12), 'str', '\n')
        # Assigning a type to the variable 's' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 's', str_242)
        
        # Getting the type of 'self' (line 104)
        self_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'self')
        # Obtaining the member '_rows' of a type (line 104)
        _rows_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 19), self_243, '_rows')
        # Testing the type of a for loop iterable (line 104)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 8), _rows_244)
        # Getting the type of the for loop variable (line 104)
        for_loop_var_245 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 8), _rows_244)
        # Assigning a type to the variable 'row' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'row', for_loop_var_245)
        # SSA begins for a for statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'row' (line 105)
        row_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'row')
        # Testing the type of a for loop iterable (line 105)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 12), row_246)
        # Getting the type of the for loop variable (line 105)
        for_loop_var_247 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 12), row_246)
        # Assigning a type to the variable 'item' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'item', for_loop_var_247)
        # SSA begins for a for statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'item' (line 106)
        item_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'item')
        # Getting the type of 'PATH' (line 106)
        PATH_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'PATH')
        # Applying the binary operator '==' (line 106)
        result_eq_250 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 19), '==', item_248, PATH_249)
        
        # Testing the type of an if condition (line 106)
        if_condition_251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 16), result_eq_250)
        # Assigning a type to the variable 'if_condition_251' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'if_condition_251', if_condition_251)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 106):
        
        # Assigning a Str to a Name (line 106):
        str_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 41), 'str', '*')
        # Assigning a type to the variable 'sitem' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'sitem', str_252)
        # SSA branch for the else part of an if statement (line 106)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'item' (line 107)
        item_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'item')
        # Getting the type of 'START' (line 107)
        START_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'START')
        # Applying the binary operator '==' (line 107)
        result_eq_255 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 21), '==', item_253, START_254)
        
        # Testing the type of an if condition (line 107)
        if_condition_256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 21), result_eq_255)
        # Assigning a type to the variable 'if_condition_256' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'if_condition_256', if_condition_256)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 107):
        
        # Assigning a Str to a Name (line 107):
        str_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 44), 'str', 'S')
        # Assigning a type to the variable 'sitem' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'sitem', str_257)
        # SSA branch for the else part of an if statement (line 107)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'item' (line 108)
        item_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'item')
        # Getting the type of 'EXIT' (line 108)
        EXIT_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'EXIT')
        # Applying the binary operator '==' (line 108)
        result_eq_260 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 21), '==', item_258, EXIT_259)
        
        # Testing the type of an if condition (line 108)
        if_condition_261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 21), result_eq_260)
        # Assigning a type to the variable 'if_condition_261' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'if_condition_261', if_condition_261)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 108):
        
        # Assigning a Str to a Name (line 108):
        str_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 43), 'str', 'E')
        # Assigning a type to the variable 'sitem' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'sitem', str_262)
        # SSA branch for the else part of an if statement (line 108)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to str(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'item' (line 109)
        item_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'item', False)
        # Processing the call keyword arguments (line 109)
        kwargs_265 = {}
        # Getting the type of 'str' (line 109)
        str_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'str', False)
        # Calling str(args, kwargs) (line 109)
        str_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 109, 30), str_263, *[item_264], **kwargs_265)
        
        # Assigning a type to the variable 'sitem' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'sitem', str_call_result_266)
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to join(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'tuple' (line 111)
        tuple_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 's' (line 111)
        s_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), tuple_269, s_270)
        # Adding element type (line 111)
        str_271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'str', '  ')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), tuple_269, str_271)
        # Adding element type (line 111)
        # Getting the type of 'sitem' (line 111)
        sitem_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'sitem', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), tuple_269, sitem_272)
        # Adding element type (line 111)
        str_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 42), 'str', '   ')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), tuple_269, str_273)
        
        # Processing the call keyword arguments (line 111)
        kwargs_274 = {}
        str_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 20), 'str', '')
        # Obtaining the member 'join' of a type (line 111)
        join_268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 20), str_267, 'join')
        # Calling join(args, kwargs) (line 111)
        join_call_result_275 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), join_268, *[tuple_269], **kwargs_274)
        
        # Assigning a type to the variable 's' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 's', join_call_result_275)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to join(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        # Getting the type of 's' (line 112)
        s_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 25), tuple_278, s_279)
        # Adding element type (line 112)
        str_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'str', '\n\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 25), tuple_278, str_280)
        
        # Processing the call keyword arguments (line 112)
        kwargs_281 = {}
        str_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 16), 'str', '')
        # Obtaining the member 'join' of a type (line 112)
        join_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), str_276, 'join')
        # Calling join(args, kwargs) (line 112)
        join_call_result_282 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), join_277, *[tuple_278], **kwargs_281)
        
        # Assigning a type to the variable 's' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 's', join_call_result_282)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 's' (line 114)
        s_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type', s_283)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_284


    @norecursion
    def __validate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__validate'
        module_type_store = module_type_store.open_function_context('__validate', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.__validate.__dict__.__setitem__('stypy_localization', localization)
        Maze.__validate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.__validate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.__validate.__dict__.__setitem__('stypy_function_name', 'Maze.__validate')
        Maze.__validate.__dict__.__setitem__('stypy_param_names_list', [])
        Maze.__validate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.__validate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.__validate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.__validate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.__validate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.__validate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.__validate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__validate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__validate(...)' code ##################

        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to len(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining the type of the subscript
        int_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'int')
        # Getting the type of 'self' (line 117)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'self', False)
        # Obtaining the member '_rows' of a type (line 117)
        _rows_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 20), self_287, '_rows')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 20), _rows_288, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 117, 20), getitem___289, int_286)
        
        # Processing the call keyword arguments (line 117)
        kwargs_291 = {}
        # Getting the type of 'len' (line 117)
        len_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'len', False)
        # Calling len(args, kwargs) (line 117)
        len_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), len_285, *[subscript_call_result_290], **kwargs_291)
        
        # Assigning a type to the variable 'width' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'width', len_call_result_292)
        
        # Assigning a ListComp to a Name (line 118):
        
        # Assigning a ListComp to a Name (line 118):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 118)
        self_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'self')
        # Obtaining the member '_rows' of a type (line 118)
        _rows_298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 38), self_297, '_rows')
        comprehension_299 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), _rows_298)
        # Assigning a type to the variable 'row' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'row', comprehension_299)
        
        # Call to len(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'row' (line 118)
        row_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'row', False)
        # Processing the call keyword arguments (line 118)
        kwargs_295 = {}
        # Getting the type of 'len' (line 118)
        len_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'len', False)
        # Calling len(args, kwargs) (line 118)
        len_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), len_293, *[row_294], **kwargs_295)
        
        list_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), list_300, len_call_result_296)
        # Assigning a type to the variable 'widths' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'widths', list_300)
        
        
        
        # Call to count(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'width' (line 119)
        width_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'width', False)
        # Processing the call keyword arguments (line 119)
        kwargs_304 = {}
        # Getting the type of 'widths' (line 119)
        widths_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'widths', False)
        # Obtaining the member 'count' of a type (line 119)
        count_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 11), widths_301, 'count')
        # Calling count(args, kwargs) (line 119)
        count_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), count_302, *[width_303], **kwargs_304)
        
        
        # Call to len(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'widths' (line 119)
        widths_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'widths', False)
        # Processing the call keyword arguments (line 119)
        kwargs_308 = {}
        # Getting the type of 'len' (line 119)
        len_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 'len', False)
        # Calling len(args, kwargs) (line 119)
        len_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 119, 34), len_306, *[widths_307], **kwargs_308)
        
        # Applying the binary operator '!=' (line 119)
        result_ne_310 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '!=', count_call_result_305, len_call_result_309)
        
        # Testing the type of an if condition (line 119)
        if_condition_311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_ne_310)
        # Assigning a type to the variable 'if_condition_311' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_311', if_condition_311)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to MazeError(...): (line 120)
        # Processing the call arguments (line 120)
        str_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 28), 'str', 'Invalid maze!')
        # Processing the call keyword arguments (line 120)
        kwargs_314 = {}
        # Getting the type of 'MazeError' (line 120)
        MazeError_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'MazeError', False)
        # Calling MazeError(args, kwargs) (line 120)
        MazeError_call_result_315 = invoke(stypy.reporting.localization.Localization(__file__, 120, 18), MazeError_312, *[str_313], **kwargs_314)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 120, 12), MazeError_call_result_315, 'raise parameter', BaseException)
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 122):
        
        # Assigning a Call to a Attribute (line 122):
        
        # Call to len(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'self' (line 122)
        self_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'self', False)
        # Obtaining the member '_rows' of a type (line 122)
        _rows_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 27), self_317, '_rows')
        # Processing the call keyword arguments (line 122)
        kwargs_319 = {}
        # Getting the type of 'len' (line 122)
        len_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'len', False)
        # Calling len(args, kwargs) (line 122)
        len_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 122, 23), len_316, *[_rows_318], **kwargs_319)
        
        # Getting the type of 'self' (line 122)
        self_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member '_height' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_321, '_height', len_call_result_320)
        
        # Assigning a Name to a Attribute (line 123):
        
        # Assigning a Name to a Attribute (line 123):
        # Getting the type of 'width' (line 123)
        width_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'width')
        # Getting the type of 'self' (line 123)
        self_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member '_width' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_323, '_width', width_322)
        
        # ################# End of '__validate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__validate' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__validate'
        return stypy_return_type_324


    @norecursion
    def __normalize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__normalize'
        module_type_store = module_type_store.open_function_context('__normalize', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.__normalize.__dict__.__setitem__('stypy_localization', localization)
        Maze.__normalize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.__normalize.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.__normalize.__dict__.__setitem__('stypy_function_name', 'Maze.__normalize')
        Maze.__normalize.__dict__.__setitem__('stypy_param_names_list', [])
        Maze.__normalize.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.__normalize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.__normalize.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.__normalize.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.__normalize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.__normalize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.__normalize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__normalize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__normalize(...)' code ##################

        
        
        # Call to range(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to len(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'self' (line 126)
        self_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'self', False)
        # Obtaining the member '_rows' of a type (line 126)
        _rows_328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 27), self_327, '_rows')
        # Processing the call keyword arguments (line 126)
        kwargs_329 = {}
        # Getting the type of 'len' (line 126)
        len_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'len', False)
        # Calling len(args, kwargs) (line 126)
        len_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), len_326, *[_rows_328], **kwargs_329)
        
        # Processing the call keyword arguments (line 126)
        kwargs_331 = {}
        # Getting the type of 'range' (line 126)
        range_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'range', False)
        # Calling range(args, kwargs) (line 126)
        range_call_result_332 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), range_325, *[len_call_result_330], **kwargs_331)
        
        # Testing the type of a for loop iterable (line 126)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 8), range_call_result_332)
        # Getting the type of the for loop variable (line 126)
        for_loop_var_333 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 8), range_call_result_332)
        # Assigning a type to the variable 'x' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', for_loop_var_333)
        # SSA begins for a for statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 127):
        
        # Assigning a Subscript to a Name (line 127):
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 127)
        x_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'x')
        # Getting the type of 'self' (line 127)
        self_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'self')
        # Obtaining the member '_rows' of a type (line 127)
        _rows_336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 18), self_335, '_rows')
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 18), _rows_336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 127, 18), getitem___337, x_334)
        
        # Assigning a type to the variable 'row' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'row', subscript_call_result_338)
        
        # Assigning a ListComp to a Name (line 128):
        
        # Assigning a ListComp to a Name (line 128):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'row' (line 128)
        row_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 43), 'row')
        comprehension_348 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 19), row_347)
        # Assigning a type to the variable 'y' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'y', comprehension_348)
        
        # Call to min(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to int(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'y' (line 128)
        y_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'y', False)
        # Processing the call keyword arguments (line 128)
        kwargs_342 = {}
        # Getting the type of 'int' (line 128)
        int_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'int', False)
        # Calling int(args, kwargs) (line 128)
        int_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 128, 23), int_340, *[y_341], **kwargs_342)
        
        int_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'int')
        # Processing the call keyword arguments (line 128)
        kwargs_345 = {}
        # Getting the type of 'min' (line 128)
        min_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'min', False)
        # Calling min(args, kwargs) (line 128)
        min_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), min_339, *[int_call_result_343, int_344], **kwargs_345)
        
        list_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 19), list_349, min_call_result_346)
        # Assigning a type to the variable 'row' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'row', list_349)
        
        # Assigning a Name to a Subscript (line 129):
        
        # Assigning a Name to a Subscript (line 129):
        # Getting the type of 'row' (line 129)
        row_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'row')
        # Getting the type of 'self' (line 129)
        self_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'self')
        # Obtaining the member '_rows' of a type (line 129)
        _rows_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), self_351, '_rows')
        # Getting the type of 'x' (line 129)
        x_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'x')
        # Storing an element on a container (line 129)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 12), _rows_352, (x_353, row_350))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__normalize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__normalize' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__normalize'
        return stypy_return_type_354


    @norecursion
    def validatePoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'validatePoint'
        module_type_store = module_type_store.open_function_context('validatePoint', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.validatePoint.__dict__.__setitem__('stypy_localization', localization)
        Maze.validatePoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.validatePoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.validatePoint.__dict__.__setitem__('stypy_function_name', 'Maze.validatePoint')
        Maze.validatePoint.__dict__.__setitem__('stypy_param_names_list', ['pt'])
        Maze.validatePoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.validatePoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.validatePoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.validatePoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.validatePoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.validatePoint.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.validatePoint', ['pt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'validatePoint', localization, ['pt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'validatePoint(...)' code ##################

        
        # Assigning a Name to a Tuple (line 132):
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 8), 'int')
        # Getting the type of 'pt' (line 132)
        pt_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'pt')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), pt_356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), getitem___357, int_355)
        
        # Assigning a type to the variable 'tuple_var_assignment_6' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_6', subscript_call_result_358)
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        int_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 8), 'int')
        # Getting the type of 'pt' (line 132)
        pt_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'pt')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), pt_360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), getitem___361, int_359)
        
        # Assigning a type to the variable 'tuple_var_assignment_7' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_7', subscript_call_result_362)
        
        # Assigning a Name to a Name (line 132):
        # Getting the type of 'tuple_var_assignment_6' (line 132)
        tuple_var_assignment_6_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_6')
        # Assigning a type to the variable 'x' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'x', tuple_var_assignment_6_363)
        
        # Assigning a Name to a Name (line 132):
        # Getting the type of 'tuple_var_assignment_7' (line 132)
        tuple_var_assignment_7_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tuple_var_assignment_7')
        # Assigning a type to the variable 'y' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 10), 'y', tuple_var_assignment_7_364)
        
        # Assigning a Attribute to a Name (line 133):
        
        # Assigning a Attribute to a Name (line 133):
        # Getting the type of 'self' (line 133)
        self_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'self')
        # Obtaining the member '_width' of a type (line 133)
        _width_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), self_365, '_width')
        # Assigning a type to the variable 'w' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'w', _width_366)
        
        # Assigning a Attribute to a Name (line 134):
        
        # Assigning a Attribute to a Name (line 134):
        # Getting the type of 'self' (line 134)
        self_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'self')
        # Obtaining the member '_height' of a type (line 134)
        _height_368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), self_367, '_height')
        # Assigning a type to the variable 'h' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'h', _height_368)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 137)
        x_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'x')
        # Getting the type of 'w' (line 137)
        w_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'w')
        int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 19), 'int')
        # Applying the binary operator '-' (line 137)
        result_sub_372 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 15), '-', w_370, int_371)
        
        # Applying the binary operator '>' (line 137)
        result_gt_373 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), '>', x_369, result_sub_372)
        
        
        # Getting the type of 'x' (line 137)
        x_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'x')
        int_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'int')
        # Applying the binary operator '<' (line 137)
        result_lt_376 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 24), '<', x_374, int_375)
        
        # Applying the binary operator 'or' (line 137)
        result_or_keyword_377 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), 'or', result_gt_373, result_lt_376)
        
        # Testing the type of an if condition (line 137)
        if_condition_378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_or_keyword_377)
        # Assigning a type to the variable 'if_condition_378' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_378', if_condition_378)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to MazeError(...): (line 138)
        # Processing the call arguments (line 138)
        str_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 28), 'str', 'x co-ordinate out of range!')
        # Processing the call keyword arguments (line 138)
        kwargs_381 = {}
        # Getting the type of 'MazeError' (line 138)
        MazeError_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'MazeError', False)
        # Calling MazeError(args, kwargs) (line 138)
        MazeError_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 138, 18), MazeError_379, *[str_380], **kwargs_381)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 138, 12), MazeError_call_result_382, 'raise parameter', BaseException)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'y' (line 140)
        y_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'y')
        # Getting the type of 'h' (line 140)
        h_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'h')
        int_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 19), 'int')
        # Applying the binary operator '-' (line 140)
        result_sub_386 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 15), '-', h_384, int_385)
        
        # Applying the binary operator '>' (line 140)
        result_gt_387 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '>', y_383, result_sub_386)
        
        
        # Getting the type of 'y' (line 140)
        y_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'y')
        int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'int')
        # Applying the binary operator '<' (line 140)
        result_lt_390 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 24), '<', y_388, int_389)
        
        # Applying the binary operator 'or' (line 140)
        result_or_keyword_391 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), 'or', result_gt_387, result_lt_390)
        
        # Testing the type of an if condition (line 140)
        if_condition_392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), result_or_keyword_391)
        # Assigning a type to the variable 'if_condition_392' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_392', if_condition_392)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to MazeError(...): (line 141)
        # Processing the call arguments (line 141)
        str_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'str', 'y co-ordinate out of range!')
        # Processing the call keyword arguments (line 141)
        kwargs_395 = {}
        # Getting the type of 'MazeError' (line 141)
        MazeError_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'MazeError', False)
        # Calling MazeError(args, kwargs) (line 141)
        MazeError_call_result_396 = invoke(stypy.reporting.localization.Localization(__file__, 141, 18), MazeError_393, *[str_394], **kwargs_395)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 141, 12), MazeError_call_result_396, 'raise parameter', BaseException)
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        pass
        
        # ################# End of 'validatePoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'validatePoint' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_397)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'validatePoint'
        return stypy_return_type_397


    @norecursion
    def getItem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getItem'
        module_type_store = module_type_store.open_function_context('getItem', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.getItem.__dict__.__setitem__('stypy_localization', localization)
        Maze.getItem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.getItem.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.getItem.__dict__.__setitem__('stypy_function_name', 'Maze.getItem')
        Maze.getItem.__dict__.__setitem__('stypy_param_names_list', ['x', 'y'])
        Maze.getItem.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.getItem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.getItem.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.getItem.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.getItem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.getItem.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.getItem', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getItem', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getItem(...)' code ##################

        
        # Call to validatePoint(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'x' (line 146)
        x_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 28), tuple_400, x_401)
        # Adding element type (line 146)
        # Getting the type of 'y' (line 146)
        y_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 28), tuple_400, y_402)
        
        # Processing the call keyword arguments (line 146)
        kwargs_403 = {}
        # Getting the type of 'self' (line 146)
        self_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member 'validatePoint' of a type (line 146)
        validatePoint_399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_398, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 146)
        validatePoint_call_result_404 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), validatePoint_399, *[tuple_400], **kwargs_403)
        
        
        # Assigning a Attribute to a Name (line 148):
        
        # Assigning a Attribute to a Name (line 148):
        # Getting the type of 'self' (line 148)
        self_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'self')
        # Obtaining the member '_width' of a type (line 148)
        _width_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), self_405, '_width')
        # Assigning a type to the variable 'w' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'w', _width_406)
        
        # Assigning a Attribute to a Name (line 149):
        
        # Assigning a Attribute to a Name (line 149):
        # Getting the type of 'self' (line 149)
        self_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self')
        # Obtaining the member '_height' of a type (line 149)
        _height_408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_407, '_height')
        # Assigning a type to the variable 'h' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'h', _height_408)
        
        # Assigning a Subscript to a Name (line 151):
        
        # Assigning a Subscript to a Name (line 151):
        
        # Obtaining the type of the subscript
        # Getting the type of 'h' (line 151)
        h_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'h')
        # Getting the type of 'y' (line 151)
        y_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'y')
        # Applying the binary operator '-' (line 151)
        result_sub_411 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 25), '-', h_409, y_410)
        
        int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'int')
        # Applying the binary operator '-' (line 151)
        result_sub_413 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 28), '-', result_sub_411, int_412)
        
        # Getting the type of 'self' (line 151)
        self_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'self')
        # Obtaining the member '_rows' of a type (line 151)
        _rows_415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 14), self_414, '_rows')
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 14), _rows_415, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), getitem___416, result_sub_413)
        
        # Assigning a type to the variable 'row' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'row', subscript_call_result_417)
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 152)
        x_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'x')
        # Getting the type of 'row' (line 152)
        row_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'row')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 15), row_419, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), getitem___420, x_418)
        
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', subscript_call_result_421)
        
        # ################# End of 'getItem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getItem' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getItem'
        return stypy_return_type_422


    @norecursion
    def setItem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setItem'
        module_type_store = module_type_store.open_function_context('setItem', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.setItem.__dict__.__setitem__('stypy_localization', localization)
        Maze.setItem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.setItem.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.setItem.__dict__.__setitem__('stypy_function_name', 'Maze.setItem')
        Maze.setItem.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'value'])
        Maze.setItem.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.setItem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.setItem.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.setItem.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.setItem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.setItem.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.setItem', ['x', 'y', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setItem', localization, ['x', 'y', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setItem(...)' code ##################

        
        # Assigning a Attribute to a Name (line 155):
        
        # Assigning a Attribute to a Name (line 155):
        # Getting the type of 'self' (line 155)
        self_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self')
        # Obtaining the member '_height' of a type (line 155)
        _height_424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_423, '_height')
        # Assigning a type to the variable 'h' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'h', _height_424)
        
        # Call to validatePoint(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Obtaining an instance of the builtin type 'tuple' (line 157)
        tuple_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 157)
        # Adding element type (line 157)
        # Getting the type of 'x' (line 157)
        x_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), tuple_427, x_428)
        # Adding element type (line 157)
        # Getting the type of 'y' (line 157)
        y_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), tuple_427, y_429)
        
        # Processing the call keyword arguments (line 157)
        kwargs_430 = {}
        # Getting the type of 'self' (line 157)
        self_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self', False)
        # Obtaining the member 'validatePoint' of a type (line 157)
        validatePoint_426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_425, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 157)
        validatePoint_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), validatePoint_426, *[tuple_427], **kwargs_430)
        
        
        # Assigning a Subscript to a Name (line 158):
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        # Getting the type of 'h' (line 158)
        h_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'h')
        # Getting the type of 'y' (line 158)
        y_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'y')
        # Applying the binary operator '-' (line 158)
        result_sub_434 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 25), '-', h_432, y_433)
        
        int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 29), 'int')
        # Applying the binary operator '-' (line 158)
        result_sub_436 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 28), '-', result_sub_434, int_435)
        
        # Getting the type of 'self' (line 158)
        self_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 14), 'self')
        # Obtaining the member '_rows' of a type (line 158)
        _rows_438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 14), self_437, '_rows')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 14), _rows_438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_440 = invoke(stypy.reporting.localization.Localization(__file__, 158, 14), getitem___439, result_sub_436)
        
        # Assigning a type to the variable 'row' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'row', subscript_call_result_440)
        
        # Assigning a Name to a Subscript (line 159):
        
        # Assigning a Name to a Subscript (line 159):
        # Getting the type of 'value' (line 159)
        value_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'value')
        # Getting the type of 'row' (line 159)
        row_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'row')
        # Getting the type of 'x' (line 159)
        x_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'x')
        # Storing an element on a container (line 159)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), row_442, (x_443, value_441))
        
        # ################# End of 'setItem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setItem' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_444)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setItem'
        return stypy_return_type_444


    @norecursion
    def getNeighBours(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getNeighBours'
        module_type_store = module_type_store.open_function_context('getNeighBours', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.getNeighBours.__dict__.__setitem__('stypy_localization', localization)
        Maze.getNeighBours.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.getNeighBours.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.getNeighBours.__dict__.__setitem__('stypy_function_name', 'Maze.getNeighBours')
        Maze.getNeighBours.__dict__.__setitem__('stypy_param_names_list', ['pt'])
        Maze.getNeighBours.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.getNeighBours.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.getNeighBours.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.getNeighBours.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.getNeighBours.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.getNeighBours.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.getNeighBours', ['pt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getNeighBours', localization, ['pt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getNeighBours(...)' code ##################

        
        # Call to validatePoint(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'pt' (line 162)
        pt_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'pt', False)
        # Processing the call keyword arguments (line 162)
        kwargs_448 = {}
        # Getting the type of 'self' (line 162)
        self_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self', False)
        # Obtaining the member 'validatePoint' of a type (line 162)
        validatePoint_446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_445, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 162)
        validatePoint_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), validatePoint_446, *[pt_447], **kwargs_448)
        
        
        # Assigning a Name to a Tuple (line 164):
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
        # Getting the type of 'pt' (line 164)
        pt_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'pt')
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), pt_451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___452, int_450)
        
        # Assigning a type to the variable 'tuple_var_assignment_8' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_8', subscript_call_result_453)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
        # Getting the type of 'pt' (line 164)
        pt_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'pt')
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), pt_455, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___456, int_454)
        
        # Assigning a type to the variable 'tuple_var_assignment_9' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_9', subscript_call_result_457)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_8' (line 164)
        tuple_var_assignment_8_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_8')
        # Assigning a type to the variable 'x' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'x', tuple_var_assignment_8_458)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_9' (line 164)
        tuple_var_assignment_9_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_9')
        # Assigning a type to the variable 'y' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 10), 'y', tuple_var_assignment_9_459)
        
        # Assigning a Attribute to a Name (line 166):
        
        # Assigning a Attribute to a Name (line 166):
        # Getting the type of 'self' (line 166)
        self_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self')
        # Obtaining the member '_height' of a type (line 166)
        _height_461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_460, '_height')
        # Assigning a type to the variable 'h' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'h', _height_461)
        
        # Assigning a Attribute to a Name (line 167):
        
        # Assigning a Attribute to a Name (line 167):
        # Getting the type of 'self' (line 167)
        self_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'self')
        # Obtaining the member '_width' of a type (line 167)
        _width_463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), self_462, '_width')
        # Assigning a type to the variable 'w' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'w', _width_463)
        
        # Assigning a Tuple to a Name (line 169):
        
        # Assigning a Tuple to a Name (line 169):
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'x')
        int_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'int')
        # Applying the binary operator '-' (line 169)
        result_sub_468 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 22), '-', x_466, int_467)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), tuple_465, result_sub_468)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), tuple_465, y_469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_465)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'x')
        int_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'int')
        # Applying the binary operator '-' (line 169)
        result_sub_473 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 30), '-', x_471, int_472)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 30), tuple_470, result_sub_473)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'y')
        int_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 36), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_476 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 34), '+', y_474, int_475)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 30), tuple_470, result_add_476)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_470)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 40), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 40), tuple_477, x_478)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 42), 'y')
        int_480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 44), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_481 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 42), '+', y_479, int_480)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 40), tuple_477, result_add_481)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_477)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 48), 'x')
        int_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 50), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_485 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 48), '+', x_483, int_484)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 48), tuple_482, result_add_485)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 52), 'y')
        int_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 54), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_488 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 52), '+', y_486, int_487)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 48), tuple_482, result_add_488)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_482)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 58), 'x')
        int_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 60), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_492 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 58), '+', x_490, int_491)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 58), tuple_489, result_add_492)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 62), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 58), tuple_489, y_493)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_489)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 66), 'x')
        int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 68), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_497 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 66), '+', x_495, int_496)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 66), tuple_494, result_add_497)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 70), 'y')
        int_499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 72), 'int')
        # Applying the binary operator '-' (line 169)
        result_sub_500 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 70), '-', y_498, int_499)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 66), tuple_494, result_sub_500)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_494)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 76), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 76), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 76), tuple_501, x_502)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 78), 'y')
        int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 80), 'int')
        # Applying the binary operator '-' (line 169)
        result_sub_505 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 78), '-', y_503, int_504)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 76), tuple_501, result_sub_505)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_501)
        # Adding element type (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 84), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'x' (line 169)
        x_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 84), 'x')
        int_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 86), 'int')
        # Applying the binary operator '-' (line 169)
        result_sub_509 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 84), '-', x_507, int_508)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 84), tuple_506, result_sub_509)
        # Adding element type (line 169)
        # Getting the type of 'y' (line 169)
        y_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 88), 'y')
        int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 90), 'int')
        # Applying the binary operator '-' (line 169)
        result_sub_512 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 88), '-', y_510, int_511)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 84), tuple_506, result_sub_512)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 21), tuple_464, tuple_506)
        
        # Assigning a type to the variable 'poss_nbors' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'poss_nbors', tuple_464)
        
        # Assigning a List to a Name (line 171):
        
        # Assigning a List to a Name (line 171):
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        
        # Assigning a type to the variable 'nbors' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'nbors', list_513)
        
        # Getting the type of 'poss_nbors' (line 172)
        poss_nbors_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'poss_nbors')
        # Testing the type of a for loop iterable (line 172)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 172, 8), poss_nbors_514)
        # Getting the type of the for loop variable (line 172)
        for_loop_var_515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 172, 8), poss_nbors_514)
        # Assigning a type to the variable 'xx' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'xx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 8), for_loop_var_515))
        # Assigning a type to the variable 'yy' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'yy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 8), for_loop_var_515))
        # SSA begins for a for statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'xx' (line 173)
        xx_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'xx')
        int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 20), 'int')
        # Applying the binary operator '>=' (line 173)
        result_ge_518 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '>=', xx_516, int_517)
        
        
        # Getting the type of 'xx' (line 173)
        xx_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 26), 'xx')
        # Getting the type of 'w' (line 173)
        w_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'w')
        int_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 32), 'int')
        # Applying the binary operator '-' (line 173)
        result_sub_522 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 30), '-', w_520, int_521)
        
        # Applying the binary operator '<=' (line 173)
        result_le_523 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 26), '<=', xx_519, result_sub_522)
        
        # Applying the binary operator 'and' (line 173)
        result_and_keyword_524 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), 'and', result_ge_518, result_le_523)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'yy' (line 173)
        yy_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 40), 'yy')
        int_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 44), 'int')
        # Applying the binary operator '>=' (line 173)
        result_ge_527 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 40), '>=', yy_525, int_526)
        
        
        # Getting the type of 'yy' (line 173)
        yy_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'yy')
        # Getting the type of 'h' (line 173)
        h_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 54), 'h')
        int_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 56), 'int')
        # Applying the binary operator '-' (line 173)
        result_sub_531 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 54), '-', h_529, int_530)
        
        # Applying the binary operator '<=' (line 173)
        result_le_532 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 50), '<=', yy_528, result_sub_531)
        
        # Applying the binary operator 'and' (line 173)
        result_and_keyword_533 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 40), 'and', result_ge_527, result_le_532)
        
        # Applying the binary operator 'and' (line 173)
        result_and_keyword_534 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), 'and', result_and_keyword_524, result_and_keyword_533)
        
        # Testing the type of an if condition (line 173)
        if_condition_535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), result_and_keyword_534)
        # Assigning a type to the variable 'if_condition_535' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_535', if_condition_535)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Obtaining an instance of the builtin type 'tuple' (line 174)
        tuple_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 174)
        # Adding element type (line 174)
        # Getting the type of 'xx' (line 174)
        xx_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'xx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 30), tuple_538, xx_539)
        # Adding element type (line 174)
        # Getting the type of 'yy' (line 174)
        yy_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 33), 'yy', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 30), tuple_538, yy_540)
        
        # Processing the call keyword arguments (line 174)
        kwargs_541 = {}
        # Getting the type of 'nbors' (line 174)
        nbors_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'nbors', False)
        # Obtaining the member 'append' of a type (line 174)
        append_537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), nbors_536, 'append')
        # Calling append(args, kwargs) (line 174)
        append_call_result_542 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), append_537, *[tuple_538], **kwargs_541)
        
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'nbors' (line 176)
        nbors_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'nbors')
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', nbors_543)
        
        # ################# End of 'getNeighBours(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getNeighBours' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getNeighBours'
        return stypy_return_type_544


    @norecursion
    def getExitPoints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getExitPoints'
        module_type_store = module_type_store.open_function_context('getExitPoints', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.getExitPoints.__dict__.__setitem__('stypy_localization', localization)
        Maze.getExitPoints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.getExitPoints.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.getExitPoints.__dict__.__setitem__('stypy_function_name', 'Maze.getExitPoints')
        Maze.getExitPoints.__dict__.__setitem__('stypy_param_names_list', ['pt'])
        Maze.getExitPoints.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.getExitPoints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.getExitPoints.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.getExitPoints.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.getExitPoints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.getExitPoints.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.getExitPoints', ['pt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getExitPoints', localization, ['pt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getExitPoints(...)' code ##################

        
        # Assigning a List to a Name (line 179):
        
        # Assigning a List to a Name (line 179):
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        
        # Assigning a type to the variable 'exits' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'exits', list_545)
        
        
        # Call to getNeighBours(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'pt' (line 180)
        pt_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 40), 'pt', False)
        # Processing the call keyword arguments (line 180)
        kwargs_549 = {}
        # Getting the type of 'self' (line 180)
        self_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'self', False)
        # Obtaining the member 'getNeighBours' of a type (line 180)
        getNeighBours_547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 21), self_546, 'getNeighBours')
        # Calling getNeighBours(args, kwargs) (line 180)
        getNeighBours_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 180, 21), getNeighBours_547, *[pt_548], **kwargs_549)
        
        # Testing the type of a for loop iterable (line 180)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 8), getNeighBours_call_result_550)
        # Getting the type of the for loop variable (line 180)
        for_loop_var_551 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 8), getNeighBours_call_result_550)
        # Assigning a type to the variable 'xx' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'xx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 8), for_loop_var_551))
        # Assigning a type to the variable 'yy' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'yy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 8), for_loop_var_551))
        # SSA begins for a for statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to getItem(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'xx' (line 181)
        xx_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'xx', False)
        # Getting the type of 'yy' (line 181)
        yy_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'yy', False)
        # Processing the call keyword arguments (line 181)
        kwargs_556 = {}
        # Getting the type of 'self' (line 181)
        self_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'self', False)
        # Obtaining the member 'getItem' of a type (line 181)
        getItem_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), self_552, 'getItem')
        # Calling getItem(args, kwargs) (line 181)
        getItem_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), getItem_553, *[xx_554, yy_555], **kwargs_556)
        
        int_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 36), 'int')
        # Applying the binary operator '==' (line 181)
        result_eq_559 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 15), '==', getItem_call_result_557, int_558)
        
        # Testing the type of an if condition (line 181)
        if_condition_560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 12), result_eq_559)
        # Assigning a type to the variable 'if_condition_560' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'if_condition_560', if_condition_560)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        # Getting the type of 'xx' (line 182)
        xx_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'xx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 30), tuple_563, xx_564)
        # Adding element type (line 182)
        # Getting the type of 'yy' (line 182)
        yy_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'yy', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 30), tuple_563, yy_565)
        
        # Processing the call keyword arguments (line 182)
        kwargs_566 = {}
        # Getting the type of 'exits' (line 182)
        exits_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'exits', False)
        # Obtaining the member 'append' of a type (line 182)
        append_562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 16), exits_561, 'append')
        # Calling append(args, kwargs) (line 182)
        append_call_result_567 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), append_562, *[tuple_563], **kwargs_566)
        
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'exits' (line 184)
        exits_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'exits')
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', exits_568)
        
        # ################# End of 'getExitPoints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getExitPoints' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getExitPoints'
        return stypy_return_type_569


    @norecursion
    def calcDistance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'calcDistance'
        module_type_store = module_type_store.open_function_context('calcDistance', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Maze.calcDistance.__dict__.__setitem__('stypy_localization', localization)
        Maze.calcDistance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Maze.calcDistance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Maze.calcDistance.__dict__.__setitem__('stypy_function_name', 'Maze.calcDistance')
        Maze.calcDistance.__dict__.__setitem__('stypy_param_names_list', ['pt1', 'pt2'])
        Maze.calcDistance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Maze.calcDistance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Maze.calcDistance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Maze.calcDistance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Maze.calcDistance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Maze.calcDistance.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Maze.calcDistance', ['pt1', 'pt2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'calcDistance', localization, ['pt1', 'pt2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'calcDistance(...)' code ##################

        
        # Call to validatePoint(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'pt1' (line 187)
        pt1_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 27), 'pt1', False)
        # Processing the call keyword arguments (line 187)
        kwargs_573 = {}
        # Getting the type of 'self' (line 187)
        self_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self', False)
        # Obtaining the member 'validatePoint' of a type (line 187)
        validatePoint_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_570, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 187)
        validatePoint_call_result_574 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), validatePoint_571, *[pt1_572], **kwargs_573)
        
        
        # Call to validatePoint(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'pt2' (line 188)
        pt2_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'pt2', False)
        # Processing the call keyword arguments (line 188)
        kwargs_578 = {}
        # Getting the type of 'self' (line 188)
        self_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member 'validatePoint' of a type (line 188)
        validatePoint_576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_575, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 188)
        validatePoint_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), validatePoint_576, *[pt2_577], **kwargs_578)
        
        
        # Assigning a Name to a Tuple (line 190):
        
        # Assigning a Subscript to a Name (line 190):
        
        # Obtaining the type of the subscript
        int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
        # Getting the type of 'pt1' (line 190)
        pt1_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'pt1')
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), pt1_581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), getitem___582, int_580)
        
        # Assigning a type to the variable 'tuple_var_assignment_10' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'tuple_var_assignment_10', subscript_call_result_583)
        
        # Assigning a Subscript to a Name (line 190):
        
        # Obtaining the type of the subscript
        int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
        # Getting the type of 'pt1' (line 190)
        pt1_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'pt1')
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), pt1_585, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), getitem___586, int_584)
        
        # Assigning a type to the variable 'tuple_var_assignment_11' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'tuple_var_assignment_11', subscript_call_result_587)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'tuple_var_assignment_10' (line 190)
        tuple_var_assignment_10_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'tuple_var_assignment_10')
        # Assigning a type to the variable 'x1' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'x1', tuple_var_assignment_10_588)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'tuple_var_assignment_11' (line 190)
        tuple_var_assignment_11_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'tuple_var_assignment_11')
        # Assigning a type to the variable 'y1' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'y1', tuple_var_assignment_11_589)
        
        # Assigning a Name to a Tuple (line 191):
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'pt2' (line 191)
        pt2_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'pt2')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), pt2_591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___592, int_590)
        
        # Assigning a type to the variable 'tuple_var_assignment_12' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_12', subscript_call_result_593)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'pt2' (line 191)
        pt2_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'pt2')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), pt2_595, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___596, int_594)
        
        # Assigning a type to the variable 'tuple_var_assignment_13' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_13', subscript_call_result_597)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_12' (line 191)
        tuple_var_assignment_12_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_12')
        # Assigning a type to the variable 'x2' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'x2', tuple_var_assignment_12_598)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_13' (line 191)
        tuple_var_assignment_13_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_13')
        # Assigning a type to the variable 'y2' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'y2', tuple_var_assignment_13_599)
        
        # Call to pow(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Call to pow(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'x1' (line 193)
        x1_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'x1', False)
        # Getting the type of 'x2' (line 193)
        x2_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'x2', False)
        # Applying the binary operator '-' (line 193)
        result_sub_604 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 26), '-', x1_602, x2_603)
        
        int_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 34), 'int')
        # Processing the call keyword arguments (line 193)
        kwargs_606 = {}
        # Getting the type of 'pow' (line 193)
        pow_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 21), 'pow', False)
        # Calling pow(args, kwargs) (line 193)
        pow_call_result_607 = invoke(stypy.reporting.localization.Localization(__file__, 193, 21), pow_601, *[result_sub_604, int_605], **kwargs_606)
        
        
        # Call to pow(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'y1' (line 193)
        y1_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 44), 'y1', False)
        # Getting the type of 'y2' (line 193)
        y2_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 47), 'y2', False)
        # Applying the binary operator '-' (line 193)
        result_sub_611 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 44), '-', y1_609, y2_610)
        
        int_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 51), 'int')
        # Processing the call keyword arguments (line 193)
        kwargs_613 = {}
        # Getting the type of 'pow' (line 193)
        pow_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 39), 'pow', False)
        # Calling pow(args, kwargs) (line 193)
        pow_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 193, 39), pow_608, *[result_sub_611, int_612], **kwargs_613)
        
        # Applying the binary operator '+' (line 193)
        result_add_615 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 21), '+', pow_call_result_607, pow_call_result_614)
        
        float_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 56), 'float')
        # Processing the call keyword arguments (line 193)
        kwargs_617 = {}
        # Getting the type of 'pow' (line 193)
        pow_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'pow', False)
        # Calling pow(args, kwargs) (line 193)
        pow_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 193, 15), pow_600, *[result_add_615, float_616], **kwargs_617)
        
        # Assigning a type to the variable 'stypy_return_type' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', pow_call_result_618)
        
        # ################# End of 'calcDistance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'calcDistance' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_619)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'calcDistance'
        return stypy_return_type_619


# Assigning a type to the variable 'Maze' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'Maze', Maze)
# Declaration of the 'MazeSolver' class

class MazeSolver(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.__init__', ['maze'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['maze'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 197):
        
        # Assigning a Name to a Attribute (line 197):
        # Getting the type of 'maze' (line 197)
        maze_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'maze')
        # Getting the type of 'self' (line 197)
        self_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self')
        # Setting the type of the member 'maze' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_621, 'maze', maze_620)
        
        # Assigning a Tuple to a Attribute (line 198):
        
        # Assigning a Tuple to a Attribute (line 198):
        
        # Obtaining an instance of the builtin type 'tuple' (line 198)
        tuple_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 198)
        # Adding element type (line 198)
        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 23), tuple_622, int_623)
        # Adding element type (line 198)
        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 23), tuple_622, int_624)
        
        # Getting the type of 'self' (line 198)
        self_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member '_start' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_625, '_start', tuple_622)
        
        # Assigning a Tuple to a Attribute (line 199):
        
        # Assigning a Tuple to a Attribute (line 199):
        
        # Obtaining an instance of the builtin type 'tuple' (line 199)
        tuple_626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 199)
        # Adding element type (line 199)
        int_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), tuple_626, int_627)
        # Adding element type (line 199)
        int_628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), tuple_626, int_628)
        
        # Getting the type of 'self' (line 199)
        self_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self')
        # Setting the type of the member '_end' of a type (line 199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_629, '_end', tuple_626)
        
        # Assigning a Tuple to a Attribute (line 200):
        
        # Assigning a Tuple to a Attribute (line 200):
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        int_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 25), tuple_630, int_631)
        # Adding element type (line 200)
        int_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 25), tuple_630, int_632)
        
        # Getting the type of 'self' (line 200)
        self_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member '_current' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_633, '_current', tuple_630)
        
        # Assigning a Num to a Attribute (line 201):
        
        # Assigning a Num to a Attribute (line 201):
        int_634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 22), 'int')
        # Getting the type of 'self' (line 201)
        self_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self')
        # Setting the type of the member '_steps' of a type (line 201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_635, '_steps', int_634)
        
        # Assigning a List to a Attribute (line 202):
        
        # Assigning a List to a Attribute (line 202):
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        
        # Getting the type of 'self' (line 202)
        self_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member '_path' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_637, '_path', list_636)
        
        # Assigning a Name to a Attribute (line 203):
        
        # Assigning a Name to a Attribute (line 203):
        # Getting the type of 'False' (line 203)
        False_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'False')
        # Getting the type of 'self' (line 203)
        self_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self')
        # Setting the type of the member '_tryalternate' of a type (line 203)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_639, '_tryalternate', False_638)
        
        # Assigning a Name to a Attribute (line 204):
        
        # Assigning a Name to a Attribute (line 204):
        # Getting the type of 'False' (line 204)
        False_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 28), 'False')
        # Getting the type of 'self' (line 204)
        self_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member '_trynextbest' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_641, '_trynextbest', False_640)
        
        # Assigning a Tuple to a Attribute (line 205):
        
        # Assigning a Tuple to a Attribute (line 205):
        
        # Obtaining an instance of the builtin type 'tuple' (line 205)
        tuple_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 205)
        # Adding element type (line 205)
        int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), tuple_642, int_643)
        # Adding element type (line 205)
        int_644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), tuple_642, int_644)
        
        # Getting the type of 'self' (line 205)
        self_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member '_disputed' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_645, '_disputed', tuple_642)
        
        # Assigning a Num to a Attribute (line 206):
        
        # Assigning a Num to a Attribute (line 206):
        int_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'int')
        # Getting the type of 'self' (line 206)
        self_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member '_loops' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_647, '_loops', int_646)
        
        # Assigning a Name to a Attribute (line 207):
        
        # Assigning a Name to a Attribute (line 207):
        # Getting the type of 'False' (line 207)
        False_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'False')
        # Getting the type of 'self' (line 207)
        self_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member '_retrace' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_649, '_retrace', False_648)
        
        # Assigning a Num to a Attribute (line 208):
        
        # Assigning a Num to a Attribute (line 208):
        int_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'int')
        # Getting the type of 'self' (line 208)
        self_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'self')
        # Setting the type of the member '_numretraces' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), self_651, '_numretraces', int_650)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setStartPoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setStartPoint'
        module_type_store = module_type_store.open_function_context('setStartPoint', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.setStartPoint')
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_param_names_list', ['pt'])
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.setStartPoint.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.setStartPoint', ['pt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setStartPoint', localization, ['pt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setStartPoint(...)' code ##################

        
        # Call to validatePoint(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'pt' (line 211)
        pt_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 32), 'pt', False)
        # Processing the call keyword arguments (line 211)
        kwargs_656 = {}
        # Getting the type of 'self' (line 211)
        self_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self', False)
        # Obtaining the member 'maze' of a type (line 211)
        maze_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_652, 'maze')
        # Obtaining the member 'validatePoint' of a type (line 211)
        validatePoint_654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), maze_653, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 211)
        validatePoint_call_result_657 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), validatePoint_654, *[pt_655], **kwargs_656)
        
        
        # Assigning a Name to a Attribute (line 212):
        
        # Assigning a Name to a Attribute (line 212):
        # Getting the type of 'pt' (line 212)
        pt_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'pt')
        # Getting the type of 'self' (line 212)
        self_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self')
        # Setting the type of the member '_start' of a type (line 212)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_659, '_start', pt_658)
        
        # ################# End of 'setStartPoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setStartPoint' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setStartPoint'
        return stypy_return_type_660


    @norecursion
    def setEndPoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setEndPoint'
        module_type_store = module_type_store.open_function_context('setEndPoint', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.setEndPoint')
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_param_names_list', ['pt'])
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.setEndPoint.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.setEndPoint', ['pt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setEndPoint', localization, ['pt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setEndPoint(...)' code ##################

        
        # Call to validatePoint(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'pt' (line 215)
        pt_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'pt', False)
        # Processing the call keyword arguments (line 215)
        kwargs_665 = {}
        # Getting the type of 'self' (line 215)
        self_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'self', False)
        # Obtaining the member 'maze' of a type (line 215)
        maze_662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), self_661, 'maze')
        # Obtaining the member 'validatePoint' of a type (line 215)
        validatePoint_663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), maze_662, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 215)
        validatePoint_call_result_666 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), validatePoint_663, *[pt_664], **kwargs_665)
        
        
        # Assigning a Name to a Attribute (line 216):
        
        # Assigning a Name to a Attribute (line 216):
        # Getting the type of 'pt' (line 216)
        pt_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'pt')
        # Getting the type of 'self' (line 216)
        self_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self')
        # Setting the type of the member '_end' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_668, '_end', pt_667)
        
        # ################# End of 'setEndPoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setEndPoint' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setEndPoint'
        return stypy_return_type_669


    @norecursion
    def boundaryCheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'boundaryCheck'
        module_type_store = module_type_store.open_function_context('boundaryCheck', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_function_name', 'MazeSolver.boundaryCheck')
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_param_names_list', [])
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.boundaryCheck.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.boundaryCheck', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'boundaryCheck', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'boundaryCheck(...)' code ##################

        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to getExitPoints(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'self' (line 219)
        self_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 41), 'self', False)
        # Obtaining the member '_start' of a type (line 219)
        _start_674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 41), self_673, '_start')
        # Processing the call keyword arguments (line 219)
        kwargs_675 = {}
        # Getting the type of 'self' (line 219)
        self_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'self', False)
        # Obtaining the member 'maze' of a type (line 219)
        maze_671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 17), self_670, 'maze')
        # Obtaining the member 'getExitPoints' of a type (line 219)
        getExitPoints_672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 17), maze_671, 'getExitPoints')
        # Calling getExitPoints(args, kwargs) (line 219)
        getExitPoints_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 219, 17), getExitPoints_672, *[_start_674], **kwargs_675)
        
        # Assigning a type to the variable 'exits1' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'exits1', getExitPoints_call_result_676)
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to getExitPoints(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 41), 'self', False)
        # Obtaining the member '_end' of a type (line 220)
        _end_681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 41), self_680, '_end')
        # Processing the call keyword arguments (line 220)
        kwargs_682 = {}
        # Getting the type of 'self' (line 220)
        self_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 'self', False)
        # Obtaining the member 'maze' of a type (line 220)
        maze_678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 17), self_677, 'maze')
        # Obtaining the member 'getExitPoints' of a type (line 220)
        getExitPoints_679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 17), maze_678, 'getExitPoints')
        # Calling getExitPoints(args, kwargs) (line 220)
        getExitPoints_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 220, 17), getExitPoints_679, *[_end_681], **kwargs_682)
        
        # Assigning a type to the variable 'exits2' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'exits2', getExitPoints_call_result_683)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'exits1' (line 222)
        exits1_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'exits1', False)
        # Processing the call keyword arguments (line 222)
        kwargs_686 = {}
        # Getting the type of 'len' (line 222)
        len_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'len', False)
        # Calling len(args, kwargs) (line 222)
        len_call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 222, 11), len_684, *[exits1_685], **kwargs_686)
        
        int_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 24), 'int')
        # Applying the binary operator '==' (line 222)
        result_eq_689 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), '==', len_call_result_687, int_688)
        
        
        
        # Call to len(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'exits2' (line 222)
        exits2_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'exits2', False)
        # Processing the call keyword arguments (line 222)
        kwargs_692 = {}
        # Getting the type of 'len' (line 222)
        len_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 29), 'len', False)
        # Calling len(args, kwargs) (line 222)
        len_call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 222, 29), len_690, *[exits2_691], **kwargs_692)
        
        int_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 42), 'int')
        # Applying the binary operator '==' (line 222)
        result_eq_695 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 29), '==', len_call_result_693, int_694)
        
        # Applying the binary operator 'or' (line 222)
        result_or_keyword_696 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), 'or', result_eq_689, result_eq_695)
        
        # Testing the type of an if condition (line 222)
        if_condition_697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), result_or_keyword_696)
        # Assigning a type to the variable 'if_condition_697' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_697', if_condition_697)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 223)
        False_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'stypy_return_type', False_698)
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 225)
        True_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'stypy_return_type', True_699)
        
        # ################# End of 'boundaryCheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'boundaryCheck' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_700)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'boundaryCheck'
        return stypy_return_type_700


    @norecursion
    def setCurrentPoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setCurrentPoint'
        module_type_store = module_type_store.open_function_context('setCurrentPoint', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.setCurrentPoint')
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_param_names_list', ['point'])
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.setCurrentPoint.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.setCurrentPoint', ['point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setCurrentPoint', localization, ['point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setCurrentPoint(...)' code ##################

        
        # Assigning a Name to a Attribute (line 228):
        
        # Assigning a Name to a Attribute (line 228):
        # Getting the type of 'point' (line 228)
        point_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'point')
        # Getting the type of 'self' (line 228)
        self_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member '_current' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_702, '_current', point_701)
        
        # Call to append(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'point' (line 229)
        point_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 26), 'point', False)
        # Processing the call keyword arguments (line 229)
        kwargs_707 = {}
        # Getting the type of 'self' (line 229)
        self_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self', False)
        # Obtaining the member '_path' of a type (line 229)
        _path_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_703, '_path')
        # Obtaining the member 'append' of a type (line 229)
        append_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), _path_704, 'append')
        # Calling append(args, kwargs) (line 229)
        append_call_result_708 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), append_705, *[point_706], **kwargs_707)
        
        
        # ################# End of 'setCurrentPoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setCurrentPoint' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setCurrentPoint'
        return stypy_return_type_709


    @norecursion
    def isSolved(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isSolved'
        module_type_store = module_type_store.open_function_context('isSolved', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.isSolved.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_function_name', 'MazeSolver.isSolved')
        MazeSolver.isSolved.__dict__.__setitem__('stypy_param_names_list', [])
        MazeSolver.isSolved.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.isSolved.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.isSolved', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isSolved', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isSolved(...)' code ##################

        
        # Getting the type of 'self' (line 232)
        self_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'self')
        # Obtaining the member '_current' of a type (line 232)
        _current_711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 16), self_710, '_current')
        # Getting the type of 'self' (line 232)
        self_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'self')
        # Obtaining the member '_end' of a type (line 232)
        _end_713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 33), self_712, '_end')
        # Applying the binary operator '==' (line 232)
        result_eq_714 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 16), '==', _current_711, _end_713)
        
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', result_eq_714)
        
        # ################# End of 'isSolved(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isSolved' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isSolved'
        return stypy_return_type_715


    @norecursion
    def getNextPoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getNextPoint'
        module_type_store = module_type_store.open_function_context('getNextPoint', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.getNextPoint')
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_param_names_list', [])
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.getNextPoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.getNextPoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getNextPoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getNextPoint(...)' code ##################

        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to getExitPoints(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 41), 'self', False)
        # Obtaining the member '_current' of a type (line 235)
        _current_720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 41), self_719, '_current')
        # Processing the call keyword arguments (line 235)
        kwargs_721 = {}
        # Getting the type of 'self' (line 235)
        self_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'self', False)
        # Obtaining the member 'maze' of a type (line 235)
        maze_717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), self_716, 'maze')
        # Obtaining the member 'getExitPoints' of a type (line 235)
        getExitPoints_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), maze_717, 'getExitPoints')
        # Calling getExitPoints(args, kwargs) (line 235)
        getExitPoints_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 235, 17), getExitPoints_718, *[_current_720], **kwargs_721)
        
        # Assigning a type to the variable 'points' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'points', getExitPoints_call_result_722)
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to getBestPoint(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'points' (line 237)
        points_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 34), 'points', False)
        # Processing the call keyword arguments (line 237)
        kwargs_726 = {}
        # Getting the type of 'self' (line 237)
        self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'self', False)
        # Obtaining the member 'getBestPoint' of a type (line 237)
        getBestPoint_724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), self_723, 'getBestPoint')
        # Calling getBestPoint(args, kwargs) (line 237)
        getBestPoint_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 237, 16), getBestPoint_724, *[points_725], **kwargs_726)
        
        # Assigning a type to the variable 'point' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'point', getBestPoint_call_result_727)
        
        
        # Call to checkClosedLoop(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'point' (line 239)
        point_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'point', False)
        # Processing the call keyword arguments (line 239)
        kwargs_731 = {}
        # Getting the type of 'self' (line 239)
        self_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'self', False)
        # Obtaining the member 'checkClosedLoop' of a type (line 239)
        checkClosedLoop_729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 14), self_728, 'checkClosedLoop')
        # Calling checkClosedLoop(args, kwargs) (line 239)
        checkClosedLoop_call_result_732 = invoke(stypy.reporting.localization.Localization(__file__, 239, 14), checkClosedLoop_729, *[point_730], **kwargs_731)
        
        # Testing the type of an if condition (line 239)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 8), checkClosedLoop_call_result_732)
        # SSA begins for while statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Call to endlessLoop(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_735 = {}
        # Getting the type of 'self' (line 241)
        self_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'self', False)
        # Obtaining the member 'endlessLoop' of a type (line 241)
        endlessLoop_734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), self_733, 'endlessLoop')
        # Calling endlessLoop(args, kwargs) (line 241)
        endlessLoop_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 241, 15), endlessLoop_734, *[], **kwargs_735)
        
        # Testing the type of an if condition (line 241)
        if_condition_737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 12), endlessLoop_call_result_736)
        # Assigning a type to the variable 'if_condition_737' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'if_condition_737', if_condition_737)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 243):
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'None' (line 243)
        None_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'None')
        # Assigning a type to the variable 'point' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'point', None_738)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 246):
        
        # Assigning a Name to a Name (line 246):
        # Getting the type of 'point' (line 246)
        point_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'point')
        # Assigning a type to the variable 'point2' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'point2', point_739)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'point' (line 247)
        point_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'point')
        # Getting the type of 'self' (line 247)
        self_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'self')
        # Obtaining the member '_start' of a type (line 247)
        _start_742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 22), self_741, '_start')
        # Applying the binary operator '==' (line 247)
        result_eq_743 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 15), '==', point_740, _start_742)
        
        
        
        # Call to len(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'self' (line 247)
        self_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 42), 'self', False)
        # Obtaining the member '_path' of a type (line 247)
        _path_746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 42), self_745, '_path')
        # Processing the call keyword arguments (line 247)
        kwargs_747 = {}
        # Getting the type of 'len' (line 247)
        len_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 38), 'len', False)
        # Calling len(args, kwargs) (line 247)
        len_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 247, 38), len_744, *[_path_746], **kwargs_747)
        
        int_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 54), 'int')
        # Applying the binary operator '>' (line 247)
        result_gt_750 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 38), '>', len_call_result_748, int_749)
        
        # Applying the binary operator 'and' (line 247)
        result_and_keyword_751 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 15), 'and', result_eq_743, result_gt_750)
        
        # Testing the type of an if condition (line 247)
        if_condition_752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 12), result_and_keyword_751)
        # Assigning a type to the variable 'if_condition_752' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'if_condition_752', if_condition_752)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 248):
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'True' (line 248)
        True_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 37), 'True')
        # Getting the type of 'self' (line 248)
        self_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'self')
        # Setting the type of the member '_tryalternate' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), self_754, '_tryalternate', True_753)
        # SSA branch for the else part of an if statement (line 247)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to getNextClosestPointNotInPath(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'points' (line 251)
        points_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 58), 'points', False)
        # Getting the type of 'point2' (line 251)
        point2_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 66), 'point2', False)
        # Processing the call keyword arguments (line 251)
        kwargs_759 = {}
        # Getting the type of 'self' (line 251)
        self_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'self', False)
        # Obtaining the member 'getNextClosestPointNotInPath' of a type (line 251)
        getNextClosestPointNotInPath_756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 24), self_755, 'getNextClosestPointNotInPath')
        # Calling getNextClosestPointNotInPath(args, kwargs) (line 251)
        getNextClosestPointNotInPath_call_result_760 = invoke(stypy.reporting.localization.Localization(__file__, 251, 24), getNextClosestPointNotInPath_756, *[points_757, point2_758], **kwargs_759)
        
        # Assigning a type to the variable 'point' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'point', getNextClosestPointNotInPath_call_result_760)
        
        
        # Getting the type of 'point' (line 252)
        point_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'point')
        # Applying the 'not' unary operator (line 252)
        result_not__762 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 19), 'not', point_761)
        
        # Testing the type of an if condition (line 252)
        if_condition_763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 16), result_not__762)
        # Assigning a type to the variable 'if_condition_763' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'if_condition_763', if_condition_763)
        # SSA begins for if statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to retracePath(...): (line 253)
        # Processing the call keyword arguments (line 253)
        kwargs_766 = {}
        # Getting the type of 'self' (line 253)
        self_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'self', False)
        # Obtaining the member 'retracePath' of a type (line 253)
        retracePath_765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), self_764, 'retracePath')
        # Calling retracePath(args, kwargs) (line 253)
        retracePath_call_result_767 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), retracePath_765, *[], **kwargs_766)
        
        
        # Assigning a Name to a Attribute (line 254):
        
        # Assigning a Name to a Attribute (line 254):
        # Getting the type of 'True' (line 254)
        True_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 41), 'True')
        # Getting the type of 'self' (line 254)
        self_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'self')
        # Setting the type of the member '_tryalternate' of a type (line 254)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), self_769, '_tryalternate', True_768)
        
        # Assigning a Attribute to a Name (line 255):
        
        # Assigning a Attribute to a Name (line 255):
        # Getting the type of 'self' (line 255)
        self_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 28), 'self')
        # Obtaining the member '_start' of a type (line 255)
        _start_771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 28), self_770, '_start')
        # Assigning a type to the variable 'point' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'point', _start_771)
        # SSA join for if statement (line 252)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'point' (line 258)
        point_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'point')
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', point_772)
        
        # ################# End of 'getNextPoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getNextPoint' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_773)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getNextPoint'
        return stypy_return_type_773


    @norecursion
    def retracePath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'retracePath'
        module_type_store = module_type_store.open_function_context('retracePath', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.retracePath.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_function_name', 'MazeSolver.retracePath')
        MazeSolver.retracePath.__dict__.__setitem__('stypy_param_names_list', [])
        MazeSolver.retracePath.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.retracePath.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.retracePath', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'retracePath', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'retracePath(...)' code ##################

        
        # Assigning a Name to a Attribute (line 262):
        
        # Assigning a Name to a Attribute (line 262):
        # Getting the type of 'True' (line 262)
        True_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'True')
        # Getting the type of 'self' (line 262)
        self_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self')
        # Setting the type of the member '_retrace' of a type (line 262)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_775, '_retrace', True_774)
        
        # Assigning a Subscript to a Name (line 264):
        
        # Assigning a Subscript to a Name (line 264):
        
        # Obtaining the type of the subscript
        slice_776 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 16), None, None, None)
        # Getting the type of 'self' (line 264)
        self_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'self')
        # Obtaining the member '_path' of a type (line 264)
        _path_778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), self_777, '_path')
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), _path_778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_780 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), getitem___779, slice_776)
        
        # Assigning a type to the variable 'path2' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'path2', subscript_call_result_780)
        
        # Call to reverse(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_783 = {}
        # Getting the type of 'path2' (line 265)
        path2_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'path2', False)
        # Obtaining the member 'reverse' of a type (line 265)
        reverse_782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), path2_781, 'reverse')
        # Calling reverse(args, kwargs) (line 265)
        reverse_call_result_784 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), reverse_782, *[], **kwargs_783)
        
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to index(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'self' (line 267)
        self_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 26), 'self', False)
        # Obtaining the member '_start' of a type (line 267)
        _start_788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 26), self_787, '_start')
        # Processing the call keyword arguments (line 267)
        kwargs_789 = {}
        # Getting the type of 'path2' (line 267)
        path2_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 14), 'path2', False)
        # Obtaining the member 'index' of a type (line 267)
        index_786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 14), path2_785, 'index')
        # Calling index(args, kwargs) (line 267)
        index_call_result_790 = invoke(stypy.reporting.localization.Localization(__file__, 267, 14), index_786, *[_start_788], **kwargs_789)
        
        # Assigning a type to the variable 'idx' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'idx', index_call_result_790)
        
        # Getting the type of 'self' (line 268)
        self_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self')
        # Obtaining the member '_path' of a type (line 268)
        _path_792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_791, '_path')
        
        # Obtaining the type of the subscript
        int_793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 33), 'int')
        # Getting the type of 'idx' (line 268)
        idx_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'idx')
        int_795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'int')
        slice_796 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 22), int_793, idx_794, int_795)
        # Getting the type of 'self' (line 268)
        self_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'self')
        # Obtaining the member '_path' of a type (line 268)
        _path_798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 22), self_797, '_path')
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 22), _path_798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_800 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), getitem___799, slice_796)
        
        # Applying the binary operator '+=' (line 268)
        result_iadd_801 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 8), '+=', _path_792, subscript_call_result_800)
        # Getting the type of 'self' (line 268)
        self_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self')
        # Setting the type of the member '_path' of a type (line 268)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_802, '_path', result_iadd_801)
        
        
        # Getting the type of 'self' (line 269)
        self_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Obtaining the member '_numretraces' of a type (line 269)
        _numretraces_804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_803, '_numretraces')
        int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 29), 'int')
        # Applying the binary operator '+=' (line 269)
        result_iadd_806 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 8), '+=', _numretraces_804, int_805)
        # Getting the type of 'self' (line 269)
        self_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Setting the type of the member '_numretraces' of a type (line 269)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_807, '_numretraces', result_iadd_806)
        
        
        # ################# End of 'retracePath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'retracePath' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_808)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'retracePath'
        return stypy_return_type_808


    @norecursion
    def endlessLoop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'endlessLoop'
        module_type_store = module_type_store.open_function_context('endlessLoop', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_function_name', 'MazeSolver.endlessLoop')
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_param_names_list', [])
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.endlessLoop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.endlessLoop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'endlessLoop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'endlessLoop(...)' code ##################

        
        
        # Getting the type of 'self' (line 272)
        self_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'self')
        # Obtaining the member '_loops' of a type (line 272)
        _loops_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 11), self_809, '_loops')
        int_811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 23), 'int')
        # Applying the binary operator '>' (line 272)
        result_gt_812 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 11), '>', _loops_810, int_811)
        
        # Testing the type of an if condition (line 272)
        if_condition_813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 8), result_gt_812)
        # Assigning a type to the variable 'if_condition_813' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'if_condition_813', if_condition_813)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 274)
        True_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type', True_814)
        # SSA branch for the else part of an if statement (line 272)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 275)
        self_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'self')
        # Obtaining the member '_numretraces' of a type (line 275)
        _numretraces_816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 13), self_815, '_numretraces')
        int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 31), 'int')
        # Applying the binary operator '>' (line 275)
        result_gt_818 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 13), '>', _numretraces_816, int_817)
        
        # Testing the type of an if condition (line 275)
        if_condition_819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 13), result_gt_818)
        # Assigning a type to the variable 'if_condition_819' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'if_condition_819', if_condition_819)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 277)
        True_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'stypy_return_type', True_820)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 279)
        False_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'stypy_return_type', False_821)
        
        # ################# End of 'endlessLoop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'endlessLoop' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'endlessLoop'
        return stypy_return_type_822


    @norecursion
    def checkClosedLoop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'checkClosedLoop'
        module_type_store = module_type_store.open_function_context('checkClosedLoop', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_function_name', 'MazeSolver.checkClosedLoop')
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_param_names_list', ['point'])
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.checkClosedLoop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.checkClosedLoop', ['point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'checkClosedLoop', localization, ['point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'checkClosedLoop(...)' code ##################

        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to range(...): (line 282)
        # Processing the call arguments (line 282)
        int_824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'int')
        
        # Call to len(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 25), 'self', False)
        # Obtaining the member '_path' of a type (line 282)
        _path_827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 25), self_826, '_path')
        # Processing the call keyword arguments (line 282)
        kwargs_828 = {}
        # Getting the type of 'len' (line 282)
        len_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'len', False)
        # Calling len(args, kwargs) (line 282)
        len_call_result_829 = invoke(stypy.reporting.localization.Localization(__file__, 282, 21), len_825, *[_path_827], **kwargs_828)
        
        int_830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 37), 'int')
        # Applying the binary operator '-' (line 282)
        result_sub_831 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 21), '-', len_call_result_829, int_830)
        
        int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 40), 'int')
        # Processing the call keyword arguments (line 282)
        kwargs_833 = {}
        # Getting the type of 'range' (line 282)
        range_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'range', False)
        # Calling range(args, kwargs) (line 282)
        range_call_result_834 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), range_823, *[int_824, result_sub_831, int_832], **kwargs_833)
        
        # Assigning a type to the variable 'l' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'l', range_call_result_834)
        
        # Call to reverse(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_837 = {}
        # Getting the type of 'l' (line 283)
        l_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'l', False)
        # Obtaining the member 'reverse' of a type (line 283)
        reverse_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), l_835, 'reverse')
        # Calling reverse(args, kwargs) (line 283)
        reverse_call_result_838 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), reverse_836, *[], **kwargs_837)
        
        
        # Getting the type of 'l' (line 285)
        l_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 17), 'l')
        # Testing the type of a for loop iterable (line 285)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 285, 8), l_839)
        # Getting the type of the for loop variable (line 285)
        for_loop_var_840 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 285, 8), l_839)
        # Assigning a type to the variable 'x' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'x', for_loop_var_840)
        # SSA begins for a for statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 286)
        x_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'x')
        # Getting the type of 'self' (line 286)
        self_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'self')
        # Obtaining the member '_path' of a type (line 286)
        _path_843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), self_842, '_path')
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), _path_843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 286, 15), getitem___844, x_841)
        
        # Getting the type of 'point' (line 286)
        point_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 32), 'point')
        # Applying the binary operator '==' (line 286)
        result_eq_847 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 15), '==', subscript_call_result_845, point_846)
        
        # Testing the type of an if condition (line 286)
        if_condition_848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 12), result_eq_847)
        # Assigning a type to the variable 'if_condition_848' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'if_condition_848', if_condition_848)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 287)
        self_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'self')
        # Obtaining the member '_loops' of a type (line 287)
        _loops_850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), self_849, '_loops')
        int_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'int')
        # Applying the binary operator '+=' (line 287)
        result_iadd_852 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 16), '+=', _loops_850, int_851)
        # Getting the type of 'self' (line 287)
        self_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'self')
        # Setting the type of the member '_loops' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), self_853, '_loops', result_iadd_852)
        
        # Getting the type of 'True' (line 288)
        True_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'stypy_return_type', True_854)
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 290)
        False_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', False_855)
        
        # ################# End of 'checkClosedLoop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'checkClosedLoop' in the type store
        # Getting the type of 'stypy_return_type' (line 281)
        stypy_return_type_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'checkClosedLoop'
        return stypy_return_type_856


    @norecursion
    def getBestPoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getBestPoint'
        module_type_store = module_type_store.open_function_context('getBestPoint', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.getBestPoint')
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_param_names_list', ['points'])
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.getBestPoint.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.getBestPoint', ['points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getBestPoint', localization, ['points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getBestPoint(...)' code ##################

        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to getClosestPoint(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'points' (line 293)
        points_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 37), 'points', False)
        # Processing the call keyword arguments (line 293)
        kwargs_860 = {}
        # Getting the type of 'self' (line 293)
        self_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'self', False)
        # Obtaining the member 'getClosestPoint' of a type (line 293)
        getClosestPoint_858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 16), self_857, 'getClosestPoint')
        # Calling getClosestPoint(args, kwargs) (line 293)
        getClosestPoint_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 293, 16), getClosestPoint_858, *[points_859], **kwargs_860)
        
        # Assigning a type to the variable 'point' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'point', getClosestPoint_call_result_861)
        
        # Assigning a Name to a Name (line 294):
        
        # Assigning a Name to a Name (line 294):
        # Getting the type of 'point' (line 294)
        point_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'point')
        # Assigning a type to the variable 'point2' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'point2', point_862)
        
        # Assigning a Name to a Name (line 295):
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'point' (line 295)
        point_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'point')
        # Assigning a type to the variable 'altpoint' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'altpoint', point_863)
        
        
        # Getting the type of 'point2' (line 297)
        point2_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'point2')
        # Getting the type of 'self' (line 297)
        self_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'self')
        # Obtaining the member '_path' of a type (line 297)
        _path_866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 21), self_865, '_path')
        # Applying the binary operator 'in' (line 297)
        result_contains_867 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 11), 'in', point2_864, _path_866)
        
        # Testing the type of an if condition (line 297)
        if_condition_868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 8), result_contains_867)
        # Assigning a type to the variable 'if_condition_868' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'if_condition_868', if_condition_868)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 298):
        
        # Assigning a Call to a Name (line 298):
        
        # Call to getNextClosestPointNotInPath(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'points' (line 298)
        points_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 54), 'points', False)
        # Getting the type of 'point2' (line 298)
        point2_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 62), 'point2', False)
        # Processing the call keyword arguments (line 298)
        kwargs_873 = {}
        # Getting the type of 'self' (line 298)
        self_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'self', False)
        # Obtaining the member 'getNextClosestPointNotInPath' of a type (line 298)
        getNextClosestPointNotInPath_870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 20), self_869, 'getNextClosestPointNotInPath')
        # Calling getNextClosestPointNotInPath(args, kwargs) (line 298)
        getNextClosestPointNotInPath_call_result_874 = invoke(stypy.reporting.localization.Localization(__file__, 298, 20), getNextClosestPointNotInPath_870, *[points_871, point2_872], **kwargs_873)
        
        # Assigning a type to the variable 'point' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'point', getNextClosestPointNotInPath_call_result_874)
        
        
        # Getting the type of 'point' (line 299)
        point_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'point')
        # Applying the 'not' unary operator (line 299)
        result_not__876 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 15), 'not', point_875)
        
        # Testing the type of an if condition (line 299)
        if_condition_877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 12), result_not__876)
        # Assigning a type to the variable 'if_condition_877' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'if_condition_877', if_condition_877)
        # SSA begins for if statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 300):
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'point2' (line 300)
        point2_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'point2')
        # Assigning a type to the variable 'point' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'point', point2_878)
        # SSA join for if statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 302)
        self_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'self')
        # Obtaining the member '_tryalternate' of a type (line 302)
        _tryalternate_880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 11), self_879, '_tryalternate')
        # Testing the type of an if condition (line 302)
        if_condition_881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), _tryalternate_880)
        # Assigning a type to the variable 'if_condition_881' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'if_condition_881', if_condition_881)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 303):
        
        # Assigning a Call to a Name (line 303):
        
        # Call to getAlternatePoint(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'points' (line 303)
        points_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 43), 'points', False)
        # Getting the type of 'altpoint' (line 303)
        altpoint_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 51), 'altpoint', False)
        # Processing the call keyword arguments (line 303)
        kwargs_886 = {}
        # Getting the type of 'self' (line 303)
        self_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'self', False)
        # Obtaining the member 'getAlternatePoint' of a type (line 303)
        getAlternatePoint_883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), self_882, 'getAlternatePoint')
        # Calling getAlternatePoint(args, kwargs) (line 303)
        getAlternatePoint_call_result_887 = invoke(stypy.reporting.localization.Localization(__file__, 303, 20), getAlternatePoint_883, *[points_884, altpoint_885], **kwargs_886)
        
        # Assigning a type to the variable 'point' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'point', getAlternatePoint_call_result_887)
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 306):
        
        # Assigning a Name to a Attribute (line 306):
        # Getting the type of 'False' (line 306)
        False_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 28), 'False')
        # Getting the type of 'self' (line 306)
        self_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self')
        # Setting the type of the member '_trynextbest' of a type (line 306)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_889, '_trynextbest', False_888)
        
        # Assigning a Name to a Attribute (line 307):
        
        # Assigning a Name to a Attribute (line 307):
        # Getting the type of 'False' (line 307)
        False_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 29), 'False')
        # Getting the type of 'self' (line 307)
        self_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self')
        # Setting the type of the member '_tryalternate' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_891, '_tryalternate', False_890)
        
        # Assigning a Name to a Attribute (line 308):
        
        # Assigning a Name to a Attribute (line 308):
        # Getting the type of 'False' (line 308)
        False_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'False')
        # Getting the type of 'self' (line 308)
        self_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self')
        # Setting the type of the member '_retrace' of a type (line 308)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_893, '_retrace', False_892)
        # Getting the type of 'point' (line 310)
        point_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'point')
        # Assigning a type to the variable 'stypy_return_type' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type', point_894)
        
        # ################# End of 'getBestPoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getBestPoint' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_895)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getBestPoint'
        return stypy_return_type_895


    @norecursion
    def sortPoints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sortPoints'
        module_type_store = module_type_store.open_function_context('sortPoints', 312, 4, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_function_name', 'MazeSolver.sortPoints')
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_param_names_list', ['points'])
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.sortPoints.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.sortPoints', ['points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sortPoints', localization, ['points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sortPoints(...)' code ##################

        
        # Assigning a ListComp to a Name (line 313):
        
        # Assigning a ListComp to a Name (line 313):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'points' (line 313)
        points_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 75), 'points')
        comprehension_905 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 21), points_904)
        # Assigning a type to the variable 'point' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 'point', comprehension_905)
        
        # Call to calcDistance(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'point' (line 313)
        point_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 44), 'point', False)
        # Getting the type of 'self' (line 313)
        self_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 51), 'self', False)
        # Obtaining the member '_end' of a type (line 313)
        _end_901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 51), self_900, '_end')
        # Processing the call keyword arguments (line 313)
        kwargs_902 = {}
        # Getting the type of 'self' (line 313)
        self_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 'self', False)
        # Obtaining the member 'maze' of a type (line 313)
        maze_897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 21), self_896, 'maze')
        # Obtaining the member 'calcDistance' of a type (line 313)
        calcDistance_898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 21), maze_897, 'calcDistance')
        # Calling calcDistance(args, kwargs) (line 313)
        calcDistance_call_result_903 = invoke(stypy.reporting.localization.Localization(__file__, 313, 21), calcDistance_898, *[point_899, _end_901], **kwargs_902)
        
        list_906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 21), list_906, calcDistance_call_result_903)
        # Assigning a type to the variable 'distances' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'distances', list_906)
        
        # Assigning a Subscript to a Name (line 314):
        
        # Assigning a Subscript to a Name (line 314):
        
        # Obtaining the type of the subscript
        slice_907 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 21), None, None, None)
        # Getting the type of 'distances' (line 314)
        distances_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 21), 'distances')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 21), distances_908, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_910 = invoke(stypy.reporting.localization.Localization(__file__, 314, 21), getitem___909, slice_907)
        
        # Assigning a type to the variable 'distances2' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'distances2', subscript_call_result_910)
        
        # Call to sort(...): (line 316)
        # Processing the call keyword arguments (line 316)
        kwargs_913 = {}
        # Getting the type of 'distances' (line 316)
        distances_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'distances', False)
        # Obtaining the member 'sort' of a type (line 316)
        sort_912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), distances_911, 'sort')
        # Calling sort(args, kwargs) (line 316)
        sort_call_result_914 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), sort_912, *[], **kwargs_913)
        
        
        # Assigning a BinOp to a Name (line 318):
        
        # Assigning a BinOp to a Name (line 318):
        
        # Obtaining an instance of the builtin type 'list' (line 318)
        list_915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 318)
        # Adding element type (line 318)
        
        # Obtaining an instance of the builtin type 'tuple' (line 318)
        tuple_916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 318)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 18), list_915, tuple_916)
        
        
        # Call to len(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'points' (line 318)
        points_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'points', False)
        # Processing the call keyword arguments (line 318)
        kwargs_919 = {}
        # Getting the type of 'len' (line 318)
        len_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'len', False)
        # Calling len(args, kwargs) (line 318)
        len_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 318, 23), len_917, *[points_918], **kwargs_919)
        
        # Applying the binary operator '*' (line 318)
        result_mul_921 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 18), '*', list_915, len_call_result_920)
        
        # Assigning a type to the variable 'points2' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'points2', result_mul_921)
        
        # Assigning a Num to a Name (line 319):
        
        # Assigning a Num to a Name (line 319):
        int_922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 16), 'int')
        # Assigning a type to the variable 'count' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'count', int_922)
        
        # Getting the type of 'distances' (line 321)
        distances_923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 20), 'distances')
        # Testing the type of a for loop iterable (line 321)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 321, 8), distances_923)
        # Getting the type of the for loop variable (line 321)
        for_loop_var_924 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 321, 8), distances_923)
        # Assigning a type to the variable 'dist' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'dist', for_loop_var_924)
        # SSA begins for a for statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to index(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'dist' (line 322)
        dist_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 35), 'dist', False)
        # Processing the call keyword arguments (line 322)
        kwargs_928 = {}
        # Getting the type of 'distances2' (line 322)
        distances2_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 18), 'distances2', False)
        # Obtaining the member 'index' of a type (line 322)
        index_926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 18), distances2_925, 'index')
        # Calling index(args, kwargs) (line 322)
        index_call_result_929 = invoke(stypy.reporting.localization.Localization(__file__, 322, 18), index_926, *[dist_927], **kwargs_928)
        
        # Assigning a type to the variable 'idx' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'idx', index_call_result_929)
        
        # Assigning a Subscript to a Name (line 323):
        
        # Assigning a Subscript to a Name (line 323):
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 323)
        idx_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'idx')
        # Getting the type of 'points' (line 323)
        points_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'points')
        # Obtaining the member '__getitem__' of a type (line 323)
        getitem___932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 20), points_931, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 323)
        subscript_call_result_933 = invoke(stypy.reporting.localization.Localization(__file__, 323, 20), getitem___932, idx_930)
        
        # Assigning a type to the variable 'point' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'point', subscript_call_result_933)
        
        
        # Getting the type of 'point' (line 325)
        point_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 18), 'point')
        # Getting the type of 'points2' (line 325)
        points2_935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 27), 'points2')
        # Applying the binary operator 'in' (line 325)
        result_contains_936 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 18), 'in', point_934, points2_935)
        
        # Testing the type of an if condition (line 325)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 12), result_contains_936)
        # SSA begins for while statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to index(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'dist' (line 326)
        dist_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 39), 'dist', False)
        # Getting the type of 'idx' (line 326)
        idx_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 45), 'idx', False)
        int_941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 49), 'int')
        # Applying the binary operator '+' (line 326)
        result_add_942 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 45), '+', idx_940, int_941)
        
        # Processing the call keyword arguments (line 326)
        kwargs_943 = {}
        # Getting the type of 'distances2' (line 326)
        distances2_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'distances2', False)
        # Obtaining the member 'index' of a type (line 326)
        index_938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 22), distances2_937, 'index')
        # Calling index(args, kwargs) (line 326)
        index_call_result_944 = invoke(stypy.reporting.localization.Localization(__file__, 326, 22), index_938, *[dist_939, result_add_942], **kwargs_943)
        
        # Assigning a type to the variable 'idx' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'idx', index_call_result_944)
        
        # Assigning a Subscript to a Name (line 327):
        
        # Assigning a Subscript to a Name (line 327):
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 327)
        idx_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 31), 'idx')
        # Getting the type of 'points' (line 327)
        points_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'points')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 24), points_946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_948 = invoke(stypy.reporting.localization.Localization(__file__, 327, 24), getitem___947, idx_945)
        
        # Assigning a type to the variable 'point' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'point', subscript_call_result_948)
        # SSA join for while statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 329):
        
        # Assigning a Name to a Subscript (line 329):
        # Getting the type of 'point' (line 329)
        point_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 29), 'point')
        # Getting the type of 'points2' (line 329)
        points2_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'points2')
        # Getting the type of 'count' (line 329)
        count_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'count')
        # Storing an element on a container (line 329)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 12), points2_950, (count_951, point_949))
        
        # Getting the type of 'count' (line 330)
        count_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'count')
        int_953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 21), 'int')
        # Applying the binary operator '+=' (line 330)
        result_iadd_954 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 12), '+=', count_952, int_953)
        # Assigning a type to the variable 'count' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'count', result_iadd_954)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'points2' (line 332)
        points2_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'points2')
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', points2_955)
        
        # ################# End of 'sortPoints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sortPoints' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sortPoints'
        return stypy_return_type_956


    @norecursion
    def getClosestPoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getClosestPoint'
        module_type_store = module_type_store.open_function_context('getClosestPoint', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.getClosestPoint')
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_param_names_list', ['points'])
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.getClosestPoint.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.getClosestPoint', ['points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getClosestPoint', localization, ['points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getClosestPoint(...)' code ##################

        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to sortPoints(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'points' (line 335)
        points_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 34), 'points', False)
        # Processing the call keyword arguments (line 335)
        kwargs_960 = {}
        # Getting the type of 'self' (line 335)
        self_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 18), 'self', False)
        # Obtaining the member 'sortPoints' of a type (line 335)
        sortPoints_958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 18), self_957, 'sortPoints')
        # Calling sortPoints(args, kwargs) (line 335)
        sortPoints_call_result_961 = invoke(stypy.reporting.localization.Localization(__file__, 335, 18), sortPoints_958, *[points_959], **kwargs_960)
        
        # Assigning a type to the variable 'points2' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'points2', sortPoints_call_result_961)
        
        # Assigning a Subscript to a Name (line 337):
        
        # Assigning a Subscript to a Name (line 337):
        
        # Obtaining the type of the subscript
        int_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 26), 'int')
        # Getting the type of 'points2' (line 337)
        points2_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), 'points2')
        # Obtaining the member '__getitem__' of a type (line 337)
        getitem___964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 18), points2_963, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 337)
        subscript_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 337, 18), getitem___964, int_962)
        
        # Assigning a type to the variable 'closest' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'closest', subscript_call_result_965)
        # Getting the type of 'closest' (line 338)
        closest_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'closest')
        # Assigning a type to the variable 'stypy_return_type' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'stypy_return_type', closest_966)
        
        # ################# End of 'getClosestPoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getClosestPoint' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getClosestPoint'
        return stypy_return_type_967


    @norecursion
    def getAlternatePoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getAlternatePoint'
        module_type_store = module_type_store.open_function_context('getAlternatePoint', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.getAlternatePoint')
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_param_names_list', ['points', 'point'])
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.getAlternatePoint.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.getAlternatePoint', ['points', 'point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getAlternatePoint', localization, ['points', 'point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getAlternatePoint(...)' code ##################

        
        # Assigning a Subscript to a Name (line 341):
        
        # Assigning a Subscript to a Name (line 341):
        
        # Obtaining the type of the subscript
        slice_968 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 18), None, None, None)
        # Getting the type of 'points' (line 341)
        points_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'points')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 18), points_969, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), getitem___970, slice_968)
        
        # Assigning a type to the variable 'points2' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'points2', subscript_call_result_971)
        
        # Call to remove(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'point' (line 344)
        point_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 23), 'point', False)
        # Processing the call keyword arguments (line 344)
        kwargs_975 = {}
        # Getting the type of 'points2' (line 344)
        points2_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'points2', False)
        # Obtaining the member 'remove' of a type (line 344)
        remove_973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), points2_972, 'remove')
        # Calling remove(args, kwargs) (line 344)
        remove_call_result_976 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), remove_973, *[point_974], **kwargs_975)
        
        
        # Getting the type of 'points2' (line 345)
        points2_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'points2')
        # Testing the type of an if condition (line 345)
        if_condition_978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 345, 8), points2_977)
        # Assigning a type to the variable 'if_condition_978' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'if_condition_978', if_condition_978)
        # SSA begins for if statement (line 345)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to choice(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'points2' (line 346)
        points2_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 33), 'points2', False)
        # Processing the call keyword arguments (line 346)
        kwargs_982 = {}
        # Getting the type of 'random' (line 346)
        random_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 19), 'random', False)
        # Obtaining the member 'choice' of a type (line 346)
        choice_980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 19), random_979, 'choice')
        # Calling choice(args, kwargs) (line 346)
        choice_call_result_983 = invoke(stypy.reporting.localization.Localization(__file__, 346, 19), choice_980, *[points2_981], **kwargs_982)
        
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'stypy_return_type', choice_call_result_983)
        # SSA join for if statement (line 345)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 348)
        None_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'stypy_return_type', None_984)
        
        # ################# End of 'getAlternatePoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getAlternatePoint' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_985)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getAlternatePoint'
        return stypy_return_type_985


    @norecursion
    def getNextClosestPoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getNextClosestPoint'
        module_type_store = module_type_store.open_function_context('getNextClosestPoint', 350, 4, False)
        # Assigning a type to the variable 'self' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_function_name', 'MazeSolver.getNextClosestPoint')
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_param_names_list', ['points', 'point'])
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.getNextClosestPoint.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.getNextClosestPoint', ['points', 'point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getNextClosestPoint', localization, ['points', 'point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getNextClosestPoint(...)' code ##################

        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to sortPoints(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'points' (line 351)
        points_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 34), 'points', False)
        # Processing the call keyword arguments (line 351)
        kwargs_989 = {}
        # Getting the type of 'self' (line 351)
        self_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 18), 'self', False)
        # Obtaining the member 'sortPoints' of a type (line 351)
        sortPoints_987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 18), self_986, 'sortPoints')
        # Calling sortPoints(args, kwargs) (line 351)
        sortPoints_call_result_990 = invoke(stypy.reporting.localization.Localization(__file__, 351, 18), sortPoints_987, *[points_988], **kwargs_989)
        
        # Assigning a type to the variable 'points2' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'points2', sortPoints_call_result_990)
        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Call to index(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'point' (line 352)
        point_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 28), 'point', False)
        # Processing the call keyword arguments (line 352)
        kwargs_994 = {}
        # Getting the type of 'points2' (line 352)
        points2_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 14), 'points2', False)
        # Obtaining the member 'index' of a type (line 352)
        index_992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 14), points2_991, 'index')
        # Calling index(args, kwargs) (line 352)
        index_call_result_995 = invoke(stypy.reporting.localization.Localization(__file__, 352, 14), index_992, *[point_993], **kwargs_994)
        
        # Assigning a type to the variable 'idx' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'idx', index_call_result_995)
        
        
        # SSA begins for try-except statement (line 354)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 355)
        idx_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 27), 'idx')
        int_997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 31), 'int')
        # Applying the binary operator '+' (line 355)
        result_add_998 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 27), '+', idx_996, int_997)
        
        # Getting the type of 'points2' (line 355)
        points2_999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 'points2')
        # Obtaining the member '__getitem__' of a type (line 355)
        getitem___1000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 19), points2_999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 355)
        subscript_call_result_1001 = invoke(stypy.reporting.localization.Localization(__file__, 355, 19), getitem___1000, result_add_998)
        
        # Assigning a type to the variable 'stypy_return_type' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'stypy_return_type', subscript_call_result_1001)
        # SSA branch for the except part of a try statement (line 354)
        # SSA branch for the except '<any exception>' branch of a try statement (line 354)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 357)
        None_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', None_1002)
        # SSA join for try-except statement (line 354)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'getNextClosestPoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getNextClosestPoint' in the type store
        # Getting the type of 'stypy_return_type' (line 350)
        stypy_return_type_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1003)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getNextClosestPoint'
        return stypy_return_type_1003


    @norecursion
    def getNextClosestPointNotInPath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getNextClosestPointNotInPath'
        module_type_store = module_type_store.open_function_context('getNextClosestPointNotInPath', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_function_name', 'MazeSolver.getNextClosestPointNotInPath')
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_param_names_list', ['points', 'point'])
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.getNextClosestPointNotInPath.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.getNextClosestPointNotInPath', ['points', 'point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getNextClosestPointNotInPath', localization, ['points', 'point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getNextClosestPointNotInPath(...)' code ##################

        
        # Assigning a Call to a Name (line 362):
        
        # Assigning a Call to a Name (line 362):
        
        # Call to getNextClosestPoint(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'points' (line 362)
        points_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 42), 'points', False)
        # Getting the type of 'point' (line 362)
        point_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 50), 'point', False)
        # Processing the call keyword arguments (line 362)
        kwargs_1008 = {}
        # Getting the type of 'self' (line 362)
        self_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 17), 'self', False)
        # Obtaining the member 'getNextClosestPoint' of a type (line 362)
        getNextClosestPoint_1005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 17), self_1004, 'getNextClosestPoint')
        # Calling getNextClosestPoint(args, kwargs) (line 362)
        getNextClosestPoint_call_result_1009 = invoke(stypy.reporting.localization.Localization(__file__, 362, 17), getNextClosestPoint_1005, *[points_1006, point_1007], **kwargs_1008)
        
        # Assigning a type to the variable 'point2' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'point2', getNextClosestPoint_call_result_1009)
        
        
        # Getting the type of 'point2' (line 363)
        point2_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 14), 'point2')
        # Getting the type of 'self' (line 363)
        self_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 24), 'self')
        # Obtaining the member '_path' of a type (line 363)
        _path_1012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 24), self_1011, '_path')
        # Applying the binary operator 'in' (line 363)
        result_contains_1013 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 14), 'in', point2_1010, _path_1012)
        
        # Testing the type of an if condition (line 363)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 8), result_contains_1013)
        # SSA begins for while statement (line 363)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 364):
        
        # Assigning a Call to a Name (line 364):
        
        # Call to getNextClosestPoint(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'points' (line 364)
        points_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 46), 'points', False)
        # Getting the type of 'point2' (line 364)
        point2_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 54), 'point2', False)
        # Processing the call keyword arguments (line 364)
        kwargs_1018 = {}
        # Getting the type of 'self' (line 364)
        self_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 21), 'self', False)
        # Obtaining the member 'getNextClosestPoint' of a type (line 364)
        getNextClosestPoint_1015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 21), self_1014, 'getNextClosestPoint')
        # Calling getNextClosestPoint(args, kwargs) (line 364)
        getNextClosestPoint_call_result_1019 = invoke(stypy.reporting.localization.Localization(__file__, 364, 21), getNextClosestPoint_1015, *[points_1016, point2_1017], **kwargs_1018)
        
        # Assigning a type to the variable 'point2' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'point2', getNextClosestPoint_call_result_1019)
        # SSA join for while statement (line 363)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'point2' (line 366)
        point2_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'point2')
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'stypy_return_type', point2_1020)
        
        # ################# End of 'getNextClosestPointNotInPath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getNextClosestPointNotInPath' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getNextClosestPointNotInPath'
        return stypy_return_type_1021


    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.solve.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.solve.__dict__.__setitem__('stypy_function_name', 'MazeSolver.solve')
        MazeSolver.solve.__dict__.__setitem__('stypy_param_names_list', [])
        MazeSolver.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.solve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.solve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        
        
        # Getting the type of 'self' (line 373)
        self_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), 'self')
        # Obtaining the member '_start' of a type (line 373)
        _start_1023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 11), self_1022, '_start')
        # Getting the type of 'self' (line 373)
        self_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'self')
        # Obtaining the member '_end' of a type (line 373)
        _end_1025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 26), self_1024, '_end')
        # Applying the binary operator '==' (line 373)
        result_eq_1026 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 11), '==', _start_1023, _end_1025)
        
        # Testing the type of an if condition (line 373)
        if_condition_1027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 8), result_eq_1026)
        # Assigning a type to the variable 'if_condition_1027' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'if_condition_1027', if_condition_1027)
        # SSA begins for if statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 376)
        None_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', None_1028)
        # SSA join for if statement (line 373)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to boundaryCheck(...): (line 379)
        # Processing the call keyword arguments (line 379)
        kwargs_1031 = {}
        # Getting the type of 'self' (line 379)
        self_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'self', False)
        # Obtaining the member 'boundaryCheck' of a type (line 379)
        boundaryCheck_1030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), self_1029, 'boundaryCheck')
        # Calling boundaryCheck(args, kwargs) (line 379)
        boundaryCheck_call_result_1032 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), boundaryCheck_1030, *[], **kwargs_1031)
        
        # Applying the 'not' unary operator (line 379)
        result_not__1033 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 11), 'not', boundaryCheck_call_result_1032)
        
        # Testing the type of an if condition (line 379)
        if_condition_1034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), result_not__1033)
        # Assigning a type to the variable 'if_condition_1034' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'if_condition_1034', if_condition_1034)
        # SSA begins for if statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 381)
        None_1035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'stypy_return_type', None_1035)
        # SSA join for if statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to setCurrentPoint(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'self' (line 387)
        self_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 29), 'self', False)
        # Obtaining the member '_start' of a type (line 387)
        _start_1039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 29), self_1038, '_start')
        # Processing the call keyword arguments (line 387)
        kwargs_1040 = {}
        # Getting the type of 'self' (line 387)
        self_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'self', False)
        # Obtaining the member 'setCurrentPoint' of a type (line 387)
        setCurrentPoint_1037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), self_1036, 'setCurrentPoint')
        # Calling setCurrentPoint(args, kwargs) (line 387)
        setCurrentPoint_call_result_1041 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), setCurrentPoint_1037, *[_start_1039], **kwargs_1040)
        
        
        # Assigning a Name to a Name (line 389):
        
        # Assigning a Name to a Name (line 389):
        # Getting the type of 'False' (line 389)
        False_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), 'False')
        # Assigning a type to the variable 'unsolvable' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'unsolvable', False_1042)
        
        
        
        # Call to isSolved(...): (line 391)
        # Processing the call keyword arguments (line 391)
        kwargs_1045 = {}
        # Getting the type of 'self' (line 391)
        self_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 18), 'self', False)
        # Obtaining the member 'isSolved' of a type (line 391)
        isSolved_1044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 18), self_1043, 'isSolved')
        # Calling isSolved(args, kwargs) (line 391)
        isSolved_call_result_1046 = invoke(stypy.reporting.localization.Localization(__file__, 391, 18), isSolved_1044, *[], **kwargs_1045)
        
        # Applying the 'not' unary operator (line 391)
        result_not__1047 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 14), 'not', isSolved_call_result_1046)
        
        # Testing the type of an if condition (line 391)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 8), result_not__1047)
        # SSA begins for while statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'self' (line 392)
        self_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'self')
        # Obtaining the member '_steps' of a type (line 392)
        _steps_1049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), self_1048, '_steps')
        int_1050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 27), 'int')
        # Applying the binary operator '+=' (line 392)
        result_iadd_1051 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 12), '+=', _steps_1049, int_1050)
        # Getting the type of 'self' (line 392)
        self_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'self')
        # Setting the type of the member '_steps' of a type (line 392)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), self_1052, '_steps', result_iadd_1051)
        
        
        # Assigning a Call to a Name (line 393):
        
        # Assigning a Call to a Name (line 393):
        
        # Call to getNextPoint(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_1055 = {}
        # Getting the type of 'self' (line 393)
        self_1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 17), 'self', False)
        # Obtaining the member 'getNextPoint' of a type (line 393)
        getNextPoint_1054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 17), self_1053, 'getNextPoint')
        # Calling getNextPoint(args, kwargs) (line 393)
        getNextPoint_call_result_1056 = invoke(stypy.reporting.localization.Localization(__file__, 393, 17), getNextPoint_1054, *[], **kwargs_1055)
        
        # Assigning a type to the variable 'pt' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'pt', getNextPoint_call_result_1056)
        
        # Getting the type of 'pt' (line 395)
        pt_1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 15), 'pt')
        # Testing the type of an if condition (line 395)
        if_condition_1058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 12), pt_1057)
        # Assigning a type to the variable 'if_condition_1058' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'if_condition_1058', if_condition_1058)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setCurrentPoint(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'pt' (line 396)
        pt_1061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 37), 'pt', False)
        # Processing the call keyword arguments (line 396)
        kwargs_1062 = {}
        # Getting the type of 'self' (line 396)
        self_1059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 'self', False)
        # Obtaining the member 'setCurrentPoint' of a type (line 396)
        setCurrentPoint_1060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), self_1059, 'setCurrentPoint')
        # Calling setCurrentPoint(args, kwargs) (line 396)
        setCurrentPoint_call_result_1063 = invoke(stypy.reporting.localization.Localization(__file__, 396, 16), setCurrentPoint_1060, *[pt_1061], **kwargs_1062)
        
        # SSA branch for the else part of an if statement (line 395)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 399):
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'True' (line 399)
        True_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 29), 'True')
        # Assigning a type to the variable 'unsolvable' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'unsolvable', True_1064)
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 391)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'unsolvable' (line 402)
        unsolvable_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'unsolvable')
        # Applying the 'not' unary operator (line 402)
        result_not__1066 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 11), 'not', unsolvable_1065)
        
        # Testing the type of an if condition (line 402)
        if_condition_1067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), result_not__1066)
        # Assigning a type to the variable 'if_condition_1067' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_1067', if_condition_1067)
        # SSA begins for if statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 402)
        module_type_store.open_ssa_branch('else')
        pass
        # SSA join for if statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to printResult(...): (line 407)
        # Processing the call keyword arguments (line 407)
        kwargs_1070 = {}
        # Getting the type of 'self' (line 407)
        self_1068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'self', False)
        # Obtaining the member 'printResult' of a type (line 407)
        printResult_1069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 8), self_1068, 'printResult')
        # Calling printResult(args, kwargs) (line 407)
        printResult_call_result_1071 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), printResult_1069, *[], **kwargs_1070)
        
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1072)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_1072


    @norecursion
    def printResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'printResult'
        module_type_store = module_type_store.open_function_context('printResult', 409, 4, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeSolver.printResult.__dict__.__setitem__('stypy_localization', localization)
        MazeSolver.printResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeSolver.printResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeSolver.printResult.__dict__.__setitem__('stypy_function_name', 'MazeSolver.printResult')
        MazeSolver.printResult.__dict__.__setitem__('stypy_param_names_list', [])
        MazeSolver.printResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeSolver.printResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeSolver.printResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeSolver.printResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeSolver.printResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeSolver.printResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeSolver.printResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'printResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'printResult(...)' code ##################

        str_1073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 8), 'str', ' Print the maze showing the path ')
        
        # Getting the type of 'self' (line 412)
        self_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 19), 'self')
        # Obtaining the member '_path' of a type (line 412)
        _path_1075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 19), self_1074, '_path')
        # Testing the type of a for loop iterable (line 412)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 412, 8), _path_1075)
        # Getting the type of the for loop variable (line 412)
        for_loop_var_1076 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 412, 8), _path_1075)
        # Assigning a type to the variable 'x' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 8), for_loop_var_1076))
        # Assigning a type to the variable 'y' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 8), for_loop_var_1076))
        # SSA begins for a for statement (line 412)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setItem(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'x' (line 413)
        x_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 30), 'x', False)
        # Getting the type of 'y' (line 413)
        y_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 32), 'y', False)
        # Getting the type of 'PATH' (line 413)
        PATH_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'PATH', False)
        # Processing the call keyword arguments (line 413)
        kwargs_1083 = {}
        # Getting the type of 'self' (line 413)
        self_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'self', False)
        # Obtaining the member 'maze' of a type (line 413)
        maze_1078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), self_1077, 'maze')
        # Obtaining the member 'setItem' of a type (line 413)
        setItem_1079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), maze_1078, 'setItem')
        # Calling setItem(args, kwargs) (line 413)
        setItem_call_result_1084 = invoke(stypy.reporting.localization.Localization(__file__, 413, 12), setItem_1079, *[x_1080, y_1081, PATH_1082], **kwargs_1083)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to setItem(...): (line 415)
        # Processing the call arguments (line 415)
        
        # Obtaining the type of the subscript
        int_1088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 38), 'int')
        # Getting the type of 'self' (line 415)
        self_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 26), 'self', False)
        # Obtaining the member '_start' of a type (line 415)
        _start_1090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 26), self_1089, '_start')
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___1091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 26), _start_1090, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_1092 = invoke(stypy.reporting.localization.Localization(__file__, 415, 26), getitem___1091, int_1088)
        
        
        # Obtaining the type of the subscript
        int_1093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 54), 'int')
        # Getting the type of 'self' (line 415)
        self_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 42), 'self', False)
        # Obtaining the member '_start' of a type (line 415)
        _start_1095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 42), self_1094, '_start')
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___1096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 42), _start_1095, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_1097 = invoke(stypy.reporting.localization.Localization(__file__, 415, 42), getitem___1096, int_1093)
        
        # Getting the type of 'START' (line 415)
        START_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 58), 'START', False)
        # Processing the call keyword arguments (line 415)
        kwargs_1099 = {}
        # Getting the type of 'self' (line 415)
        self_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'self', False)
        # Obtaining the member 'maze' of a type (line 415)
        maze_1086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), self_1085, 'maze')
        # Obtaining the member 'setItem' of a type (line 415)
        setItem_1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), maze_1086, 'setItem')
        # Calling setItem(args, kwargs) (line 415)
        setItem_call_result_1100 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), setItem_1087, *[subscript_call_result_1092, subscript_call_result_1097, START_1098], **kwargs_1099)
        
        
        # Call to setItem(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Obtaining the type of the subscript
        int_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 36), 'int')
        # Getting the type of 'self' (line 416)
        self_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 26), 'self', False)
        # Obtaining the member '_end' of a type (line 416)
        _end_1106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 26), self_1105, '_end')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___1107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 26), _end_1106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_1108 = invoke(stypy.reporting.localization.Localization(__file__, 416, 26), getitem___1107, int_1104)
        
        
        # Obtaining the type of the subscript
        int_1109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 50), 'int')
        # Getting the type of 'self' (line 416)
        self_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 40), 'self', False)
        # Obtaining the member '_end' of a type (line 416)
        _end_1111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 40), self_1110, '_end')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___1112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 40), _end_1111, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 416, 40), getitem___1112, int_1109)
        
        # Getting the type of 'EXIT' (line 416)
        EXIT_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 54), 'EXIT', False)
        # Processing the call keyword arguments (line 416)
        kwargs_1115 = {}
        # Getting the type of 'self' (line 416)
        self_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'self', False)
        # Obtaining the member 'maze' of a type (line 416)
        maze_1102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), self_1101, 'maze')
        # Obtaining the member 'setItem' of a type (line 416)
        setItem_1103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), maze_1102, 'setItem')
        # Calling setItem(args, kwargs) (line 416)
        setItem_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), setItem_1103, *[subscript_call_result_1108, subscript_call_result_1113, EXIT_1114], **kwargs_1115)
        
        
        # ################# End of 'printResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'printResult' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'printResult'
        return stypy_return_type_1117


# Assigning a type to the variable 'MazeSolver' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'MazeSolver', MazeSolver)
# Declaration of the 'MazeGame' class

class MazeGame(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 423, 4, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeGame.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Tuple to a Attribute (line 424):
        
        # Assigning a Tuple to a Attribute (line 424):
        
        # Obtaining an instance of the builtin type 'tuple' (line 424)
        tuple_1118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 424)
        # Adding element type (line 424)
        int_1119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 23), tuple_1118, int_1119)
        # Adding element type (line 424)
        int_1120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 23), tuple_1118, int_1120)
        
        # Getting the type of 'self' (line 424)
        self_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'self')
        # Setting the type of the member '_start' of a type (line 424)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), self_1121, '_start', tuple_1118)
        
        # Assigning a Tuple to a Attribute (line 425):
        
        # Assigning a Tuple to a Attribute (line 425):
        
        # Obtaining an instance of the builtin type 'tuple' (line 425)
        tuple_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 425)
        # Adding element type (line 425)
        int_1123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 21), tuple_1122, int_1123)
        # Adding element type (line 425)
        int_1124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 21), tuple_1122, int_1124)
        
        # Getting the type of 'self' (line 425)
        self_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'self')
        # Setting the type of the member '_end' of a type (line 425)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 8), self_1125, '_end', tuple_1122)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def runGame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runGame'
        module_type_store = module_type_store.open_function_context('runGame', 433, 4, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MazeGame.runGame.__dict__.__setitem__('stypy_localization', localization)
        MazeGame.runGame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MazeGame.runGame.__dict__.__setitem__('stypy_type_store', module_type_store)
        MazeGame.runGame.__dict__.__setitem__('stypy_function_name', 'MazeGame.runGame')
        MazeGame.runGame.__dict__.__setitem__('stypy_param_names_list', [])
        MazeGame.runGame.__dict__.__setitem__('stypy_varargs_param_name', None)
        MazeGame.runGame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MazeGame.runGame.__dict__.__setitem__('stypy_call_defaults', defaults)
        MazeGame.runGame.__dict__.__setitem__('stypy_call_varargs', varargs)
        MazeGame.runGame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MazeGame.runGame.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MazeGame.runGame', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runGame', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runGame(...)' code ##################

        
        # Assigning a Call to a Name (line 434):
        
        # Assigning a Call to a Name (line 434):
        
        # Call to createMaze(...): (line 434)
        # Processing the call keyword arguments (line 434)
        kwargs_1128 = {}
        # Getting the type of 'self' (line 434)
        self_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'self', False)
        # Obtaining the member 'createMaze' of a type (line 434)
        createMaze_1127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), self_1126, 'createMaze')
        # Calling createMaze(args, kwargs) (line 434)
        createMaze_call_result_1129 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), createMaze_1127, *[], **kwargs_1128)
        
        # Assigning a type to the variable 'maze' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'maze', createMaze_call_result_1129)
        
        
        # Getting the type of 'maze' (line 435)
        maze_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'maze')
        # Applying the 'not' unary operator (line 435)
        result_not__1131 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 11), 'not', maze_1130)
        
        # Testing the type of an if condition (line 435)
        if_condition_1132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 8), result_not__1131)
        # Assigning a type to the variable 'if_condition_1132' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'if_condition_1132', if_condition_1132)
        # SSA begins for if statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 436)
        None_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'stypy_return_type', None_1133)
        # SSA join for if statement (line 435)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to getStartEndPoints(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'maze' (line 439)
        maze_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 31), 'maze', False)
        # Processing the call keyword arguments (line 439)
        kwargs_1137 = {}
        # Getting the type of 'self' (line 439)
        self_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self', False)
        # Obtaining the member 'getStartEndPoints' of a type (line 439)
        getStartEndPoints_1135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_1134, 'getStartEndPoints')
        # Calling getStartEndPoints(args, kwargs) (line 439)
        getStartEndPoints_call_result_1138 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), getStartEndPoints_1135, *[maze_1136], **kwargs_1137)
        
        
        # Assigning a Call to a Name (line 443):
        
        # Assigning a Call to a Name (line 443):
        
        # Call to MazeSolver(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'maze' (line 443)
        maze_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 28), 'maze', False)
        # Processing the call keyword arguments (line 443)
        kwargs_1141 = {}
        # Getting the type of 'MazeSolver' (line 443)
        MazeSolver_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 17), 'MazeSolver', False)
        # Calling MazeSolver(args, kwargs) (line 443)
        MazeSolver_call_result_1142 = invoke(stypy.reporting.localization.Localization(__file__, 443, 17), MazeSolver_1139, *[maze_1140], **kwargs_1141)
        
        # Assigning a type to the variable 'solver' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'solver', MazeSolver_call_result_1142)
        
        # Call to setStartPoint(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'self' (line 446)
        self_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 29), 'self', False)
        # Obtaining the member '_start' of a type (line 446)
        _start_1146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 29), self_1145, '_start')
        # Processing the call keyword arguments (line 446)
        kwargs_1147 = {}
        # Getting the type of 'solver' (line 446)
        solver_1143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'solver', False)
        # Obtaining the member 'setStartPoint' of a type (line 446)
        setStartPoint_1144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), solver_1143, 'setStartPoint')
        # Calling setStartPoint(args, kwargs) (line 446)
        setStartPoint_call_result_1148 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), setStartPoint_1144, *[_start_1146], **kwargs_1147)
        
        
        # Call to setEndPoint(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'self' (line 447)
        self_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 27), 'self', False)
        # Obtaining the member '_end' of a type (line 447)
        _end_1152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 27), self_1151, '_end')
        # Processing the call keyword arguments (line 447)
        kwargs_1153 = {}
        # Getting the type of 'solver' (line 447)
        solver_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'solver', False)
        # Obtaining the member 'setEndPoint' of a type (line 447)
        setEndPoint_1150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), solver_1149, 'setEndPoint')
        # Calling setEndPoint(args, kwargs) (line 447)
        setEndPoint_call_result_1154 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), setEndPoint_1150, *[_end_1152], **kwargs_1153)
        
        
        # Call to solve(...): (line 448)
        # Processing the call keyword arguments (line 448)
        kwargs_1157 = {}
        # Getting the type of 'solver' (line 448)
        solver_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'solver', False)
        # Obtaining the member 'solve' of a type (line 448)
        solve_1156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), solver_1155, 'solve')
        # Calling solve(args, kwargs) (line 448)
        solve_call_result_1158 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), solve_1156, *[], **kwargs_1157)
        
        
        # ################# End of 'runGame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runGame' in the type store
        # Getting the type of 'stypy_return_type' (line 433)
        stypy_return_type_1159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runGame'
        return stypy_return_type_1159


# Assigning a type to the variable 'MazeGame' (line 422)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 0), 'MazeGame', MazeGame)
# Declaration of the 'FilebasedMazeGame' class
# Getting the type of 'MazeGame' (line 450)
MazeGame_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'MazeGame')

class FilebasedMazeGame(MazeGame_1160, ):

    @norecursion
    def createMaze(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'createMaze'
        module_type_store = module_type_store.open_function_context('createMaze', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_localization', localization)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_type_store', module_type_store)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_function_name', 'FilebasedMazeGame.createMaze')
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_param_names_list', [])
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_varargs_param_name', None)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_call_defaults', defaults)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_call_varargs', varargs)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FilebasedMazeGame.createMaze.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FilebasedMazeGame.createMaze', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'createMaze', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'createMaze(...)' code ##################

        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to MazeFactory(...): (line 453)
        # Processing the call keyword arguments (line 453)
        kwargs_1162 = {}
        # Getting the type of 'MazeFactory' (line 453)
        MazeFactory_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'MazeFactory', False)
        # Calling MazeFactory(args, kwargs) (line 453)
        MazeFactory_call_result_1163 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), MazeFactory_1161, *[], **kwargs_1162)
        
        # Assigning a type to the variable 'f' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'f', MazeFactory_call_result_1163)
        
        # Assigning a Call to a Name (line 454):
        
        # Assigning a Call to a Name (line 454):
        
        # Call to makeMaze(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'FILE_' (line 454)
        FILE__1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 23), 'FILE_', False)
        # Processing the call keyword arguments (line 454)
        kwargs_1167 = {}
        # Getting the type of 'f' (line 454)
        f_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'f', False)
        # Obtaining the member 'makeMaze' of a type (line 454)
        makeMaze_1165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 12), f_1164, 'makeMaze')
        # Calling makeMaze(args, kwargs) (line 454)
        makeMaze_call_result_1168 = invoke(stypy.reporting.localization.Localization(__file__, 454, 12), makeMaze_1165, *[FILE__1166], **kwargs_1167)
        
        # Assigning a type to the variable 'm' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'm', makeMaze_call_result_1168)
        # Getting the type of 'm' (line 456)
        m_1169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'm')
        # Assigning a type to the variable 'stypy_return_type' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'stypy_return_type', m_1169)
        
        # ################# End of 'createMaze(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'createMaze' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'createMaze'
        return stypy_return_type_1170


    @norecursion
    def getStartEndPoints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getStartEndPoints'
        module_type_store = module_type_store.open_function_context('getStartEndPoints', 458, 4, False)
        # Assigning a type to the variable 'self' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_localization', localization)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_type_store', module_type_store)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_function_name', 'FilebasedMazeGame.getStartEndPoints')
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_param_names_list', ['maze'])
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_varargs_param_name', None)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_call_defaults', defaults)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_call_varargs', varargs)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FilebasedMazeGame.getStartEndPoints.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FilebasedMazeGame.getStartEndPoints', ['maze'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getStartEndPoints', localization, ['maze'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getStartEndPoints(...)' code ##################

        
        # Getting the type of 'True' (line 460)
        True_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 14), 'True')
        # Testing the type of an if condition (line 460)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), True_1171)
        # SSA begins for while statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # SSA begins for try-except statement (line 461)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Str to a Name (line 463):
        
        # Assigning a Str to a Name (line 463):
        str_1172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 22), 'str', '0 4')
        # Assigning a type to the variable 'pt1' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'pt1', str_1172)
        
        # Assigning a Call to a Tuple (line 464):
        
        # Assigning a Call to a Name:
        
        # Call to split(...): (line 464)
        # Processing the call keyword arguments (line 464)
        kwargs_1175 = {}
        # Getting the type of 'pt1' (line 464)
        pt1_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 22), 'pt1', False)
        # Obtaining the member 'split' of a type (line 464)
        split_1174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 22), pt1_1173, 'split')
        # Calling split(args, kwargs) (line 464)
        split_call_result_1176 = invoke(stypy.reporting.localization.Localization(__file__, 464, 22), split_1174, *[], **kwargs_1175)
        
        # Assigning a type to the variable 'call_assignment_14' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'call_assignment_14', split_call_result_1176)
        
        # Assigning a Call to a Name (line 464):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 16), 'int')
        # Processing the call keyword arguments
        kwargs_1180 = {}
        # Getting the type of 'call_assignment_14' (line 464)
        call_assignment_14_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'call_assignment_14', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___1178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 16), call_assignment_14_1177, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1181 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1178, *[int_1179], **kwargs_1180)
        
        # Assigning a type to the variable 'call_assignment_15' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'call_assignment_15', getitem___call_result_1181)
        
        # Assigning a Name to a Name (line 464):
        # Getting the type of 'call_assignment_15' (line 464)
        call_assignment_15_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'call_assignment_15')
        # Assigning a type to the variable 'x' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'x', call_assignment_15_1182)
        
        # Assigning a Call to a Name (line 464):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 16), 'int')
        # Processing the call keyword arguments
        kwargs_1186 = {}
        # Getting the type of 'call_assignment_14' (line 464)
        call_assignment_14_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'call_assignment_14', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___1184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 16), call_assignment_14_1183, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1187 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1184, *[int_1185], **kwargs_1186)
        
        # Assigning a type to the variable 'call_assignment_16' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'call_assignment_16', getitem___call_result_1187)
        
        # Assigning a Name to a Name (line 464):
        # Getting the type of 'call_assignment_16' (line 464)
        call_assignment_16_1188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'call_assignment_16')
        # Assigning a type to the variable 'y' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 18), 'y', call_assignment_16_1188)
        
        # Assigning a Tuple to a Attribute (line 465):
        
        # Assigning a Tuple to a Attribute (line 465):
        
        # Obtaining an instance of the builtin type 'tuple' (line 465)
        tuple_1189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 465)
        # Adding element type (line 465)
        
        # Call to int(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'x' (line 465)
        x_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 35), 'x', False)
        # Processing the call keyword arguments (line 465)
        kwargs_1192 = {}
        # Getting the type of 'int' (line 465)
        int_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 31), 'int', False)
        # Calling int(args, kwargs) (line 465)
        int_call_result_1193 = invoke(stypy.reporting.localization.Localization(__file__, 465, 31), int_1190, *[x_1191], **kwargs_1192)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 31), tuple_1189, int_call_result_1193)
        # Adding element type (line 465)
        
        # Call to int(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'y' (line 465)
        y_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 43), 'y', False)
        # Processing the call keyword arguments (line 465)
        kwargs_1196 = {}
        # Getting the type of 'int' (line 465)
        int_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 39), 'int', False)
        # Calling int(args, kwargs) (line 465)
        int_call_result_1197 = invoke(stypy.reporting.localization.Localization(__file__, 465, 39), int_1194, *[y_1195], **kwargs_1196)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 31), tuple_1189, int_call_result_1197)
        
        # Getting the type of 'self' (line 465)
        self_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'self')
        # Setting the type of the member '_start' of a type (line 465)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 16), self_1198, '_start', tuple_1189)
        
        # Call to validatePoint(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'self' (line 466)
        self_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 35), 'self', False)
        # Obtaining the member '_start' of a type (line 466)
        _start_1202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 35), self_1201, '_start')
        # Processing the call keyword arguments (line 466)
        kwargs_1203 = {}
        # Getting the type of 'maze' (line 466)
        maze_1199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'maze', False)
        # Obtaining the member 'validatePoint' of a type (line 466)
        validatePoint_1200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), maze_1199, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 466)
        validatePoint_call_result_1204 = invoke(stypy.reporting.localization.Localization(__file__, 466, 16), validatePoint_1200, *[_start_1202], **kwargs_1203)
        
        # SSA branch for the except part of a try statement (line 461)
        # SSA branch for the except '<any exception>' branch of a try statement (line 461)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 461)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'True' (line 471)
        True_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 14), 'True')
        # Testing the type of an if condition (line 471)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 8), True_1205)
        # SSA begins for while statement (line 471)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # SSA begins for try-except statement (line 472)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Str to a Name (line 473):
        
        # Assigning a Str to a Name (line 473):
        str_1206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 22), 'str', '5 4')
        # Assigning a type to the variable 'pt2' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 16), 'pt2', str_1206)
        
        # Assigning a Call to a Tuple (line 474):
        
        # Assigning a Call to a Name:
        
        # Call to split(...): (line 474)
        # Processing the call keyword arguments (line 474)
        kwargs_1209 = {}
        # Getting the type of 'pt2' (line 474)
        pt2_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 22), 'pt2', False)
        # Obtaining the member 'split' of a type (line 474)
        split_1208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 22), pt2_1207, 'split')
        # Calling split(args, kwargs) (line 474)
        split_call_result_1210 = invoke(stypy.reporting.localization.Localization(__file__, 474, 22), split_1208, *[], **kwargs_1209)
        
        # Assigning a type to the variable 'call_assignment_17' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'call_assignment_17', split_call_result_1210)
        
        # Assigning a Call to a Name (line 474):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 16), 'int')
        # Processing the call keyword arguments
        kwargs_1214 = {}
        # Getting the type of 'call_assignment_17' (line 474)
        call_assignment_17_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'call_assignment_17', False)
        # Obtaining the member '__getitem__' of a type (line 474)
        getitem___1212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 16), call_assignment_17_1211, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1215 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1212, *[int_1213], **kwargs_1214)
        
        # Assigning a type to the variable 'call_assignment_18' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'call_assignment_18', getitem___call_result_1215)
        
        # Assigning a Name to a Name (line 474):
        # Getting the type of 'call_assignment_18' (line 474)
        call_assignment_18_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'call_assignment_18')
        # Assigning a type to the variable 'x' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'x', call_assignment_18_1216)
        
        # Assigning a Call to a Name (line 474):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 16), 'int')
        # Processing the call keyword arguments
        kwargs_1220 = {}
        # Getting the type of 'call_assignment_17' (line 474)
        call_assignment_17_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'call_assignment_17', False)
        # Obtaining the member '__getitem__' of a type (line 474)
        getitem___1218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 16), call_assignment_17_1217, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1221 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1218, *[int_1219], **kwargs_1220)
        
        # Assigning a type to the variable 'call_assignment_19' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'call_assignment_19', getitem___call_result_1221)
        
        # Assigning a Name to a Name (line 474):
        # Getting the type of 'call_assignment_19' (line 474)
        call_assignment_19_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'call_assignment_19')
        # Assigning a type to the variable 'y' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'y', call_assignment_19_1222)
        
        # Assigning a Tuple to a Attribute (line 475):
        
        # Assigning a Tuple to a Attribute (line 475):
        
        # Obtaining an instance of the builtin type 'tuple' (line 475)
        tuple_1223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 475)
        # Adding element type (line 475)
        
        # Call to int(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'x' (line 475)
        x_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 33), 'x', False)
        # Processing the call keyword arguments (line 475)
        kwargs_1226 = {}
        # Getting the type of 'int' (line 475)
        int_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 29), 'int', False)
        # Calling int(args, kwargs) (line 475)
        int_call_result_1227 = invoke(stypy.reporting.localization.Localization(__file__, 475, 29), int_1224, *[x_1225], **kwargs_1226)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 29), tuple_1223, int_call_result_1227)
        # Adding element type (line 475)
        
        # Call to int(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'y' (line 475)
        y_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 41), 'y', False)
        # Processing the call keyword arguments (line 475)
        kwargs_1230 = {}
        # Getting the type of 'int' (line 475)
        int_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 37), 'int', False)
        # Calling int(args, kwargs) (line 475)
        int_call_result_1231 = invoke(stypy.reporting.localization.Localization(__file__, 475, 37), int_1228, *[y_1229], **kwargs_1230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 29), tuple_1223, int_call_result_1231)
        
        # Getting the type of 'self' (line 475)
        self_1232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'self')
        # Setting the type of the member '_end' of a type (line 475)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 16), self_1232, '_end', tuple_1223)
        
        # Call to validatePoint(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'self' (line 476)
        self_1235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 35), 'self', False)
        # Obtaining the member '_end' of a type (line 476)
        _end_1236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 35), self_1235, '_end')
        # Processing the call keyword arguments (line 476)
        kwargs_1237 = {}
        # Getting the type of 'maze' (line 476)
        maze_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'maze', False)
        # Obtaining the member 'validatePoint' of a type (line 476)
        validatePoint_1234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 16), maze_1233, 'validatePoint')
        # Calling validatePoint(args, kwargs) (line 476)
        validatePoint_call_result_1238 = invoke(stypy.reporting.localization.Localization(__file__, 476, 16), validatePoint_1234, *[_end_1236], **kwargs_1237)
        
        # SSA branch for the except part of a try statement (line 472)
        # SSA branch for the except '<any exception>' branch of a try statement (line 472)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 472)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 471)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'getStartEndPoints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getStartEndPoints' in the type store
        # Getting the type of 'stypy_return_type' (line 458)
        stypy_return_type_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1239)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getStartEndPoints'
        return stypy_return_type_1239


# Assigning a type to the variable 'FilebasedMazeGame' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'FilebasedMazeGame', FilebasedMazeGame)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 481, 0, False)
    
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

    
    # Assigning a Call to a Name (line 482):
    
    # Assigning a Call to a Name (line 482):
    
    # Call to FilebasedMazeGame(...): (line 482)
    # Processing the call keyword arguments (line 482)
    kwargs_1241 = {}
    # Getting the type of 'FilebasedMazeGame' (line 482)
    FilebasedMazeGame_1240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'FilebasedMazeGame', False)
    # Calling FilebasedMazeGame(args, kwargs) (line 482)
    FilebasedMazeGame_call_result_1242 = invoke(stypy.reporting.localization.Localization(__file__, 482, 11), FilebasedMazeGame_1240, *[], **kwargs_1241)
    
    # Assigning a type to the variable 'game' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'game', FilebasedMazeGame_call_result_1242)
    
    
    # Call to range(...): (line 483)
    # Processing the call arguments (line 483)
    int_1244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 19), 'int')
    # Processing the call keyword arguments (line 483)
    kwargs_1245 = {}
    # Getting the type of 'range' (line 483)
    range_1243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 13), 'range', False)
    # Calling range(args, kwargs) (line 483)
    range_call_result_1246 = invoke(stypy.reporting.localization.Localization(__file__, 483, 13), range_1243, *[int_1244], **kwargs_1245)
    
    # Testing the type of a for loop iterable (line 483)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 483, 4), range_call_result_1246)
    # Getting the type of the for loop variable (line 483)
    for_loop_var_1247 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 483, 4), range_call_result_1246)
    # Assigning a type to the variable 'x' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'x', for_loop_var_1247)
    # SSA begins for a for statement (line 483)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to runGame(...): (line 484)
    # Processing the call keyword arguments (line 484)
    kwargs_1250 = {}
    # Getting the type of 'game' (line 484)
    game_1248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'game', False)
    # Obtaining the member 'runGame' of a type (line 484)
    runGame_1249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), game_1248, 'runGame')
    # Calling runGame(args, kwargs) (line 484)
    runGame_call_result_1251 = invoke(stypy.reporting.localization.Localization(__file__, 484, 8), runGame_1249, *[], **kwargs_1250)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 485)
    True_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'stypy_return_type', True_1252)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 481)
    stypy_return_type_1253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1253)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1253

# Assigning a type to the variable 'run' (line 481)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'run', run)

# Call to run(...): (line 487)
# Processing the call keyword arguments (line 487)
kwargs_1255 = {}
# Getting the type of 'run' (line 487)
run_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), 'run', False)
# Calling run(args, kwargs) (line 487)
run_call_result_1256 = invoke(stypy.reporting.localization.Localization(__file__, 487, 0), run_1254, *[], **kwargs_1255)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

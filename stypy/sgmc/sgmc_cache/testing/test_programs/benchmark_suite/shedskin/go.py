
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import random, math, sys
2: 
3: ''' copyright mark dufour, license GPL v3 '''
4: 
5: SIZE = 9
6: # GAMES = 15000
7: KOMI = 7.5
8: EMPTY, WHITE, BLACK = 0, 1, 2
9: SHOW = {EMPTY: '.', WHITE: 'o', BLACK: 'x'}
10: PASS = -1
11: MAXMOVES = SIZE * SIZE * 3
12: TIMESTAMP = REMOVESTAMP = 0
13: MOVES = 0
14: 
15: 
16: def to_pos(x, y):
17:     return y * SIZE + x
18: 
19: 
20: def to_xy(pos):
21:     y, x = divmod(pos, SIZE)
22:     return x, y
23: 
24: 
25: class Square:
26:     def __init__(self, board, pos):
27:         self.board = board
28:         self.pos = pos
29:         self.liberties = 0
30:         self.timestamp = TIMESTAMP
31:         self.timestamp2 = TIMESTAMP
32:         self.findstamp = TIMESTAMP
33:         self.removestamp = REMOVESTAMP
34:         self.zobrist_strings = [random.randrange(sys.maxint) for i in range(3)]
35: 
36:     def set_neighbours(self):
37:         x, y = self.pos % SIZE, self.pos / SIZE;
38:         self.neighbours = []
39:         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
40:             newx, newy = x + dx, y + dy
41:             if 0 <= newx < SIZE and 0 <= newy < SIZE:
42:                 self.neighbours.append(self.board.squares[to_pos(newx, newy)])
43: 
44:     def count_liberties(self, reference=None):
45:         if not reference:
46:             reference = self
47:             self.liberties = 0
48:         self.timestamp = TIMESTAMP
49:         for neighbour in self.neighbours:
50:             if neighbour.timestamp != TIMESTAMP:
51:                 neighbour.timestamp = TIMESTAMP
52:                 if neighbour.color == EMPTY:
53:                     reference.liberties += 1
54:                 elif neighbour.color == self.color:
55:                     neighbour.count_liberties(reference)
56: 
57:     def liberty(self):
58:         self.findstamp = TIMESTAMP
59:         for neighbour in self.neighbours:
60:             if neighbour.findstamp != TIMESTAMP:
61:                 neighbour.findstamp = TIMESTAMP
62:                 if neighbour.color == EMPTY:
63:                     return neighbour
64:                 elif neighbour.color == self.color:
65:                     liberty = neighbour.liberty()
66:                     if liberty:
67:                         return liberty
68: 
69:     def move(self, color):
70:         global TIMESTAMP, MOVES
71:         TIMESTAMP += 1
72:         MOVES += 1
73:         self.board.zobrist.update(self, color)
74:         self.color = color
75:         self.reference = self
76:         self.members = 1
77:         self.used = True
78:         self.board.atari = None
79:         for neighbour in self.neighbours:
80:             if neighbour.color != EMPTY:
81:                 neighbour_ref = neighbour.find(update=True)
82:                 if neighbour_ref.timestamp != TIMESTAMP:
83:                     neighbour_ref.timestamp = TIMESTAMP
84:                     if neighbour.color == color:
85:                         neighbour_ref.reference = self
86:                         self.members += neighbour_ref.members
87:                     else:
88:                         neighbour_ref.liberties -= 1
89:                         if neighbour_ref.liberties == 0:
90:                             neighbour_ref.remove(neighbour_ref, update=True)
91:                         elif neighbour_ref.liberties == 1:
92:                             self.board.atari = neighbour_ref
93:         TIMESTAMP += 1
94:         self.count_liberties()
95:         self.board.zobrist.add()
96: 
97:     def remove(self, reference, update=True):
98:         global REMOVESTAMP
99:         REMOVESTAMP += 1
100:         removestamp = REMOVESTAMP
101:         self.board.zobrist.update(self, EMPTY)
102:         self.timestamp2 = TIMESTAMP
103:         if update:
104:             self.color = EMPTY
105:             self.board.emptyset.add(self.pos)
106:         #            if color == BLACK:
107:         #                self.board.black_dead += 1
108:         #            else:
109:         #                self.board.white_dead += 1
110:         if update:
111:             for neighbour in self.neighbours:
112:                 if neighbour.color != EMPTY:
113:                     neighbour_ref = neighbour.find(update)
114:                     if neighbour_ref.pos != self.pos and neighbour_ref.removestamp != removestamp:
115:                         neighbour_ref.removestamp = removestamp
116:                         neighbour_ref.liberties += 1
117:         for neighbour in self.neighbours:
118:             if neighbour.color != EMPTY:
119:                 neighbour_ref = neighbour.find(update)
120:                 if neighbour_ref.pos == reference.pos and neighbour.timestamp2 != TIMESTAMP:
121:                     neighbour.remove(reference, update)
122: 
123:     def find(self, update=False):
124:         reference = self.reference
125:         if reference.pos != self.pos:
126:             reference = reference.find(update)
127:             if update:
128:                 self.reference = reference
129:         return reference
130: 
131:     def __repr__(self):
132:         return repr(to_xy(self.pos))
133: 
134: 
135: class EmptySet:
136:     def __init__(self, board):
137:         self.board = board
138:         self.empties = range(SIZE * SIZE)
139:         self.empty_pos = range(SIZE * SIZE)
140: 
141:     def random_choice(self):
142:         choices = len(self.empties)
143:         while choices:
144:             i = int(random.random() * choices)
145:             pos = self.empties[i]
146:             if self.board.useful(pos):
147:                 return pos
148:             choices -= 1
149:             self.set(i, self.empties[choices])
150:             self.set(choices, pos)
151:         return PASS
152: 
153:     def add(self, pos):
154:         self.empty_pos[pos] = len(self.empties)
155:         self.empties.append(pos)
156: 
157:     def remove(self, pos):
158:         self.set(self.empty_pos[pos], self.empties[len(self.empties) - 1])
159:         self.empties.pop()
160: 
161:     def set(self, i, pos):
162:         self.empties[i] = pos
163:         self.empty_pos[pos] = i
164: 
165: 
166: class ZobristHash:
167:     def __init__(self, board):
168:         self.board = board
169:         self.hash_set = set()
170:         self.hash = 0
171:         for square in self.board.squares:
172:             self.hash ^= square.zobrist_strings[EMPTY]
173:         self.hash_set.clear()
174:         self.hash_set.add(self.hash)
175: 
176:     def update(self, square, color):
177:         self.hash ^= square.zobrist_strings[square.color]
178:         self.hash ^= square.zobrist_strings[color]
179: 
180:     def add(self):
181:         self.hash_set.add(self.hash)
182: 
183:     def dupe(self):
184:         return self.hash in self.hash_set
185: 
186: 
187: class Board:
188:     def __init__(self):
189:         self.squares = [Square(self, pos) for pos in range(SIZE * SIZE)]
190:         for square in self.squares:
191:             square.set_neighbours()
192:         self.reset()
193: 
194:     def reset(self):
195:         for square in self.squares:
196:             square.color = EMPTY
197:             square.used = False
198:         self.emptyset = EmptySet(self)
199:         self.zobrist = ZobristHash(self)
200:         self.color = BLACK
201:         self.finished = False
202:         self.lastmove = -2
203:         self.history = []
204:         self.atari = None
205:         self.white_dead = 0
206:         self.black_dead = 0
207: 
208:     def move(self, pos):
209:         square = self.squares[pos]
210:         if pos != PASS:
211:             square.move(self.color)
212:             self.emptyset.remove(square.pos)
213:         elif self.lastmove == PASS:
214:             self.finished = True
215:         if self.color == BLACK:
216:             self.color = WHITE
217:         else:
218:             self.color = BLACK
219:         self.lastmove = pos
220:         self.history.append(pos)
221: 
222:     def random_move(self):
223:         return self.emptyset.random_choice()
224: 
225:     def useful_fast(self, square):
226:         if not square.used:
227:             for neighbour in square.neighbours:
228:                 if neighbour.color == EMPTY:
229:                     return True
230:         return False
231: 
232:     def useful(self, pos):
233:         global TIMESTAMP
234:         TIMESTAMP += 1
235:         square = self.squares[pos]
236:         if self.useful_fast(square):
237:             return True
238:         old_hash = self.zobrist.hash
239:         self.zobrist.update(square, self.color)
240:         empties = strong_opps = weak_opps = strong_neighs = weak_neighs = 0
241:         for neighbour in square.neighbours:
242:             if neighbour.color == EMPTY:
243:                 empties += 1
244:             else:
245:                 neighbour_ref = neighbour.find()
246:                 if neighbour_ref.timestamp != TIMESTAMP:
247:                     neighbour_ref.timestamp = TIMESTAMP
248:                     weak = (neighbour_ref.liberties == 1)
249:                     if neighbour.color == self.color:
250:                         if weak:
251:                             weak_neighs += 1
252:                         else:
253:                             strong_neighs += 1
254:                     else:
255:                         if weak:
256:                             weak_opps += 1
257:                             neighbour_ref.remove(neighbour_ref, update=False)
258:                         else:
259:                             strong_opps += 1
260:         dupe = self.zobrist.dupe()
261:         self.zobrist.hash = old_hash
262:         return not dupe and \
263:                bool(empties or weak_opps or (strong_neighs and (strong_opps or weak_neighs)))
264: 
265:     def useful_moves(self):
266:         return [pos for pos in self.emptyset.empties if self.useful(pos)]
267: 
268:     def replay(self, history):
269:         for pos in history:
270:             self.move(pos)
271: 
272:     def score(self, color):
273:         if color == WHITE:
274:             count = KOMI + self.black_dead
275:         else:
276:             count = self.white_dead
277:         for square in self.squares:
278:             squarecolor = square.color
279:             if squarecolor == color:
280:                 count += 1
281:             elif squarecolor == EMPTY:
282:                 surround = 0
283:                 for neighbour in square.neighbours:
284:                     if neighbour.color == color:
285:                         surround += 1
286:                 if surround == len(square.neighbours):
287:                     count += 1
288:         return count
289: 
290:     def check(self):
291:         for square in self.squares:
292:             if square.color == EMPTY:
293:                 continue
294: 
295:             members1 = set([square])
296:             changed = True
297:             while changed:
298:                 changed = False
299:                 for member in members1.copy():
300:                     for neighbour in member.neighbours:
301:                         if neighbour.color == square.color and neighbour not in members1:
302:                             changed = True
303:                             members1.add(neighbour)
304:             liberties1 = set()
305:             for member in members1:
306:                 for neighbour in member.neighbours:
307:                     if neighbour.color == EMPTY:
308:                         liberties1.add(neighbour.pos)
309: 
310:             root = square.find()
311: 
312:             # print 'members1', square, root, members1
313:             # print 'ledges1', square, ledges1
314: 
315:             members2 = set()
316:             for square2 in self.squares:
317:                 if square2.color != EMPTY and square2.find() == root:
318:                     members2.add(square2)
319: 
320:             liberties2 = root.liberties
321:             # print 'members2', square, root, members1
322:             # print 'ledges2', square, ledges2
323: 
324:             assert members1 == members2
325:             assert len(liberties1) == liberties2, (
326:                     'liberties differ at %r: %d %d' % (root, len(liberties1), liberties2))
327: 
328:             empties1 = set(self.emptyset.empties)
329: 
330:             empties2 = set()
331:             for square in self.squares:
332:                 if square.color == EMPTY:
333:                     empties2.add(square.pos)
334: 
335:             assert empties1 == empties2
336: 
337:     def __repr__(self):
338:         result = []
339:         for y in range(SIZE):
340:             start = to_pos(0, y)
341:             result.append(''.join([SHOW[square.color] + ' ' for square in self.squares[start:start + SIZE]]))
342:         return '\n'.join(result)
343: 
344: 
345: class UCTNode:
346:     def __init__(self):
347:         self.bestchild = None
348:         self.pos = -1
349:         self.wins = 0
350:         self.losses = 0
351:         self.pos_child = [None for x in range(SIZE * SIZE)]
352:         self.amafvisits = 0
353:         self.pos_amaf_wins = [0 for x in range(SIZE * SIZE)]
354:         self.pos_amaf_losses = [0 for x in range(SIZE * SIZE)]
355:         self.parent = None
356: 
357:     def play(self, board):
358:         ''' uct tree search '''
359:         color = board.color
360:         node = self
361:         path = [node]
362:         histpos = len(board.history)
363:         while True:
364:             pos = node.select(board)
365:             if pos == PASS:
366:                 break
367:             board.move(pos)
368:             child = node.pos_child[pos]
369:             if not child:
370:                 child = node.pos_child[pos] = UCTNode()
371:                 child.unexplored = board.useful_moves()
372:                 child.pos = pos
373:                 child.parent = node
374:                 path.append(child)
375:                 break
376:             path.append(child)
377:             node = child
378:         self.random_playout(board)
379:         self.update_path(board, histpos, color, path)
380: 
381:     def select(self, board):
382:         ''' select move; unexplored children first, then according to uct value '''
383:         if self.unexplored:
384:             i = random.randrange(len(self.unexplored))
385:             pos = self.unexplored[i]
386:             self.unexplored[i] = self.unexplored[len(self.unexplored) - 1]
387:             self.unexplored.pop()
388:             return pos
389:         elif self.bestchild:
390:             return self.bestchild.pos
391:         else:
392:             return PASS
393: 
394:     def random_playout(self, board):
395:         ''' random play until both players pass '''
396:         for x in range(MAXMOVES):  # XXX while not self.finished?
397:             if board.finished:
398:                 break
399:             pos = PASS
400:             if board.atari:
401:                 liberty = board.atari.liberty()
402:                 if board.useful(liberty.pos):
403:                     pos = liberty.pos
404:             if pos == PASS:
405:                 pos = board.random_move()
406:             #            print 'pos color', to_xy(pos), SHOW[board.color]
407:             board.move(pos)
408: 
409:     #            print board
410:     #            board.check()
411:     #        print 'WHITE:', board.score(WHITE)
412:     #        print 'BLACK:', board.score(BLACK)
413: 
414:     def update_path(self, board, histpos, color, path):
415:         ''' update win/loss count along path '''
416:         wins = board.score(BLACK) >= board.score(WHITE)
417:         for node in path:
418:             if color == BLACK:
419:                 color = WHITE
420:             else:
421:                 color = BLACK
422:             if wins == (color == BLACK):
423:                 node.wins += 1
424:             else:
425:                 node.losses += 1
426:             if node.parent:
427:                 for i in range(histpos + 2, len(board.history), 2):
428:                     pos = board.history[i]
429:                     if pos == PASS:
430:                         break
431:                     if wins == (color == BLACK):
432:                         node.parent.pos_amaf_wins[pos] += 1
433:                     else:
434:                         node.parent.pos_amaf_losses[pos] += 1
435:                     node.parent.amafvisits += 1
436:                 node.parent.bestchild = node.parent.best_child()
437: 
438:     def score(self):
439:         winrate = self.wins / float(self.wins + self.losses)
440:         parentvisits = self.parent.wins + self.parent.losses
441:         if not parentvisits:
442:             return winrate
443:         nodevisits = self.wins + self.losses
444:         uct_score = winrate + math.sqrt((math.log(parentvisits)) / (5 * nodevisits))
445: 
446:         amafvisits = self.parent.pos_amaf_wins[self.pos] + self.parent.pos_amaf_losses[self.pos]
447:         if not amafvisits:
448:             return uct_score
449:         amafwinrate = self.parent.pos_amaf_wins[self.pos] / float(amafvisits)
450:         uct_amaf = amafwinrate + math.sqrt((math.log(self.parent.amafvisits)) / (5 * amafvisits))
451: 
452:         beta = math.sqrt(1000.0 / (3 * parentvisits + 1000.0))
453:         return beta * uct_amaf + (1 - beta) * uct_score
454: 
455:     def best_child(self):
456:         maxscore = -1
457:         maxchild = None
458:         for child in self.pos_child:
459:             if child and child.score() > maxscore:
460:                 maxchild = child
461:                 maxscore = child.score()
462:         return maxchild
463: 
464:     def best_visited(self):
465:         maxvisits = -1
466:         maxchild = None
467:         for child in self.pos_child:
468:             #            if child:
469:             #                print to_xy(child.pos), child.wins, child.losses, child.score()
470:             if child and (child.wins + child.losses) > maxvisits:
471:                 maxvisits, maxchild = (child.wins + child.losses), child
472:         return maxchild
473: 
474: 
475: def user_move(board):
476:     pos = to_pos(0, 0)
477:     return pos
478: 
479: 
480: ##    while True:
481: ##        text = raw_input('?').strip()
482: ##        if text == 'p':
483: ##            return PASS
484: ##        if text == 'q':
485: ##            raise EOFError
486: ##        try:
487: ##            x, y = [int(i) for i in text.split()]
488: ##        except ValueError:
489: ##            continue
490: ##        if not (0 <= x < SIZE and 0 <= y < SIZE):
491: ##            continue
492: ##        pos = to_pos(x, y)
493: ##        if board.useful(pos): 
494: ##            return pos
495: 
496: def computer_move(board):
497:     global MOVES
498:     pos = board.random_move()
499:     if pos == PASS:
500:         return PASS
501:     tree = UCTNode()
502:     tree.unexplored = board.useful_moves()
503:     nboard = Board()
504:     GAMES = min(25000 - (1000 * len(board.history)) / 4, 1000)
505:     #    GAMES = 100000
506:     for game in range(GAMES):
507:         node = tree
508:         nboard.reset()
509:         nboard.replay(board.history)
510:         node.play(nboard)
511:     #    for pos in range(SIZE*SIZE):
512:     #        print 'amaf', to_xy(pos), node.pos_child[pos].score() #node.pos_amaf_wins[pos]/float(node.pos_amaf_wins[pos]+node.pos_amaf_losses[pos])
513:     #    print 'moves', MOVES
514:     return tree.best_visited().pos
515: 
516: 
517: def versus_cpu():
518:     board = Board()
519:     maxturns = 4
520:     turns = 0
521:     while turns < maxturns:
522:         if board.lastmove != PASS:
523:             pass  # print board
524:         # print 'thinking..'
525:         pos = computer_move(board)
526:         if pos == PASS:
527:             pass  # print 'I pass.'
528:         else:
529:             # print 'I move here:', to_xy(pos)
530:             to_xy(pos)
531:         board.move(pos)
532:         # break
533:         # board.check()
534:         if board.finished:
535:             break
536:         if board.lastmove != PASS:
537:             pass  # print board
538:         pos = user_move(board)
539:         board.move(pos)
540:         # board.check()
541:         if board.finished:
542:             break
543:         turns += 1
544:     # print 'WHITE:', board.score(WHITE)
545:     board.score(WHITE)
546:     # print 'BLACK:', board.score(BLACK)
547:     board.score(BLACK)
548: 
549: 
550: def run():
551:     random.seed(1)
552:     try:
553:         versus_cpu()
554:     except EOFError:
555:         pass
556:     return True
557: 
558: 
559: run()
560: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# Multiple import statement. import random (1/3) (line 1)
import random

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'random', random, module_type_store)
# Multiple import statement. import math (2/3) (line 1)
import math

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', math, module_type_store)
# Multiple import statement. import sys (3/3) (line 1)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sys', sys, module_type_store)

str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 0), 'str', ' copyright mark dufour, license GPL v3 ')

# Assigning a Num to a Name (line 5):

# Assigning a Num to a Name (line 5):
int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 7), 'int')
# Assigning a type to the variable 'SIZE' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'SIZE', int_14)

# Assigning a Num to a Name (line 7):

# Assigning a Num to a Name (line 7):
float_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 7), 'float')
# Assigning a type to the variable 'KOMI' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'KOMI', float_15)

# Assigning a Tuple to a Tuple (line 8):

# Assigning a Num to a Name (line 8):
int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 22), 'int')
# Assigning a type to the variable 'tuple_assignment_1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_1', int_16)

# Assigning a Num to a Name (line 8):
int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 25), 'int')
# Assigning a type to the variable 'tuple_assignment_2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_2', int_17)

# Assigning a Num to a Name (line 8):
int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 28), 'int')
# Assigning a type to the variable 'tuple_assignment_3' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_3', int_18)

# Assigning a Name to a Name (line 8):
# Getting the type of 'tuple_assignment_1' (line 8)
tuple_assignment_1_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_1')
# Assigning a type to the variable 'EMPTY' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'EMPTY', tuple_assignment_1_19)

# Assigning a Name to a Name (line 8):
# Getting the type of 'tuple_assignment_2' (line 8)
tuple_assignment_2_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_2')
# Assigning a type to the variable 'WHITE' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'WHITE', tuple_assignment_2_20)

# Assigning a Name to a Name (line 8):
# Getting the type of 'tuple_assignment_3' (line 8)
tuple_assignment_3_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_3')
# Assigning a type to the variable 'BLACK' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'BLACK', tuple_assignment_3_21)

# Assigning a Dict to a Name (line 9):

# Assigning a Dict to a Name (line 9):

# Obtaining an instance of the builtin type 'dict' (line 9)
dict_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 9)
# Adding element type (key, value) (line 9)
# Getting the type of 'EMPTY' (line 9)
EMPTY_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'EMPTY')
str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', '.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 7), dict_22, (EMPTY_23, str_24))
# Adding element type (key, value) (line 9)
# Getting the type of 'WHITE' (line 9)
WHITE_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'WHITE')
str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'str', 'o')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 7), dict_22, (WHITE_25, str_26))
# Adding element type (key, value) (line 9)
# Getting the type of 'BLACK' (line 9)
BLACK_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 32), 'BLACK')
str_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 39), 'str', 'x')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 7), dict_22, (BLACK_27, str_28))

# Assigning a type to the variable 'SHOW' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'SHOW', dict_22)

# Assigning a Num to a Name (line 10):

# Assigning a Num to a Name (line 10):
int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 7), 'int')
# Assigning a type to the variable 'PASS' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'PASS', int_29)

# Assigning a BinOp to a Name (line 11):

# Assigning a BinOp to a Name (line 11):
# Getting the type of 'SIZE' (line 11)
SIZE_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'SIZE')
# Getting the type of 'SIZE' (line 11)
SIZE_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'SIZE')
# Applying the binary operator '*' (line 11)
result_mul_32 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 11), '*', SIZE_30, SIZE_31)

int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 25), 'int')
# Applying the binary operator '*' (line 11)
result_mul_34 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 23), '*', result_mul_32, int_33)

# Assigning a type to the variable 'MAXMOVES' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'MAXMOVES', result_mul_34)

# Multiple assignment of 2 elements.

# Assigning a Num to a Name (line 12):
int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'int')
# Assigning a type to the variable 'REMOVESTAMP' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'REMOVESTAMP', int_35)

# Assigning a Name to a Name (line 12):
# Getting the type of 'REMOVESTAMP' (line 12)
REMOVESTAMP_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'REMOVESTAMP')
# Assigning a type to the variable 'TIMESTAMP' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TIMESTAMP', REMOVESTAMP_36)

# Assigning a Num to a Name (line 13):

# Assigning a Num to a Name (line 13):
int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 8), 'int')
# Assigning a type to the variable 'MOVES' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MOVES', int_37)

@norecursion
def to_pos(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'to_pos'
    module_type_store = module_type_store.open_function_context('to_pos', 16, 0, False)
    
    # Passed parameters checking function
    to_pos.stypy_localization = localization
    to_pos.stypy_type_of_self = None
    to_pos.stypy_type_store = module_type_store
    to_pos.stypy_function_name = 'to_pos'
    to_pos.stypy_param_names_list = ['x', 'y']
    to_pos.stypy_varargs_param_name = None
    to_pos.stypy_kwargs_param_name = None
    to_pos.stypy_call_defaults = defaults
    to_pos.stypy_call_varargs = varargs
    to_pos.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'to_pos', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'to_pos', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'to_pos(...)' code ##################

    # Getting the type of 'y' (line 17)
    y_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'y')
    # Getting the type of 'SIZE' (line 17)
    SIZE_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'SIZE')
    # Applying the binary operator '*' (line 17)
    result_mul_40 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 11), '*', y_38, SIZE_39)
    
    # Getting the type of 'x' (line 17)
    x_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'x')
    # Applying the binary operator '+' (line 17)
    result_add_42 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 11), '+', result_mul_40, x_41)
    
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', result_add_42)
    
    # ################# End of 'to_pos(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_pos' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_43)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_pos'
    return stypy_return_type_43

# Assigning a type to the variable 'to_pos' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'to_pos', to_pos)

@norecursion
def to_xy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'to_xy'
    module_type_store = module_type_store.open_function_context('to_xy', 20, 0, False)
    
    # Passed parameters checking function
    to_xy.stypy_localization = localization
    to_xy.stypy_type_of_self = None
    to_xy.stypy_type_store = module_type_store
    to_xy.stypy_function_name = 'to_xy'
    to_xy.stypy_param_names_list = ['pos']
    to_xy.stypy_varargs_param_name = None
    to_xy.stypy_kwargs_param_name = None
    to_xy.stypy_call_defaults = defaults
    to_xy.stypy_call_varargs = varargs
    to_xy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'to_xy', ['pos'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'to_xy', localization, ['pos'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'to_xy(...)' code ##################

    
    # Assigning a Call to a Tuple (line 21):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'pos' (line 21)
    pos_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'pos', False)
    # Getting the type of 'SIZE' (line 21)
    SIZE_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'SIZE', False)
    # Processing the call keyword arguments (line 21)
    kwargs_47 = {}
    # Getting the type of 'divmod' (line 21)
    divmod_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'divmod', False)
    # Calling divmod(args, kwargs) (line 21)
    divmod_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), divmod_44, *[pos_45, SIZE_46], **kwargs_47)
    
    # Assigning a type to the variable 'call_assignment_4' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_4', divmod_call_result_48)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_4' (line 21)
    call_assignment_4_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_4', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_50 = stypy_get_value_from_tuple(call_assignment_4_49, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_5' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_5', stypy_get_value_from_tuple_call_result_50)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'call_assignment_5' (line 21)
    call_assignment_5_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_5')
    # Assigning a type to the variable 'y' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'y', call_assignment_5_51)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_4' (line 21)
    call_assignment_4_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_4', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_53 = stypy_get_value_from_tuple(call_assignment_4_52, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_6' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_6', stypy_get_value_from_tuple_call_result_53)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'call_assignment_6' (line 21)
    call_assignment_6_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_6')
    # Assigning a type to the variable 'x' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'x', call_assignment_6_54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    # Getting the type of 'x' (line 22)
    x_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_55, x_56)
    # Adding element type (line 22)
    # Getting the type of 'y' (line 22)
    y_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_55, y_57)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', tuple_55)
    
    # ################# End of 'to_xy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_xy' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_xy'
    return stypy_return_type_58

# Assigning a type to the variable 'to_xy' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'to_xy', to_xy)
# Declaration of the 'Square' class

class Square:

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.__init__', ['board', 'pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['board', 'pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 27):
        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'board' (line 27)
        board_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'board')
        # Getting the type of 'self' (line 27)
        self_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'board' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_60, 'board', board_59)
        
        # Assigning a Name to a Attribute (line 28):
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'pos' (line 28)
        pos_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'pos')
        # Getting the type of 'self' (line 28)
        self_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_62, 'pos', pos_61)
        
        # Assigning a Num to a Attribute (line 29):
        
        # Assigning a Num to a Attribute (line 29):
        int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'int')
        # Getting the type of 'self' (line 29)
        self_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'liberties' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_64, 'liberties', int_63)
        
        # Assigning a Name to a Attribute (line 30):
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'TIMESTAMP' (line 30)
        TIMESTAMP_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 30)
        self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'timestamp' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_66, 'timestamp', TIMESTAMP_65)
        
        # Assigning a Name to a Attribute (line 31):
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'TIMESTAMP' (line 31)
        TIMESTAMP_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'TIMESTAMP')
        # Getting the type of 'self' (line 31)
        self_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'timestamp2' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_68, 'timestamp2', TIMESTAMP_67)
        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'TIMESTAMP' (line 32)
        TIMESTAMP_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 32)
        self_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'findstamp' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_70, 'findstamp', TIMESTAMP_69)
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'REMOVESTAMP' (line 33)
        REMOVESTAMP_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'REMOVESTAMP')
        # Getting the type of 'self' (line 33)
        self_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'removestamp' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_72, 'removestamp', REMOVESTAMP_71)
        
        # Assigning a ListComp to a Attribute (line 34):
        
        # Assigning a ListComp to a Attribute (line 34):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 34)
        # Processing the call arguments (line 34)
        int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 76), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_81 = {}
        # Getting the type of 'range' (line 34)
        range_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 70), 'range', False)
        # Calling range(args, kwargs) (line 34)
        range_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 34, 70), range_79, *[int_80], **kwargs_81)
        
        comprehension_83 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), range_call_result_82)
        # Assigning a type to the variable 'i' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'i', comprehension_83)
        
        # Call to randrange(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'sys' (line 34)
        sys_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 49), 'sys', False)
        # Obtaining the member 'maxint' of a type (line 34)
        maxint_76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 49), sys_75, 'maxint')
        # Processing the call keyword arguments (line 34)
        kwargs_77 = {}
        # Getting the type of 'random' (line 34)
        random_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'random', False)
        # Obtaining the member 'randrange' of a type (line 34)
        randrange_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 32), random_73, 'randrange')
        # Calling randrange(args, kwargs) (line 34)
        randrange_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 34, 32), randrange_74, *[maxint_76], **kwargs_77)
        
        list_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), list_84, randrange_call_result_78)
        # Getting the type of 'self' (line 34)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'zobrist_strings' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_85, 'zobrist_strings', list_84)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_neighbours(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_neighbours'
        module_type_store = module_type_store.open_function_context('set_neighbours', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Square.set_neighbours.__dict__.__setitem__('stypy_localization', localization)
        Square.set_neighbours.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Square.set_neighbours.__dict__.__setitem__('stypy_type_store', module_type_store)
        Square.set_neighbours.__dict__.__setitem__('stypy_function_name', 'Square.set_neighbours')
        Square.set_neighbours.__dict__.__setitem__('stypy_param_names_list', [])
        Square.set_neighbours.__dict__.__setitem__('stypy_varargs_param_name', None)
        Square.set_neighbours.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Square.set_neighbours.__dict__.__setitem__('stypy_call_defaults', defaults)
        Square.set_neighbours.__dict__.__setitem__('stypy_call_varargs', varargs)
        Square.set_neighbours.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Square.set_neighbours.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.set_neighbours', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_neighbours', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_neighbours(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 37):
        
        # Assigning a BinOp to a Name (line 37):
        # Getting the type of 'self' (line 37)
        self_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self')
        # Obtaining the member 'pos' of a type (line 37)
        pos_87 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), self_86, 'pos')
        # Getting the type of 'SIZE' (line 37)
        SIZE_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'SIZE')
        # Applying the binary operator '%' (line 37)
        result_mod_89 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 15), '%', pos_87, SIZE_88)
        
        # Assigning a type to the variable 'tuple_assignment_7' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_7', result_mod_89)
        
        # Assigning a BinOp to a Name (line 37):
        # Getting the type of 'self' (line 37)
        self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 32), 'self')
        # Obtaining the member 'pos' of a type (line 37)
        pos_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 32), self_90, 'pos')
        # Getting the type of 'SIZE' (line 37)
        SIZE_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 43), 'SIZE')
        # Applying the binary operator 'div' (line 37)
        result_div_93 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 32), 'div', pos_91, SIZE_92)
        
        # Assigning a type to the variable 'tuple_assignment_8' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_8', result_div_93)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'tuple_assignment_7' (line 37)
        tuple_assignment_7_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_7')
        # Assigning a type to the variable 'x' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'x', tuple_assignment_7_94)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'tuple_assignment_8' (line 37)
        tuple_assignment_8_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_8')
        # Assigning a type to the variable 'y' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'y', tuple_assignment_8_95)
        
        # Assigning a List to a Attribute (line 38):
        
        # Assigning a List to a Attribute (line 38):
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        
        # Getting the type of 'self' (line 38)
        self_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'neighbours' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_97, 'neighbours', list_96)
        
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 24), tuple_99, int_100)
        # Adding element type (line 39)
        int_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 24), tuple_99, int_101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_98, tuple_99)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 33), tuple_102, int_103)
        # Adding element type (line 39)
        int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 33), tuple_102, int_104)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_98, tuple_102)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 41), tuple_105, int_106)
        # Adding element type (line 39)
        int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 41), tuple_105, int_107)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_98, tuple_105)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 50), tuple_108, int_109)
        # Adding element type (line 39)
        int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 50), tuple_108, int_110)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_98, tuple_108)
        
        # Assigning a type to the variable 'list_98' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'list_98', list_98)
        # Testing if the for loop is going to be iterated (line 39)
        # Testing the type of a for loop iterable (line 39)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 8), list_98)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 39, 8), list_98):
            # Getting the type of the for loop variable (line 39)
            for_loop_var_111 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 8), list_98)
            # Assigning a type to the variable 'dx' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'dx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 8), for_loop_var_111, 2, 0))
            # Assigning a type to the variable 'dy' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'dy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 8), for_loop_var_111, 2, 1))
            # SSA begins for a for statement (line 39)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Tuple to a Tuple (line 40):
            
            # Assigning a BinOp to a Name (line 40):
            # Getting the type of 'x' (line 40)
            x_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'x')
            # Getting the type of 'dx' (line 40)
            dx_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'dx')
            # Applying the binary operator '+' (line 40)
            result_add_114 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 25), '+', x_112, dx_113)
            
            # Assigning a type to the variable 'tuple_assignment_9' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_9', result_add_114)
            
            # Assigning a BinOp to a Name (line 40):
            # Getting the type of 'y' (line 40)
            y_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'y')
            # Getting the type of 'dy' (line 40)
            dy_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 37), 'dy')
            # Applying the binary operator '+' (line 40)
            result_add_117 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 33), '+', y_115, dy_116)
            
            # Assigning a type to the variable 'tuple_assignment_10' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_10', result_add_117)
            
            # Assigning a Name to a Name (line 40):
            # Getting the type of 'tuple_assignment_9' (line 40)
            tuple_assignment_9_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_9')
            # Assigning a type to the variable 'newx' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'newx', tuple_assignment_9_118)
            
            # Assigning a Name to a Name (line 40):
            # Getting the type of 'tuple_assignment_10' (line 40)
            tuple_assignment_10_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_10')
            # Assigning a type to the variable 'newy' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'newy', tuple_assignment_10_119)
            
            # Evaluating a boolean operation
            
            int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'int')
            # Getting the type of 'newx' (line 41)
            newx_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'newx')
            # Applying the binary operator '<=' (line 41)
            result_le_122 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '<=', int_120, newx_121)
            # Getting the type of 'SIZE' (line 41)
            SIZE_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'SIZE')
            # Applying the binary operator '<' (line 41)
            result_lt_124 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '<', newx_121, SIZE_123)
            # Applying the binary operator '&' (line 41)
            result_and__125 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '&', result_le_122, result_lt_124)
            
            
            int_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'int')
            # Getting the type of 'newy' (line 41)
            newy_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), 'newy')
            # Applying the binary operator '<=' (line 41)
            result_le_128 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '<=', int_126, newy_127)
            # Getting the type of 'SIZE' (line 41)
            SIZE_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'SIZE')
            # Applying the binary operator '<' (line 41)
            result_lt_130 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '<', newy_127, SIZE_129)
            # Applying the binary operator '&' (line 41)
            result_and__131 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '&', result_le_128, result_lt_130)
            
            # Applying the binary operator 'and' (line 41)
            result_and_keyword_132 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), 'and', result_and__125, result_and__131)
            
            # Testing if the type of an if condition is none (line 41)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 12), result_and_keyword_132):
                pass
            else:
                
                # Testing the type of an if condition (line 41)
                if_condition_133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), result_and_keyword_132)
                # Assigning a type to the variable 'if_condition_133' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_133', if_condition_133)
                # SSA begins for if statement (line 41)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 42)
                # Processing the call arguments (line 42)
                
                # Obtaining the type of the subscript
                
                # Call to to_pos(...): (line 42)
                # Processing the call arguments (line 42)
                # Getting the type of 'newx' (line 42)
                newx_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 65), 'newx', False)
                # Getting the type of 'newy' (line 42)
                newy_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 71), 'newy', False)
                # Processing the call keyword arguments (line 42)
                kwargs_140 = {}
                # Getting the type of 'to_pos' (line 42)
                to_pos_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 58), 'to_pos', False)
                # Calling to_pos(args, kwargs) (line 42)
                to_pos_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 42, 58), to_pos_137, *[newx_138, newy_139], **kwargs_140)
                
                # Getting the type of 'self' (line 42)
                self_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'self', False)
                # Obtaining the member 'board' of a type (line 42)
                board_143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), self_142, 'board')
                # Obtaining the member 'squares' of a type (line 42)
                squares_144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), board_143, 'squares')
                # Obtaining the member '__getitem__' of a type (line 42)
                getitem___145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), squares_144, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                subscript_call_result_146 = invoke(stypy.reporting.localization.Localization(__file__, 42, 39), getitem___145, to_pos_call_result_141)
                
                # Processing the call keyword arguments (line 42)
                kwargs_147 = {}
                # Getting the type of 'self' (line 42)
                self_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'self', False)
                # Obtaining the member 'neighbours' of a type (line 42)
                neighbours_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), self_134, 'neighbours')
                # Obtaining the member 'append' of a type (line 42)
                append_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), neighbours_135, 'append')
                # Calling append(args, kwargs) (line 42)
                append_call_result_148 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), append_136, *[subscript_call_result_146], **kwargs_147)
                
                # SSA join for if statement (line 41)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'set_neighbours(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_neighbours' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_neighbours'
        return stypy_return_type_149


    @norecursion
    def count_liberties(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 44)
        None_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'None')
        defaults = [None_150]
        # Create a new context for function 'count_liberties'
        module_type_store = module_type_store.open_function_context('count_liberties', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Square.count_liberties.__dict__.__setitem__('stypy_localization', localization)
        Square.count_liberties.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Square.count_liberties.__dict__.__setitem__('stypy_type_store', module_type_store)
        Square.count_liberties.__dict__.__setitem__('stypy_function_name', 'Square.count_liberties')
        Square.count_liberties.__dict__.__setitem__('stypy_param_names_list', ['reference'])
        Square.count_liberties.__dict__.__setitem__('stypy_varargs_param_name', None)
        Square.count_liberties.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Square.count_liberties.__dict__.__setitem__('stypy_call_defaults', defaults)
        Square.count_liberties.__dict__.__setitem__('stypy_call_varargs', varargs)
        Square.count_liberties.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Square.count_liberties.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.count_liberties', ['reference'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'count_liberties', localization, ['reference'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'count_liberties(...)' code ##################

        
        # Getting the type of 'reference' (line 45)
        reference_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'reference')
        # Applying the 'not' unary operator (line 45)
        result_not__152 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), 'not', reference_151)
        
        # Testing if the type of an if condition is none (line 45)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 8), result_not__152):
            pass
        else:
            
            # Testing the type of an if condition (line 45)
            if_condition_153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_not__152)
            # Assigning a type to the variable 'if_condition_153' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_153', if_condition_153)
            # SSA begins for if statement (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 46):
            
            # Assigning a Name to a Name (line 46):
            # Getting the type of 'self' (line 46)
            self_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'self')
            # Assigning a type to the variable 'reference' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'reference', self_154)
            
            # Assigning a Num to a Attribute (line 47):
            
            # Assigning a Num to a Attribute (line 47):
            int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'int')
            # Getting the type of 'self' (line 47)
            self_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self')
            # Setting the type of the member 'liberties' of a type (line 47)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_156, 'liberties', int_155)
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'TIMESTAMP' (line 48)
        TIMESTAMP_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 48)
        self_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'timestamp' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_158, 'timestamp', TIMESTAMP_157)
        
        # Getting the type of 'self' (line 49)
        self_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 49)
        neighbours_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), self_159, 'neighbours')
        # Assigning a type to the variable 'neighbours_160' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'neighbours_160', neighbours_160)
        # Testing if the for loop is going to be iterated (line 49)
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), neighbours_160)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 49, 8), neighbours_160):
            # Getting the type of the for loop variable (line 49)
            for_loop_var_161 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), neighbours_160)
            # Assigning a type to the variable 'neighbour' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'neighbour', for_loop_var_161)
            # SSA begins for a for statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'neighbour' (line 50)
            neighbour_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'neighbour')
            # Obtaining the member 'timestamp' of a type (line 50)
            timestamp_163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), neighbour_162, 'timestamp')
            # Getting the type of 'TIMESTAMP' (line 50)
            TIMESTAMP_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'TIMESTAMP')
            # Applying the binary operator '!=' (line 50)
            result_ne_165 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '!=', timestamp_163, TIMESTAMP_164)
            
            # Testing if the type of an if condition is none (line 50)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 50, 12), result_ne_165):
                pass
            else:
                
                # Testing the type of an if condition (line 50)
                if_condition_166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), result_ne_165)
                # Assigning a type to the variable 'if_condition_166' (line 50)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_condition_166', if_condition_166)
                # SSA begins for if statement (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'TIMESTAMP' (line 51)
                TIMESTAMP_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'TIMESTAMP')
                # Getting the type of 'neighbour' (line 51)
                neighbour_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'neighbour')
                # Setting the type of the member 'timestamp' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), neighbour_168, 'timestamp', TIMESTAMP_167)
                
                # Getting the type of 'neighbour' (line 52)
                neighbour_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'neighbour')
                # Obtaining the member 'color' of a type (line 52)
                color_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), neighbour_169, 'color')
                # Getting the type of 'EMPTY' (line 52)
                EMPTY_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'EMPTY')
                # Applying the binary operator '==' (line 52)
                result_eq_172 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 19), '==', color_170, EMPTY_171)
                
                # Testing if the type of an if condition is none (line 52)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 52, 16), result_eq_172):
                    
                    # Getting the type of 'neighbour' (line 54)
                    neighbour_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'neighbour')
                    # Obtaining the member 'color' of a type (line 54)
                    color_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), neighbour_179, 'color')
                    # Getting the type of 'self' (line 54)
                    self_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'self')
                    # Obtaining the member 'color' of a type (line 54)
                    color_182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 40), self_181, 'color')
                    # Applying the binary operator '==' (line 54)
                    result_eq_183 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 21), '==', color_180, color_182)
                    
                    # Testing if the type of an if condition is none (line 54)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 21), result_eq_183):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 54)
                        if_condition_184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 21), result_eq_183)
                        # Assigning a type to the variable 'if_condition_184' (line 54)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'if_condition_184', if_condition_184)
                        # SSA begins for if statement (line 54)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to count_liberties(...): (line 55)
                        # Processing the call arguments (line 55)
                        # Getting the type of 'reference' (line 55)
                        reference_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'reference', False)
                        # Processing the call keyword arguments (line 55)
                        kwargs_188 = {}
                        # Getting the type of 'neighbour' (line 55)
                        neighbour_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'neighbour', False)
                        # Obtaining the member 'count_liberties' of a type (line 55)
                        count_liberties_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), neighbour_185, 'count_liberties')
                        # Calling count_liberties(args, kwargs) (line 55)
                        count_liberties_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), count_liberties_186, *[reference_187], **kwargs_188)
                        
                        # SSA join for if statement (line 54)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 52)
                    if_condition_173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 16), result_eq_172)
                    # Assigning a type to the variable 'if_condition_173' (line 52)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'if_condition_173', if_condition_173)
                    # SSA begins for if statement (line 52)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'reference' (line 53)
                    reference_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'reference')
                    # Obtaining the member 'liberties' of a type (line 53)
                    liberties_175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), reference_174, 'liberties')
                    int_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 43), 'int')
                    # Applying the binary operator '+=' (line 53)
                    result_iadd_177 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 20), '+=', liberties_175, int_176)
                    # Getting the type of 'reference' (line 53)
                    reference_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'reference')
                    # Setting the type of the member 'liberties' of a type (line 53)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), reference_178, 'liberties', result_iadd_177)
                    
                    # SSA branch for the else part of an if statement (line 52)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'neighbour' (line 54)
                    neighbour_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'neighbour')
                    # Obtaining the member 'color' of a type (line 54)
                    color_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), neighbour_179, 'color')
                    # Getting the type of 'self' (line 54)
                    self_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'self')
                    # Obtaining the member 'color' of a type (line 54)
                    color_182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 40), self_181, 'color')
                    # Applying the binary operator '==' (line 54)
                    result_eq_183 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 21), '==', color_180, color_182)
                    
                    # Testing if the type of an if condition is none (line 54)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 21), result_eq_183):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 54)
                        if_condition_184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 21), result_eq_183)
                        # Assigning a type to the variable 'if_condition_184' (line 54)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'if_condition_184', if_condition_184)
                        # SSA begins for if statement (line 54)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to count_liberties(...): (line 55)
                        # Processing the call arguments (line 55)
                        # Getting the type of 'reference' (line 55)
                        reference_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'reference', False)
                        # Processing the call keyword arguments (line 55)
                        kwargs_188 = {}
                        # Getting the type of 'neighbour' (line 55)
                        neighbour_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'neighbour', False)
                        # Obtaining the member 'count_liberties' of a type (line 55)
                        count_liberties_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), neighbour_185, 'count_liberties')
                        # Calling count_liberties(args, kwargs) (line 55)
                        count_liberties_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), count_liberties_186, *[reference_187], **kwargs_188)
                        
                        # SSA join for if statement (line 54)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 52)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 50)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'count_liberties(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count_liberties' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_liberties'
        return stypy_return_type_190


    @norecursion
    def liberty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'liberty'
        module_type_store = module_type_store.open_function_context('liberty', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Square.liberty.__dict__.__setitem__('stypy_localization', localization)
        Square.liberty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Square.liberty.__dict__.__setitem__('stypy_type_store', module_type_store)
        Square.liberty.__dict__.__setitem__('stypy_function_name', 'Square.liberty')
        Square.liberty.__dict__.__setitem__('stypy_param_names_list', [])
        Square.liberty.__dict__.__setitem__('stypy_varargs_param_name', None)
        Square.liberty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Square.liberty.__dict__.__setitem__('stypy_call_defaults', defaults)
        Square.liberty.__dict__.__setitem__('stypy_call_varargs', varargs)
        Square.liberty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Square.liberty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.liberty', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'liberty', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'liberty(...)' code ##################

        
        # Assigning a Name to a Attribute (line 58):
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'TIMESTAMP' (line 58)
        TIMESTAMP_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 58)
        self_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'findstamp' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_192, 'findstamp', TIMESTAMP_191)
        
        # Getting the type of 'self' (line 59)
        self_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 59)
        neighbours_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 25), self_193, 'neighbours')
        # Assigning a type to the variable 'neighbours_194' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'neighbours_194', neighbours_194)
        # Testing if the for loop is going to be iterated (line 59)
        # Testing the type of a for loop iterable (line 59)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 8), neighbours_194)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 59, 8), neighbours_194):
            # Getting the type of the for loop variable (line 59)
            for_loop_var_195 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 8), neighbours_194)
            # Assigning a type to the variable 'neighbour' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'neighbour', for_loop_var_195)
            # SSA begins for a for statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'neighbour' (line 60)
            neighbour_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'neighbour')
            # Obtaining the member 'findstamp' of a type (line 60)
            findstamp_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), neighbour_196, 'findstamp')
            # Getting the type of 'TIMESTAMP' (line 60)
            TIMESTAMP_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'TIMESTAMP')
            # Applying the binary operator '!=' (line 60)
            result_ne_199 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 15), '!=', findstamp_197, TIMESTAMP_198)
            
            # Testing if the type of an if condition is none (line 60)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 12), result_ne_199):
                pass
            else:
                
                # Testing the type of an if condition (line 60)
                if_condition_200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 12), result_ne_199)
                # Assigning a type to the variable 'if_condition_200' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'if_condition_200', if_condition_200)
                # SSA begins for if statement (line 60)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 61):
                
                # Assigning a Name to a Attribute (line 61):
                # Getting the type of 'TIMESTAMP' (line 61)
                TIMESTAMP_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'TIMESTAMP')
                # Getting the type of 'neighbour' (line 61)
                neighbour_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'neighbour')
                # Setting the type of the member 'findstamp' of a type (line 61)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), neighbour_202, 'findstamp', TIMESTAMP_201)
                
                # Getting the type of 'neighbour' (line 62)
                neighbour_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'neighbour')
                # Obtaining the member 'color' of a type (line 62)
                color_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), neighbour_203, 'color')
                # Getting the type of 'EMPTY' (line 62)
                EMPTY_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'EMPTY')
                # Applying the binary operator '==' (line 62)
                result_eq_206 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 19), '==', color_204, EMPTY_205)
                
                # Testing if the type of an if condition is none (line 62)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 16), result_eq_206):
                    
                    # Getting the type of 'neighbour' (line 64)
                    neighbour_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'neighbour')
                    # Obtaining the member 'color' of a type (line 64)
                    color_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), neighbour_209, 'color')
                    # Getting the type of 'self' (line 64)
                    self_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'self')
                    # Obtaining the member 'color' of a type (line 64)
                    color_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), self_211, 'color')
                    # Applying the binary operator '==' (line 64)
                    result_eq_213 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 21), '==', color_210, color_212)
                    
                    # Testing if the type of an if condition is none (line 64)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 21), result_eq_213):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 64)
                        if_condition_214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 21), result_eq_213)
                        # Assigning a type to the variable 'if_condition_214' (line 64)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'if_condition_214', if_condition_214)
                        # SSA begins for if statement (line 64)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 65):
                        
                        # Assigning a Call to a Name (line 65):
                        
                        # Call to liberty(...): (line 65)
                        # Processing the call keyword arguments (line 65)
                        kwargs_217 = {}
                        # Getting the type of 'neighbour' (line 65)
                        neighbour_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'neighbour', False)
                        # Obtaining the member 'liberty' of a type (line 65)
                        liberty_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), neighbour_215, 'liberty')
                        # Calling liberty(args, kwargs) (line 65)
                        liberty_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), liberty_216, *[], **kwargs_217)
                        
                        # Assigning a type to the variable 'liberty' (line 65)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'liberty', liberty_call_result_218)
                        # Getting the type of 'liberty' (line 66)
                        liberty_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'liberty')
                        # Testing if the type of an if condition is none (line 66)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 20), liberty_219):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 66)
                            if_condition_220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 20), liberty_219)
                            # Assigning a type to the variable 'if_condition_220' (line 66)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'if_condition_220', if_condition_220)
                            # SSA begins for if statement (line 66)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # Getting the type of 'liberty' (line 67)
                            liberty_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'liberty')
                            # Assigning a type to the variable 'stypy_return_type' (line 67)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'stypy_return_type', liberty_221)
                            # SSA join for if statement (line 66)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 64)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 62)
                    if_condition_207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 16), result_eq_206)
                    # Assigning a type to the variable 'if_condition_207' (line 62)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'if_condition_207', if_condition_207)
                    # SSA begins for if statement (line 62)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'neighbour' (line 63)
                    neighbour_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'neighbour')
                    # Assigning a type to the variable 'stypy_return_type' (line 63)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'stypy_return_type', neighbour_208)
                    # SSA branch for the else part of an if statement (line 62)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'neighbour' (line 64)
                    neighbour_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'neighbour')
                    # Obtaining the member 'color' of a type (line 64)
                    color_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), neighbour_209, 'color')
                    # Getting the type of 'self' (line 64)
                    self_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'self')
                    # Obtaining the member 'color' of a type (line 64)
                    color_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), self_211, 'color')
                    # Applying the binary operator '==' (line 64)
                    result_eq_213 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 21), '==', color_210, color_212)
                    
                    # Testing if the type of an if condition is none (line 64)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 21), result_eq_213):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 64)
                        if_condition_214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 21), result_eq_213)
                        # Assigning a type to the variable 'if_condition_214' (line 64)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'if_condition_214', if_condition_214)
                        # SSA begins for if statement (line 64)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 65):
                        
                        # Assigning a Call to a Name (line 65):
                        
                        # Call to liberty(...): (line 65)
                        # Processing the call keyword arguments (line 65)
                        kwargs_217 = {}
                        # Getting the type of 'neighbour' (line 65)
                        neighbour_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'neighbour', False)
                        # Obtaining the member 'liberty' of a type (line 65)
                        liberty_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), neighbour_215, 'liberty')
                        # Calling liberty(args, kwargs) (line 65)
                        liberty_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), liberty_216, *[], **kwargs_217)
                        
                        # Assigning a type to the variable 'liberty' (line 65)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'liberty', liberty_call_result_218)
                        # Getting the type of 'liberty' (line 66)
                        liberty_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'liberty')
                        # Testing if the type of an if condition is none (line 66)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 20), liberty_219):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 66)
                            if_condition_220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 20), liberty_219)
                            # Assigning a type to the variable 'if_condition_220' (line 66)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'if_condition_220', if_condition_220)
                            # SSA begins for if statement (line 66)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # Getting the type of 'liberty' (line 67)
                            liberty_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'liberty')
                            # Assigning a type to the variable 'stypy_return_type' (line 67)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'stypy_return_type', liberty_221)
                            # SSA join for if statement (line 66)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 64)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 62)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 60)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'liberty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'liberty' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'liberty'
        return stypy_return_type_222


    @norecursion
    def move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'move'
        module_type_store = module_type_store.open_function_context('move', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Square.move.__dict__.__setitem__('stypy_localization', localization)
        Square.move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Square.move.__dict__.__setitem__('stypy_type_store', module_type_store)
        Square.move.__dict__.__setitem__('stypy_function_name', 'Square.move')
        Square.move.__dict__.__setitem__('stypy_param_names_list', ['color'])
        Square.move.__dict__.__setitem__('stypy_varargs_param_name', None)
        Square.move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Square.move.__dict__.__setitem__('stypy_call_defaults', defaults)
        Square.move.__dict__.__setitem__('stypy_call_varargs', varargs)
        Square.move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Square.move.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.move', ['color'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'move', localization, ['color'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'move(...)' code ##################

        # Marking variables as global (line 70)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 70, 8), 'TIMESTAMP')
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 70, 8), 'MOVES')
        
        # Getting the type of 'TIMESTAMP' (line 71)
        TIMESTAMP_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'TIMESTAMP')
        int_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'int')
        # Applying the binary operator '+=' (line 71)
        result_iadd_225 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 8), '+=', TIMESTAMP_223, int_224)
        # Assigning a type to the variable 'TIMESTAMP' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'TIMESTAMP', result_iadd_225)
        
        
        # Getting the type of 'MOVES' (line 72)
        MOVES_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'MOVES')
        int_227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'int')
        # Applying the binary operator '+=' (line 72)
        result_iadd_228 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 8), '+=', MOVES_226, int_227)
        # Assigning a type to the variable 'MOVES' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'MOVES', result_iadd_228)
        
        
        # Call to update(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'self', False)
        # Getting the type of 'color' (line 73)
        color_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'color', False)
        # Processing the call keyword arguments (line 73)
        kwargs_235 = {}
        # Getting the type of 'self' (line 73)
        self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'board' of a type (line 73)
        board_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_229, 'board')
        # Obtaining the member 'zobrist' of a type (line 73)
        zobrist_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), board_230, 'zobrist')
        # Obtaining the member 'update' of a type (line 73)
        update_232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), zobrist_231, 'update')
        # Calling update(args, kwargs) (line 73)
        update_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), update_232, *[self_233, color_234], **kwargs_235)
        
        
        # Assigning a Name to a Attribute (line 74):
        
        # Assigning a Name to a Attribute (line 74):
        # Getting the type of 'color' (line 74)
        color_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'color')
        # Getting the type of 'self' (line 74)
        self_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'color' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_238, 'color', color_237)
        
        # Assigning a Name to a Attribute (line 75):
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'self' (line 75)
        self_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'self')
        # Getting the type of 'self' (line 75)
        self_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'reference' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_240, 'reference', self_239)
        
        # Assigning a Num to a Attribute (line 76):
        
        # Assigning a Num to a Attribute (line 76):
        int_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'int')
        # Getting the type of 'self' (line 76)
        self_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'members' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_242, 'members', int_241)
        
        # Assigning a Name to a Attribute (line 77):
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'True' (line 77)
        True_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'True')
        # Getting the type of 'self' (line 77)
        self_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'used' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_244, 'used', True_243)
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'None' (line 78)
        None_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'None')
        # Getting the type of 'self' (line 78)
        self_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Obtaining the member 'board' of a type (line 78)
        board_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_246, 'board')
        # Setting the type of the member 'atari' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), board_247, 'atari', None_245)
        
        # Getting the type of 'self' (line 79)
        self_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 79)
        neighbours_249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), self_248, 'neighbours')
        # Assigning a type to the variable 'neighbours_249' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'neighbours_249', neighbours_249)
        # Testing if the for loop is going to be iterated (line 79)
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), neighbours_249)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 79, 8), neighbours_249):
            # Getting the type of the for loop variable (line 79)
            for_loop_var_250 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), neighbours_249)
            # Assigning a type to the variable 'neighbour' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'neighbour', for_loop_var_250)
            # SSA begins for a for statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'neighbour' (line 80)
            neighbour_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'neighbour')
            # Obtaining the member 'color' of a type (line 80)
            color_252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), neighbour_251, 'color')
            # Getting the type of 'EMPTY' (line 80)
            EMPTY_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'EMPTY')
            # Applying the binary operator '!=' (line 80)
            result_ne_254 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), '!=', color_252, EMPTY_253)
            
            # Testing if the type of an if condition is none (line 80)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 12), result_ne_254):
                pass
            else:
                
                # Testing the type of an if condition (line 80)
                if_condition_255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), result_ne_254)
                # Assigning a type to the variable 'if_condition_255' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_255', if_condition_255)
                # SSA begins for if statement (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 81):
                
                # Assigning a Call to a Name (line 81):
                
                # Call to find(...): (line 81)
                # Processing the call keyword arguments (line 81)
                # Getting the type of 'True' (line 81)
                True_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 54), 'True', False)
                keyword_259 = True_258
                kwargs_260 = {'update': keyword_259}
                # Getting the type of 'neighbour' (line 81)
                neighbour_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 32), 'neighbour', False)
                # Obtaining the member 'find' of a type (line 81)
                find_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 32), neighbour_256, 'find')
                # Calling find(args, kwargs) (line 81)
                find_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 81, 32), find_257, *[], **kwargs_260)
                
                # Assigning a type to the variable 'neighbour_ref' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'neighbour_ref', find_call_result_261)
                
                # Getting the type of 'neighbour_ref' (line 82)
                neighbour_ref_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'neighbour_ref')
                # Obtaining the member 'timestamp' of a type (line 82)
                timestamp_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 19), neighbour_ref_262, 'timestamp')
                # Getting the type of 'TIMESTAMP' (line 82)
                TIMESTAMP_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 46), 'TIMESTAMP')
                # Applying the binary operator '!=' (line 82)
                result_ne_265 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '!=', timestamp_263, TIMESTAMP_264)
                
                # Testing if the type of an if condition is none (line 82)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 16), result_ne_265):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 82)
                    if_condition_266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 16), result_ne_265)
                    # Assigning a type to the variable 'if_condition_266' (line 82)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'if_condition_266', if_condition_266)
                    # SSA begins for if statement (line 82)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Attribute (line 83):
                    
                    # Assigning a Name to a Attribute (line 83):
                    # Getting the type of 'TIMESTAMP' (line 83)
                    TIMESTAMP_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'TIMESTAMP')
                    # Getting the type of 'neighbour_ref' (line 83)
                    neighbour_ref_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'neighbour_ref')
                    # Setting the type of the member 'timestamp' of a type (line 83)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), neighbour_ref_268, 'timestamp', TIMESTAMP_267)
                    
                    # Getting the type of 'neighbour' (line 84)
                    neighbour_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'neighbour')
                    # Obtaining the member 'color' of a type (line 84)
                    color_270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), neighbour_269, 'color')
                    # Getting the type of 'color' (line 84)
                    color_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'color')
                    # Applying the binary operator '==' (line 84)
                    result_eq_272 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 23), '==', color_270, color_271)
                    
                    # Testing if the type of an if condition is none (line 84)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 84, 20), result_eq_272):
                        
                        # Getting the type of 'neighbour_ref' (line 88)
                        neighbour_ref_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'neighbour_ref')
                        # Obtaining the member 'liberties' of a type (line 88)
                        liberties_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), neighbour_ref_282, 'liberties')
                        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 51), 'int')
                        # Applying the binary operator '-=' (line 88)
                        result_isub_285 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 24), '-=', liberties_283, int_284)
                        # Getting the type of 'neighbour_ref' (line 88)
                        neighbour_ref_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'neighbour_ref')
                        # Setting the type of the member 'liberties' of a type (line 88)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), neighbour_ref_286, 'liberties', result_isub_285)
                        
                        
                        # Getting the type of 'neighbour_ref' (line 89)
                        neighbour_ref_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'neighbour_ref')
                        # Obtaining the member 'liberties' of a type (line 89)
                        liberties_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), neighbour_ref_287, 'liberties')
                        int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 54), 'int')
                        # Applying the binary operator '==' (line 89)
                        result_eq_290 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 27), '==', liberties_288, int_289)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 24), result_eq_290):
                            
                            # Getting the type of 'neighbour_ref' (line 91)
                            neighbour_ref_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'neighbour_ref')
                            # Obtaining the member 'liberties' of a type (line 91)
                            liberties_300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), neighbour_ref_299, 'liberties')
                            int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'int')
                            # Applying the binary operator '==' (line 91)
                            result_eq_302 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '==', liberties_300, int_301)
                            
                            # Testing if the type of an if condition is none (line 91)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 91)
                                if_condition_303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302)
                                # Assigning a type to the variable 'if_condition_303' (line 91)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'if_condition_303', if_condition_303)
                                # SSA begins for if statement (line 91)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Attribute (line 92):
                                
                                # Assigning a Name to a Attribute (line 92):
                                # Getting the type of 'neighbour_ref' (line 92)
                                neighbour_ref_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'neighbour_ref')
                                # Getting the type of 'self' (line 92)
                                self_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'self')
                                # Obtaining the member 'board' of a type (line 92)
                                board_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), self_305, 'board')
                                # Setting the type of the member 'atari' of a type (line 92)
                                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), board_306, 'atari', neighbour_ref_304)
                                # SSA join for if statement (line 91)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 24), result_eq_290)
                            # Assigning a type to the variable 'if_condition_291' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'if_condition_291', if_condition_291)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to remove(...): (line 90)
                            # Processing the call arguments (line 90)
                            # Getting the type of 'neighbour_ref' (line 90)
                            neighbour_ref_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'neighbour_ref', False)
                            # Processing the call keyword arguments (line 90)
                            # Getting the type of 'True' (line 90)
                            True_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 71), 'True', False)
                            keyword_296 = True_295
                            kwargs_297 = {'update': keyword_296}
                            # Getting the type of 'neighbour_ref' (line 90)
                            neighbour_ref_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'neighbour_ref', False)
                            # Obtaining the member 'remove' of a type (line 90)
                            remove_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), neighbour_ref_292, 'remove')
                            # Calling remove(args, kwargs) (line 90)
                            remove_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 90, 28), remove_293, *[neighbour_ref_294], **kwargs_297)
                            
                            # SSA branch for the else part of an if statement (line 89)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'neighbour_ref' (line 91)
                            neighbour_ref_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'neighbour_ref')
                            # Obtaining the member 'liberties' of a type (line 91)
                            liberties_300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), neighbour_ref_299, 'liberties')
                            int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'int')
                            # Applying the binary operator '==' (line 91)
                            result_eq_302 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '==', liberties_300, int_301)
                            
                            # Testing if the type of an if condition is none (line 91)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 91)
                                if_condition_303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302)
                                # Assigning a type to the variable 'if_condition_303' (line 91)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'if_condition_303', if_condition_303)
                                # SSA begins for if statement (line 91)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Attribute (line 92):
                                
                                # Assigning a Name to a Attribute (line 92):
                                # Getting the type of 'neighbour_ref' (line 92)
                                neighbour_ref_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'neighbour_ref')
                                # Getting the type of 'self' (line 92)
                                self_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'self')
                                # Obtaining the member 'board' of a type (line 92)
                                board_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), self_305, 'board')
                                # Setting the type of the member 'atari' of a type (line 92)
                                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), board_306, 'atari', neighbour_ref_304)
                                # SSA join for if statement (line 91)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 84)
                        if_condition_273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 20), result_eq_272)
                        # Assigning a type to the variable 'if_condition_273' (line 84)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'if_condition_273', if_condition_273)
                        # SSA begins for if statement (line 84)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 85):
                        
                        # Assigning a Name to a Attribute (line 85):
                        # Getting the type of 'self' (line 85)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 50), 'self')
                        # Getting the type of 'neighbour_ref' (line 85)
                        neighbour_ref_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'neighbour_ref')
                        # Setting the type of the member 'reference' of a type (line 85)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), neighbour_ref_275, 'reference', self_274)
                        
                        # Getting the type of 'self' (line 86)
                        self_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'self')
                        # Obtaining the member 'members' of a type (line 86)
                        members_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), self_276, 'members')
                        # Getting the type of 'neighbour_ref' (line 86)
                        neighbour_ref_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 40), 'neighbour_ref')
                        # Obtaining the member 'members' of a type (line 86)
                        members_279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 40), neighbour_ref_278, 'members')
                        # Applying the binary operator '+=' (line 86)
                        result_iadd_280 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 24), '+=', members_277, members_279)
                        # Getting the type of 'self' (line 86)
                        self_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'self')
                        # Setting the type of the member 'members' of a type (line 86)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), self_281, 'members', result_iadd_280)
                        
                        # SSA branch for the else part of an if statement (line 84)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'neighbour_ref' (line 88)
                        neighbour_ref_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'neighbour_ref')
                        # Obtaining the member 'liberties' of a type (line 88)
                        liberties_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), neighbour_ref_282, 'liberties')
                        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 51), 'int')
                        # Applying the binary operator '-=' (line 88)
                        result_isub_285 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 24), '-=', liberties_283, int_284)
                        # Getting the type of 'neighbour_ref' (line 88)
                        neighbour_ref_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'neighbour_ref')
                        # Setting the type of the member 'liberties' of a type (line 88)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), neighbour_ref_286, 'liberties', result_isub_285)
                        
                        
                        # Getting the type of 'neighbour_ref' (line 89)
                        neighbour_ref_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'neighbour_ref')
                        # Obtaining the member 'liberties' of a type (line 89)
                        liberties_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), neighbour_ref_287, 'liberties')
                        int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 54), 'int')
                        # Applying the binary operator '==' (line 89)
                        result_eq_290 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 27), '==', liberties_288, int_289)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 24), result_eq_290):
                            
                            # Getting the type of 'neighbour_ref' (line 91)
                            neighbour_ref_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'neighbour_ref')
                            # Obtaining the member 'liberties' of a type (line 91)
                            liberties_300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), neighbour_ref_299, 'liberties')
                            int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'int')
                            # Applying the binary operator '==' (line 91)
                            result_eq_302 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '==', liberties_300, int_301)
                            
                            # Testing if the type of an if condition is none (line 91)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 91)
                                if_condition_303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302)
                                # Assigning a type to the variable 'if_condition_303' (line 91)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'if_condition_303', if_condition_303)
                                # SSA begins for if statement (line 91)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Attribute (line 92):
                                
                                # Assigning a Name to a Attribute (line 92):
                                # Getting the type of 'neighbour_ref' (line 92)
                                neighbour_ref_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'neighbour_ref')
                                # Getting the type of 'self' (line 92)
                                self_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'self')
                                # Obtaining the member 'board' of a type (line 92)
                                board_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), self_305, 'board')
                                # Setting the type of the member 'atari' of a type (line 92)
                                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), board_306, 'atari', neighbour_ref_304)
                                # SSA join for if statement (line 91)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 24), result_eq_290)
                            # Assigning a type to the variable 'if_condition_291' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'if_condition_291', if_condition_291)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to remove(...): (line 90)
                            # Processing the call arguments (line 90)
                            # Getting the type of 'neighbour_ref' (line 90)
                            neighbour_ref_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'neighbour_ref', False)
                            # Processing the call keyword arguments (line 90)
                            # Getting the type of 'True' (line 90)
                            True_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 71), 'True', False)
                            keyword_296 = True_295
                            kwargs_297 = {'update': keyword_296}
                            # Getting the type of 'neighbour_ref' (line 90)
                            neighbour_ref_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'neighbour_ref', False)
                            # Obtaining the member 'remove' of a type (line 90)
                            remove_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), neighbour_ref_292, 'remove')
                            # Calling remove(args, kwargs) (line 90)
                            remove_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 90, 28), remove_293, *[neighbour_ref_294], **kwargs_297)
                            
                            # SSA branch for the else part of an if statement (line 89)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'neighbour_ref' (line 91)
                            neighbour_ref_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'neighbour_ref')
                            # Obtaining the member 'liberties' of a type (line 91)
                            liberties_300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), neighbour_ref_299, 'liberties')
                            int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'int')
                            # Applying the binary operator '==' (line 91)
                            result_eq_302 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '==', liberties_300, int_301)
                            
                            # Testing if the type of an if condition is none (line 91)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 91)
                                if_condition_303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_302)
                                # Assigning a type to the variable 'if_condition_303' (line 91)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'if_condition_303', if_condition_303)
                                # SSA begins for if statement (line 91)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Attribute (line 92):
                                
                                # Assigning a Name to a Attribute (line 92):
                                # Getting the type of 'neighbour_ref' (line 92)
                                neighbour_ref_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'neighbour_ref')
                                # Getting the type of 'self' (line 92)
                                self_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'self')
                                # Obtaining the member 'board' of a type (line 92)
                                board_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), self_305, 'board')
                                # Setting the type of the member 'atari' of a type (line 92)
                                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), board_306, 'atari', neighbour_ref_304)
                                # SSA join for if statement (line 91)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 84)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 82)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'TIMESTAMP' (line 93)
        TIMESTAMP_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'TIMESTAMP')
        int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'int')
        # Applying the binary operator '+=' (line 93)
        result_iadd_309 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 8), '+=', TIMESTAMP_307, int_308)
        # Assigning a type to the variable 'TIMESTAMP' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'TIMESTAMP', result_iadd_309)
        
        
        # Call to count_liberties(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_312 = {}
        # Getting the type of 'self' (line 94)
        self_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'count_liberties' of a type (line 94)
        count_liberties_311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_310, 'count_liberties')
        # Calling count_liberties(args, kwargs) (line 94)
        count_liberties_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), count_liberties_311, *[], **kwargs_312)
        
        
        # Call to add(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_318 = {}
        # Getting the type of 'self' (line 95)
        self_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self', False)
        # Obtaining the member 'board' of a type (line 95)
        board_315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_314, 'board')
        # Obtaining the member 'zobrist' of a type (line 95)
        zobrist_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), board_315, 'zobrist')
        # Obtaining the member 'add' of a type (line 95)
        add_317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), zobrist_316, 'add')
        # Calling add(args, kwargs) (line 95)
        add_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), add_317, *[], **kwargs_318)
        
        
        # ################# End of 'move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_320)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move'
        return stypy_return_type_320


    @norecursion
    def remove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 97)
        True_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'True')
        defaults = [True_321]
        # Create a new context for function 'remove'
        module_type_store = module_type_store.open_function_context('remove', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Square.remove.__dict__.__setitem__('stypy_localization', localization)
        Square.remove.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Square.remove.__dict__.__setitem__('stypy_type_store', module_type_store)
        Square.remove.__dict__.__setitem__('stypy_function_name', 'Square.remove')
        Square.remove.__dict__.__setitem__('stypy_param_names_list', ['reference', 'update'])
        Square.remove.__dict__.__setitem__('stypy_varargs_param_name', None)
        Square.remove.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Square.remove.__dict__.__setitem__('stypy_call_defaults', defaults)
        Square.remove.__dict__.__setitem__('stypy_call_varargs', varargs)
        Square.remove.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Square.remove.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.remove', ['reference', 'update'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove', localization, ['reference', 'update'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove(...)' code ##################

        # Marking variables as global (line 98)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 98, 8), 'REMOVESTAMP')
        
        # Getting the type of 'REMOVESTAMP' (line 99)
        REMOVESTAMP_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'REMOVESTAMP')
        int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'int')
        # Applying the binary operator '+=' (line 99)
        result_iadd_324 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 8), '+=', REMOVESTAMP_322, int_323)
        # Assigning a type to the variable 'REMOVESTAMP' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'REMOVESTAMP', result_iadd_324)
        
        
        # Assigning a Name to a Name (line 100):
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'REMOVESTAMP' (line 100)
        REMOVESTAMP_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'REMOVESTAMP')
        # Assigning a type to the variable 'removestamp' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'removestamp', REMOVESTAMP_325)
        
        # Call to update(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 34), 'self', False)
        # Getting the type of 'EMPTY' (line 101)
        EMPTY_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'EMPTY', False)
        # Processing the call keyword arguments (line 101)
        kwargs_332 = {}
        # Getting the type of 'self' (line 101)
        self_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self', False)
        # Obtaining the member 'board' of a type (line 101)
        board_327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_326, 'board')
        # Obtaining the member 'zobrist' of a type (line 101)
        zobrist_328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), board_327, 'zobrist')
        # Obtaining the member 'update' of a type (line 101)
        update_329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), zobrist_328, 'update')
        # Calling update(args, kwargs) (line 101)
        update_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), update_329, *[self_330, EMPTY_331], **kwargs_332)
        
        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'TIMESTAMP' (line 102)
        TIMESTAMP_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 'TIMESTAMP')
        # Getting the type of 'self' (line 102)
        self_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'timestamp2' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_335, 'timestamp2', TIMESTAMP_334)
        # Getting the type of 'update' (line 103)
        update_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'update')
        # Testing if the type of an if condition is none (line 103)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 103, 8), update_336):
            pass
        else:
            
            # Testing the type of an if condition (line 103)
            if_condition_337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), update_336)
            # Assigning a type to the variable 'if_condition_337' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_337', if_condition_337)
            # SSA begins for if statement (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 104):
            
            # Assigning a Name to a Attribute (line 104):
            # Getting the type of 'EMPTY' (line 104)
            EMPTY_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'EMPTY')
            # Getting the type of 'self' (line 104)
            self_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self')
            # Setting the type of the member 'color' of a type (line 104)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_339, 'color', EMPTY_338)
            
            # Call to add(...): (line 105)
            # Processing the call arguments (line 105)
            # Getting the type of 'self' (line 105)
            self_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 36), 'self', False)
            # Obtaining the member 'pos' of a type (line 105)
            pos_345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 36), self_344, 'pos')
            # Processing the call keyword arguments (line 105)
            kwargs_346 = {}
            # Getting the type of 'self' (line 105)
            self_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self', False)
            # Obtaining the member 'board' of a type (line 105)
            board_341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_340, 'board')
            # Obtaining the member 'emptyset' of a type (line 105)
            emptyset_342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), board_341, 'emptyset')
            # Obtaining the member 'add' of a type (line 105)
            add_343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), emptyset_342, 'add')
            # Calling add(args, kwargs) (line 105)
            add_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), add_343, *[pos_345], **kwargs_346)
            
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'update' (line 110)
        update_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'update')
        # Testing if the type of an if condition is none (line 110)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 110, 8), update_348):
            pass
        else:
            
            # Testing the type of an if condition (line 110)
            if_condition_349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), update_348)
            # Assigning a type to the variable 'if_condition_349' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_349', if_condition_349)
            # SSA begins for if statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'self' (line 111)
            self_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'self')
            # Obtaining the member 'neighbours' of a type (line 111)
            neighbours_351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 29), self_350, 'neighbours')
            # Assigning a type to the variable 'neighbours_351' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'neighbours_351', neighbours_351)
            # Testing if the for loop is going to be iterated (line 111)
            # Testing the type of a for loop iterable (line 111)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), neighbours_351)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 12), neighbours_351):
                # Getting the type of the for loop variable (line 111)
                for_loop_var_352 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), neighbours_351)
                # Assigning a type to the variable 'neighbour' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'neighbour', for_loop_var_352)
                # SSA begins for a for statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'neighbour' (line 112)
                neighbour_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'neighbour')
                # Obtaining the member 'color' of a type (line 112)
                color_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), neighbour_353, 'color')
                # Getting the type of 'EMPTY' (line 112)
                EMPTY_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'EMPTY')
                # Applying the binary operator '!=' (line 112)
                result_ne_356 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), '!=', color_354, EMPTY_355)
                
                # Testing if the type of an if condition is none (line 112)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 16), result_ne_356):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 112)
                    if_condition_357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), result_ne_356)
                    # Assigning a type to the variable 'if_condition_357' (line 112)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_357', if_condition_357)
                    # SSA begins for if statement (line 112)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 113):
                    
                    # Assigning a Call to a Name (line 113):
                    
                    # Call to find(...): (line 113)
                    # Processing the call arguments (line 113)
                    # Getting the type of 'update' (line 113)
                    update_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 51), 'update', False)
                    # Processing the call keyword arguments (line 113)
                    kwargs_361 = {}
                    # Getting the type of 'neighbour' (line 113)
                    neighbour_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 36), 'neighbour', False)
                    # Obtaining the member 'find' of a type (line 113)
                    find_359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 36), neighbour_358, 'find')
                    # Calling find(args, kwargs) (line 113)
                    find_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 113, 36), find_359, *[update_360], **kwargs_361)
                    
                    # Assigning a type to the variable 'neighbour_ref' (line 113)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'neighbour_ref', find_call_result_362)
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'neighbour_ref' (line 114)
                    neighbour_ref_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'neighbour_ref')
                    # Obtaining the member 'pos' of a type (line 114)
                    pos_364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 23), neighbour_ref_363, 'pos')
                    # Getting the type of 'self' (line 114)
                    self_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 44), 'self')
                    # Obtaining the member 'pos' of a type (line 114)
                    pos_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 44), self_365, 'pos')
                    # Applying the binary operator '!=' (line 114)
                    result_ne_367 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 23), '!=', pos_364, pos_366)
                    
                    
                    # Getting the type of 'neighbour_ref' (line 114)
                    neighbour_ref_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 57), 'neighbour_ref')
                    # Obtaining the member 'removestamp' of a type (line 114)
                    removestamp_369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 57), neighbour_ref_368, 'removestamp')
                    # Getting the type of 'removestamp' (line 114)
                    removestamp_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 86), 'removestamp')
                    # Applying the binary operator '!=' (line 114)
                    result_ne_371 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 57), '!=', removestamp_369, removestamp_370)
                    
                    # Applying the binary operator 'and' (line 114)
                    result_and_keyword_372 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 23), 'and', result_ne_367, result_ne_371)
                    
                    # Testing if the type of an if condition is none (line 114)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 114, 20), result_and_keyword_372):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 114)
                        if_condition_373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 20), result_and_keyword_372)
                        # Assigning a type to the variable 'if_condition_373' (line 114)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'if_condition_373', if_condition_373)
                        # SSA begins for if statement (line 114)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 115):
                        
                        # Assigning a Name to a Attribute (line 115):
                        # Getting the type of 'removestamp' (line 115)
                        removestamp_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 52), 'removestamp')
                        # Getting the type of 'neighbour_ref' (line 115)
                        neighbour_ref_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'neighbour_ref')
                        # Setting the type of the member 'removestamp' of a type (line 115)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 24), neighbour_ref_375, 'removestamp', removestamp_374)
                        
                        # Getting the type of 'neighbour_ref' (line 116)
                        neighbour_ref_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'neighbour_ref')
                        # Obtaining the member 'liberties' of a type (line 116)
                        liberties_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), neighbour_ref_376, 'liberties')
                        int_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 51), 'int')
                        # Applying the binary operator '+=' (line 116)
                        result_iadd_379 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 24), '+=', liberties_377, int_378)
                        # Getting the type of 'neighbour_ref' (line 116)
                        neighbour_ref_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'neighbour_ref')
                        # Setting the type of the member 'liberties' of a type (line 116)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), neighbour_ref_380, 'liberties', result_iadd_379)
                        
                        # SSA join for if statement (line 114)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 112)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 117)
        self_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 117)
        neighbours_382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), self_381, 'neighbours')
        # Assigning a type to the variable 'neighbours_382' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'neighbours_382', neighbours_382)
        # Testing if the for loop is going to be iterated (line 117)
        # Testing the type of a for loop iterable (line 117)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 8), neighbours_382)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 117, 8), neighbours_382):
            # Getting the type of the for loop variable (line 117)
            for_loop_var_383 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 8), neighbours_382)
            # Assigning a type to the variable 'neighbour' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'neighbour', for_loop_var_383)
            # SSA begins for a for statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'neighbour' (line 118)
            neighbour_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'neighbour')
            # Obtaining the member 'color' of a type (line 118)
            color_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), neighbour_384, 'color')
            # Getting the type of 'EMPTY' (line 118)
            EMPTY_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'EMPTY')
            # Applying the binary operator '!=' (line 118)
            result_ne_387 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '!=', color_385, EMPTY_386)
            
            # Testing if the type of an if condition is none (line 118)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 118, 12), result_ne_387):
                pass
            else:
                
                # Testing the type of an if condition (line 118)
                if_condition_388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 12), result_ne_387)
                # Assigning a type to the variable 'if_condition_388' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'if_condition_388', if_condition_388)
                # SSA begins for if statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 119):
                
                # Assigning a Call to a Name (line 119):
                
                # Call to find(...): (line 119)
                # Processing the call arguments (line 119)
                # Getting the type of 'update' (line 119)
                update_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'update', False)
                # Processing the call keyword arguments (line 119)
                kwargs_392 = {}
                # Getting the type of 'neighbour' (line 119)
                neighbour_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 32), 'neighbour', False)
                # Obtaining the member 'find' of a type (line 119)
                find_390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 32), neighbour_389, 'find')
                # Calling find(args, kwargs) (line 119)
                find_call_result_393 = invoke(stypy.reporting.localization.Localization(__file__, 119, 32), find_390, *[update_391], **kwargs_392)
                
                # Assigning a type to the variable 'neighbour_ref' (line 119)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'neighbour_ref', find_call_result_393)
                
                # Evaluating a boolean operation
                
                # Getting the type of 'neighbour_ref' (line 120)
                neighbour_ref_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'neighbour_ref')
                # Obtaining the member 'pos' of a type (line 120)
                pos_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), neighbour_ref_394, 'pos')
                # Getting the type of 'reference' (line 120)
                reference_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'reference')
                # Obtaining the member 'pos' of a type (line 120)
                pos_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 40), reference_396, 'pos')
                # Applying the binary operator '==' (line 120)
                result_eq_398 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 19), '==', pos_395, pos_397)
                
                
                # Getting the type of 'neighbour' (line 120)
                neighbour_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 58), 'neighbour')
                # Obtaining the member 'timestamp2' of a type (line 120)
                timestamp2_400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 58), neighbour_399, 'timestamp2')
                # Getting the type of 'TIMESTAMP' (line 120)
                TIMESTAMP_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 82), 'TIMESTAMP')
                # Applying the binary operator '!=' (line 120)
                result_ne_402 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 58), '!=', timestamp2_400, TIMESTAMP_401)
                
                # Applying the binary operator 'and' (line 120)
                result_and_keyword_403 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 19), 'and', result_eq_398, result_ne_402)
                
                # Testing if the type of an if condition is none (line 120)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 120, 16), result_and_keyword_403):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 120)
                    if_condition_404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 16), result_and_keyword_403)
                    # Assigning a type to the variable 'if_condition_404' (line 120)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'if_condition_404', if_condition_404)
                    # SSA begins for if statement (line 120)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to remove(...): (line 121)
                    # Processing the call arguments (line 121)
                    # Getting the type of 'reference' (line 121)
                    reference_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 37), 'reference', False)
                    # Getting the type of 'update' (line 121)
                    update_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'update', False)
                    # Processing the call keyword arguments (line 121)
                    kwargs_409 = {}
                    # Getting the type of 'neighbour' (line 121)
                    neighbour_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'neighbour', False)
                    # Obtaining the member 'remove' of a type (line 121)
                    remove_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 20), neighbour_405, 'remove')
                    # Calling remove(args, kwargs) (line 121)
                    remove_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 121, 20), remove_406, *[reference_407, update_408], **kwargs_409)
                    
                    # SSA join for if statement (line 120)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 118)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'remove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_411)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_411


    @norecursion
    def find(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 123)
        False_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'False')
        defaults = [False_412]
        # Create a new context for function 'find'
        module_type_store = module_type_store.open_function_context('find', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Square.find.__dict__.__setitem__('stypy_localization', localization)
        Square.find.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Square.find.__dict__.__setitem__('stypy_type_store', module_type_store)
        Square.find.__dict__.__setitem__('stypy_function_name', 'Square.find')
        Square.find.__dict__.__setitem__('stypy_param_names_list', ['update'])
        Square.find.__dict__.__setitem__('stypy_varargs_param_name', None)
        Square.find.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Square.find.__dict__.__setitem__('stypy_call_defaults', defaults)
        Square.find.__dict__.__setitem__('stypy_call_varargs', varargs)
        Square.find.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Square.find.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.find', ['update'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find', localization, ['update'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find(...)' code ##################

        
        # Assigning a Attribute to a Name (line 124):
        
        # Assigning a Attribute to a Name (line 124):
        # Getting the type of 'self' (line 124)
        self_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'self')
        # Obtaining the member 'reference' of a type (line 124)
        reference_414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), self_413, 'reference')
        # Assigning a type to the variable 'reference' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'reference', reference_414)
        
        # Getting the type of 'reference' (line 125)
        reference_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'reference')
        # Obtaining the member 'pos' of a type (line 125)
        pos_416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), reference_415, 'pos')
        # Getting the type of 'self' (line 125)
        self_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'self')
        # Obtaining the member 'pos' of a type (line 125)
        pos_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 28), self_417, 'pos')
        # Applying the binary operator '!=' (line 125)
        result_ne_419 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), '!=', pos_416, pos_418)
        
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), result_ne_419):
            pass
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_ne_419)
            # Assigning a type to the variable 'if_condition_420' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_420', if_condition_420)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 126):
            
            # Assigning a Call to a Name (line 126):
            
            # Call to find(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'update' (line 126)
            update_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'update', False)
            # Processing the call keyword arguments (line 126)
            kwargs_424 = {}
            # Getting the type of 'reference' (line 126)
            reference_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'reference', False)
            # Obtaining the member 'find' of a type (line 126)
            find_422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), reference_421, 'find')
            # Calling find(args, kwargs) (line 126)
            find_call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), find_422, *[update_423], **kwargs_424)
            
            # Assigning a type to the variable 'reference' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'reference', find_call_result_425)
            # Getting the type of 'update' (line 127)
            update_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'update')
            # Testing if the type of an if condition is none (line 127)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 12), update_426):
                pass
            else:
                
                # Testing the type of an if condition (line 127)
                if_condition_427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), update_426)
                # Assigning a type to the variable 'if_condition_427' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_427', if_condition_427)
                # SSA begins for if statement (line 127)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 128):
                
                # Assigning a Name to a Attribute (line 128):
                # Getting the type of 'reference' (line 128)
                reference_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 33), 'reference')
                # Getting the type of 'self' (line 128)
                self_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'self')
                # Setting the type of the member 'reference' of a type (line 128)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), self_429, 'reference', reference_428)
                # SSA join for if statement (line 127)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'reference' (line 129)
        reference_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'reference')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', reference_430)
        
        # ################# End of 'find(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find'
        return stypy_return_type_431


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Square.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Square.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Square.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Square.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Square.stypy__repr__')
        Square.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Square.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Square.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Square.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Square.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Square.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Square.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Square.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to repr(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Call to to_xy(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 26), 'self', False)
        # Obtaining the member 'pos' of a type (line 132)
        pos_435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 26), self_434, 'pos')
        # Processing the call keyword arguments (line 132)
        kwargs_436 = {}
        # Getting the type of 'to_xy' (line 132)
        to_xy_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'to_xy', False)
        # Calling to_xy(args, kwargs) (line 132)
        to_xy_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 132, 20), to_xy_433, *[pos_435], **kwargs_436)
        
        # Processing the call keyword arguments (line 132)
        kwargs_438 = {}
        # Getting the type of 'repr' (line 132)
        repr_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'repr', False)
        # Calling repr(args, kwargs) (line 132)
        repr_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), repr_432, *[to_xy_call_result_437], **kwargs_438)
        
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', repr_call_result_439)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_440


# Assigning a type to the variable 'Square' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'Square', Square)
# Declaration of the 'EmptySet' class

class EmptySet:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EmptySet.__init__', ['board'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['board'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 137):
        
        # Assigning a Name to a Attribute (line 137):
        # Getting the type of 'board' (line 137)
        board_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'board')
        # Getting the type of 'self' (line 137)
        self_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member 'board' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_442, 'board', board_441)
        
        # Assigning a Call to a Attribute (line 138):
        
        # Assigning a Call to a Attribute (line 138):
        
        # Call to range(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'SIZE' (line 138)
        SIZE_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'SIZE', False)
        # Getting the type of 'SIZE' (line 138)
        SIZE_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 36), 'SIZE', False)
        # Applying the binary operator '*' (line 138)
        result_mul_446 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 29), '*', SIZE_444, SIZE_445)
        
        # Processing the call keyword arguments (line 138)
        kwargs_447 = {}
        # Getting the type of 'range' (line 138)
        range_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'range', False)
        # Calling range(args, kwargs) (line 138)
        range_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), range_443, *[result_mul_446], **kwargs_447)
        
        # Getting the type of 'self' (line 138)
        self_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
        # Setting the type of the member 'empties' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_449, 'empties', range_call_result_448)
        
        # Assigning a Call to a Attribute (line 139):
        
        # Assigning a Call to a Attribute (line 139):
        
        # Call to range(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'SIZE' (line 139)
        SIZE_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'SIZE', False)
        # Getting the type of 'SIZE' (line 139)
        SIZE_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'SIZE', False)
        # Applying the binary operator '*' (line 139)
        result_mul_453 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), '*', SIZE_451, SIZE_452)
        
        # Processing the call keyword arguments (line 139)
        kwargs_454 = {}
        # Getting the type of 'range' (line 139)
        range_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'range', False)
        # Calling range(args, kwargs) (line 139)
        range_call_result_455 = invoke(stypy.reporting.localization.Localization(__file__, 139, 25), range_450, *[result_mul_453], **kwargs_454)
        
        # Getting the type of 'self' (line 139)
        self_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member 'empty_pos' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_456, 'empty_pos', range_call_result_455)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def random_choice(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'random_choice'
        module_type_store = module_type_store.open_function_context('random_choice', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EmptySet.random_choice.__dict__.__setitem__('stypy_localization', localization)
        EmptySet.random_choice.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EmptySet.random_choice.__dict__.__setitem__('stypy_type_store', module_type_store)
        EmptySet.random_choice.__dict__.__setitem__('stypy_function_name', 'EmptySet.random_choice')
        EmptySet.random_choice.__dict__.__setitem__('stypy_param_names_list', [])
        EmptySet.random_choice.__dict__.__setitem__('stypy_varargs_param_name', None)
        EmptySet.random_choice.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EmptySet.random_choice.__dict__.__setitem__('stypy_call_defaults', defaults)
        EmptySet.random_choice.__dict__.__setitem__('stypy_call_varargs', varargs)
        EmptySet.random_choice.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EmptySet.random_choice.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EmptySet.random_choice', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'random_choice', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'random_choice(...)' code ##################

        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to len(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'self', False)
        # Obtaining the member 'empties' of a type (line 142)
        empties_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 22), self_458, 'empties')
        # Processing the call keyword arguments (line 142)
        kwargs_460 = {}
        # Getting the type of 'len' (line 142)
        len_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'len', False)
        # Calling len(args, kwargs) (line 142)
        len_call_result_461 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), len_457, *[empties_459], **kwargs_460)
        
        # Assigning a type to the variable 'choices' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'choices', len_call_result_461)
        
        # Getting the type of 'choices' (line 143)
        choices_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), 'choices')
        # Assigning a type to the variable 'choices_462' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'choices_462', choices_462)
        # Testing if the while is going to be iterated (line 143)
        # Testing the type of an if condition (line 143)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 8), choices_462)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 143, 8), choices_462):
            # SSA begins for while statement (line 143)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 144):
            
            # Assigning a Call to a Name (line 144):
            
            # Call to int(...): (line 144)
            # Processing the call arguments (line 144)
            
            # Call to random(...): (line 144)
            # Processing the call keyword arguments (line 144)
            kwargs_466 = {}
            # Getting the type of 'random' (line 144)
            random_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'random', False)
            # Obtaining the member 'random' of a type (line 144)
            random_465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), random_464, 'random')
            # Calling random(args, kwargs) (line 144)
            random_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), random_465, *[], **kwargs_466)
            
            # Getting the type of 'choices' (line 144)
            choices_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 38), 'choices', False)
            # Applying the binary operator '*' (line 144)
            result_mul_469 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), '*', random_call_result_467, choices_468)
            
            # Processing the call keyword arguments (line 144)
            kwargs_470 = {}
            # Getting the type of 'int' (line 144)
            int_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'int', False)
            # Calling int(args, kwargs) (line 144)
            int_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), int_463, *[result_mul_469], **kwargs_470)
            
            # Assigning a type to the variable 'i' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'i', int_call_result_471)
            
            # Assigning a Subscript to a Name (line 145):
            
            # Assigning a Subscript to a Name (line 145):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 145)
            i_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 31), 'i')
            # Getting the type of 'self' (line 145)
            self_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'self')
            # Obtaining the member 'empties' of a type (line 145)
            empties_474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), self_473, 'empties')
            # Obtaining the member '__getitem__' of a type (line 145)
            getitem___475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), empties_474, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 145)
            subscript_call_result_476 = invoke(stypy.reporting.localization.Localization(__file__, 145, 18), getitem___475, i_472)
            
            # Assigning a type to the variable 'pos' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'pos', subscript_call_result_476)
            
            # Call to useful(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'pos' (line 146)
            pos_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'pos', False)
            # Processing the call keyword arguments (line 146)
            kwargs_481 = {}
            # Getting the type of 'self' (line 146)
            self_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'self', False)
            # Obtaining the member 'board' of a type (line 146)
            board_478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), self_477, 'board')
            # Obtaining the member 'useful' of a type (line 146)
            useful_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), board_478, 'useful')
            # Calling useful(args, kwargs) (line 146)
            useful_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 146, 15), useful_479, *[pos_480], **kwargs_481)
            
            # Testing if the type of an if condition is none (line 146)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 12), useful_call_result_482):
                pass
            else:
                
                # Testing the type of an if condition (line 146)
                if_condition_483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 12), useful_call_result_482)
                # Assigning a type to the variable 'if_condition_483' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'if_condition_483', if_condition_483)
                # SSA begins for if statement (line 146)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'pos' (line 147)
                pos_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'pos')
                # Assigning a type to the variable 'stypy_return_type' (line 147)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'stypy_return_type', pos_484)
                # SSA join for if statement (line 146)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'choices' (line 148)
            choices_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'choices')
            int_486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'int')
            # Applying the binary operator '-=' (line 148)
            result_isub_487 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 12), '-=', choices_485, int_486)
            # Assigning a type to the variable 'choices' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'choices', result_isub_487)
            
            
            # Call to set(...): (line 149)
            # Processing the call arguments (line 149)
            # Getting the type of 'i' (line 149)
            i_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'i', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'choices' (line 149)
            choices_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'choices', False)
            # Getting the type of 'self' (line 149)
            self_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'self', False)
            # Obtaining the member 'empties' of a type (line 149)
            empties_493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), self_492, 'empties')
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), empties_493, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_495 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), getitem___494, choices_491)
            
            # Processing the call keyword arguments (line 149)
            kwargs_496 = {}
            # Getting the type of 'self' (line 149)
            self_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self', False)
            # Obtaining the member 'set' of a type (line 149)
            set_489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_488, 'set')
            # Calling set(args, kwargs) (line 149)
            set_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), set_489, *[i_490, subscript_call_result_495], **kwargs_496)
            
            
            # Call to set(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'choices' (line 150)
            choices_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'choices', False)
            # Getting the type of 'pos' (line 150)
            pos_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'pos', False)
            # Processing the call keyword arguments (line 150)
            kwargs_502 = {}
            # Getting the type of 'self' (line 150)
            self_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'self', False)
            # Obtaining the member 'set' of a type (line 150)
            set_499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), self_498, 'set')
            # Calling set(args, kwargs) (line 150)
            set_call_result_503 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), set_499, *[choices_500, pos_501], **kwargs_502)
            
            # SSA join for while statement (line 143)
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'PASS' (line 151)
        PASS_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'PASS')
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', PASS_504)
        
        # ################# End of 'random_choice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'random_choice' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'random_choice'
        return stypy_return_type_505


    @norecursion
    def add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EmptySet.add.__dict__.__setitem__('stypy_localization', localization)
        EmptySet.add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EmptySet.add.__dict__.__setitem__('stypy_type_store', module_type_store)
        EmptySet.add.__dict__.__setitem__('stypy_function_name', 'EmptySet.add')
        EmptySet.add.__dict__.__setitem__('stypy_param_names_list', ['pos'])
        EmptySet.add.__dict__.__setitem__('stypy_varargs_param_name', None)
        EmptySet.add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EmptySet.add.__dict__.__setitem__('stypy_call_defaults', defaults)
        EmptySet.add.__dict__.__setitem__('stypy_call_varargs', varargs)
        EmptySet.add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EmptySet.add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EmptySet.add', ['pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, ['pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        
        # Assigning a Call to a Subscript (line 154):
        
        # Assigning a Call to a Subscript (line 154):
        
        # Call to len(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'self' (line 154)
        self_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'self', False)
        # Obtaining the member 'empties' of a type (line 154)
        empties_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 34), self_507, 'empties')
        # Processing the call keyword arguments (line 154)
        kwargs_509 = {}
        # Getting the type of 'len' (line 154)
        len_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'len', False)
        # Calling len(args, kwargs) (line 154)
        len_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 154, 30), len_506, *[empties_508], **kwargs_509)
        
        # Getting the type of 'self' (line 154)
        self_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Obtaining the member 'empty_pos' of a type (line 154)
        empty_pos_512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_511, 'empty_pos')
        # Getting the type of 'pos' (line 154)
        pos_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'pos')
        # Storing an element on a container (line 154)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 8), empty_pos_512, (pos_513, len_call_result_510))
        
        # Call to append(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'pos' (line 155)
        pos_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'pos', False)
        # Processing the call keyword arguments (line 155)
        kwargs_518 = {}
        # Getting the type of 'self' (line 155)
        self_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'empties' of a type (line 155)
        empties_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_514, 'empties')
        # Obtaining the member 'append' of a type (line 155)
        append_516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), empties_515, 'append')
        # Calling append(args, kwargs) (line 155)
        append_call_result_519 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), append_516, *[pos_517], **kwargs_518)
        
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_520)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_520


    @norecursion
    def remove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove'
        module_type_store = module_type_store.open_function_context('remove', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EmptySet.remove.__dict__.__setitem__('stypy_localization', localization)
        EmptySet.remove.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EmptySet.remove.__dict__.__setitem__('stypy_type_store', module_type_store)
        EmptySet.remove.__dict__.__setitem__('stypy_function_name', 'EmptySet.remove')
        EmptySet.remove.__dict__.__setitem__('stypy_param_names_list', ['pos'])
        EmptySet.remove.__dict__.__setitem__('stypy_varargs_param_name', None)
        EmptySet.remove.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EmptySet.remove.__dict__.__setitem__('stypy_call_defaults', defaults)
        EmptySet.remove.__dict__.__setitem__('stypy_call_varargs', varargs)
        EmptySet.remove.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EmptySet.remove.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EmptySet.remove', ['pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove', localization, ['pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove(...)' code ##################

        
        # Call to set(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 158)
        pos_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'pos', False)
        # Getting the type of 'self' (line 158)
        self_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'self', False)
        # Obtaining the member 'empty_pos' of a type (line 158)
        empty_pos_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 17), self_524, 'empty_pos')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 17), empty_pos_525, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 158, 17), getitem___526, pos_523)
        
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'self', False)
        # Obtaining the member 'empties' of a type (line 158)
        empties_530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 55), self_529, 'empties')
        # Processing the call keyword arguments (line 158)
        kwargs_531 = {}
        # Getting the type of 'len' (line 158)
        len_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 51), 'len', False)
        # Calling len(args, kwargs) (line 158)
        len_call_result_532 = invoke(stypy.reporting.localization.Localization(__file__, 158, 51), len_528, *[empties_530], **kwargs_531)
        
        int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 71), 'int')
        # Applying the binary operator '-' (line 158)
        result_sub_534 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 51), '-', len_call_result_532, int_533)
        
        # Getting the type of 'self' (line 158)
        self_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'self', False)
        # Obtaining the member 'empties' of a type (line 158)
        empties_536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 38), self_535, 'empties')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 38), empties_536, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 158, 38), getitem___537, result_sub_534)
        
        # Processing the call keyword arguments (line 158)
        kwargs_539 = {}
        # Getting the type of 'self' (line 158)
        self_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self', False)
        # Obtaining the member 'set' of a type (line 158)
        set_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_521, 'set')
        # Calling set(args, kwargs) (line 158)
        set_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), set_522, *[subscript_call_result_527, subscript_call_result_538], **kwargs_539)
        
        
        # Call to pop(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_544 = {}
        # Getting the type of 'self' (line 159)
        self_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self', False)
        # Obtaining the member 'empties' of a type (line 159)
        empties_542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_541, 'empties')
        # Obtaining the member 'pop' of a type (line 159)
        pop_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), empties_542, 'pop')
        # Calling pop(args, kwargs) (line 159)
        pop_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), pop_543, *[], **kwargs_544)
        
        
        # ################# End of 'remove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_546)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_546


    @norecursion
    def set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set'
        module_type_store = module_type_store.open_function_context('set', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EmptySet.set.__dict__.__setitem__('stypy_localization', localization)
        EmptySet.set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EmptySet.set.__dict__.__setitem__('stypy_type_store', module_type_store)
        EmptySet.set.__dict__.__setitem__('stypy_function_name', 'EmptySet.set')
        EmptySet.set.__dict__.__setitem__('stypy_param_names_list', ['i', 'pos'])
        EmptySet.set.__dict__.__setitem__('stypy_varargs_param_name', None)
        EmptySet.set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EmptySet.set.__dict__.__setitem__('stypy_call_defaults', defaults)
        EmptySet.set.__dict__.__setitem__('stypy_call_varargs', varargs)
        EmptySet.set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EmptySet.set.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EmptySet.set', ['i', 'pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set', localization, ['i', 'pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set(...)' code ##################

        
        # Assigning a Name to a Subscript (line 162):
        
        # Assigning a Name to a Subscript (line 162):
        # Getting the type of 'pos' (line 162)
        pos_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'pos')
        # Getting the type of 'self' (line 162)
        self_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Obtaining the member 'empties' of a type (line 162)
        empties_549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_548, 'empties')
        # Getting the type of 'i' (line 162)
        i_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'i')
        # Storing an element on a container (line 162)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), empties_549, (i_550, pos_547))
        
        # Assigning a Name to a Subscript (line 163):
        
        # Assigning a Name to a Subscript (line 163):
        # Getting the type of 'i' (line 163)
        i_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'i')
        # Getting the type of 'self' (line 163)
        self_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Obtaining the member 'empty_pos' of a type (line 163)
        empty_pos_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_552, 'empty_pos')
        # Getting the type of 'pos' (line 163)
        pos_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'pos')
        # Storing an element on a container (line 163)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 8), empty_pos_553, (pos_554, i_551))
        
        # ################# End of 'set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set'
        return stypy_return_type_555


# Assigning a type to the variable 'EmptySet' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'EmptySet', EmptySet)
# Declaration of the 'ZobristHash' class

class ZobristHash:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZobristHash.__init__', ['board'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['board'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'board' (line 168)
        board_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'board')
        # Getting the type of 'self' (line 168)
        self_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member 'board' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_557, 'board', board_556)
        
        # Assigning a Call to a Attribute (line 169):
        
        # Assigning a Call to a Attribute (line 169):
        
        # Call to set(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_559 = {}
        # Getting the type of 'set' (line 169)
        set_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'set', False)
        # Calling set(args, kwargs) (line 169)
        set_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), set_558, *[], **kwargs_559)
        
        # Getting the type of 'self' (line 169)
        self_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member 'hash_set' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_561, 'hash_set', set_call_result_560)
        
        # Assigning a Num to a Attribute (line 170):
        
        # Assigning a Num to a Attribute (line 170):
        int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 20), 'int')
        # Getting the type of 'self' (line 170)
        self_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'hash' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_563, 'hash', int_562)
        
        # Getting the type of 'self' (line 171)
        self_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'self')
        # Obtaining the member 'board' of a type (line 171)
        board_565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), self_564, 'board')
        # Obtaining the member 'squares' of a type (line 171)
        squares_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), board_565, 'squares')
        # Assigning a type to the variable 'squares_566' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'squares_566', squares_566)
        # Testing if the for loop is going to be iterated (line 171)
        # Testing the type of a for loop iterable (line 171)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 8), squares_566)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 171, 8), squares_566):
            # Getting the type of the for loop variable (line 171)
            for_loop_var_567 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 8), squares_566)
            # Assigning a type to the variable 'square' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'square', for_loop_var_567)
            # SSA begins for a for statement (line 171)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'self' (line 172)
            self_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'self')
            # Obtaining the member 'hash' of a type (line 172)
            hash_569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), self_568, 'hash')
            
            # Obtaining the type of the subscript
            # Getting the type of 'EMPTY' (line 172)
            EMPTY_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 48), 'EMPTY')
            # Getting the type of 'square' (line 172)
            square_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'square')
            # Obtaining the member 'zobrist_strings' of a type (line 172)
            zobrist_strings_572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), square_571, 'zobrist_strings')
            # Obtaining the member '__getitem__' of a type (line 172)
            getitem___573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), zobrist_strings_572, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 172)
            subscript_call_result_574 = invoke(stypy.reporting.localization.Localization(__file__, 172, 25), getitem___573, EMPTY_570)
            
            # Applying the binary operator '^=' (line 172)
            result_ixor_575 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 12), '^=', hash_569, subscript_call_result_574)
            # Getting the type of 'self' (line 172)
            self_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'self')
            # Setting the type of the member 'hash' of a type (line 172)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), self_576, 'hash', result_ixor_575)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to clear(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_580 = {}
        # Getting the type of 'self' (line 173)
        self_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'hash_set' of a type (line 173)
        hash_set_578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_577, 'hash_set')
        # Obtaining the member 'clear' of a type (line 173)
        clear_579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), hash_set_578, 'clear')
        # Calling clear(args, kwargs) (line 173)
        clear_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), clear_579, *[], **kwargs_580)
        
        
        # Call to add(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'self' (line 174)
        self_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'self', False)
        # Obtaining the member 'hash' of a type (line 174)
        hash_586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), self_585, 'hash')
        # Processing the call keyword arguments (line 174)
        kwargs_587 = {}
        # Getting the type of 'self' (line 174)
        self_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self', False)
        # Obtaining the member 'hash_set' of a type (line 174)
        hash_set_583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_582, 'hash_set')
        # Obtaining the member 'add' of a type (line 174)
        add_584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), hash_set_583, 'add')
        # Calling add(args, kwargs) (line 174)
        add_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), add_584, *[hash_586], **kwargs_587)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZobristHash.update.__dict__.__setitem__('stypy_localization', localization)
        ZobristHash.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZobristHash.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZobristHash.update.__dict__.__setitem__('stypy_function_name', 'ZobristHash.update')
        ZobristHash.update.__dict__.__setitem__('stypy_param_names_list', ['square', 'color'])
        ZobristHash.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZobristHash.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZobristHash.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZobristHash.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZobristHash.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZobristHash.update.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZobristHash.update', ['square', 'color'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['square', 'color'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        
        # Getting the type of 'self' (line 177)
        self_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Obtaining the member 'hash' of a type (line 177)
        hash_590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_589, 'hash')
        
        # Obtaining the type of the subscript
        # Getting the type of 'square' (line 177)
        square_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 44), 'square')
        # Obtaining the member 'color' of a type (line 177)
        color_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 44), square_591, 'color')
        # Getting the type of 'square' (line 177)
        square_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'square')
        # Obtaining the member 'zobrist_strings' of a type (line 177)
        zobrist_strings_594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), square_593, 'zobrist_strings')
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), zobrist_strings_594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 177, 21), getitem___595, color_592)
        
        # Applying the binary operator '^=' (line 177)
        result_ixor_597 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 8), '^=', hash_590, subscript_call_result_596)
        # Getting the type of 'self' (line 177)
        self_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member 'hash' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_598, 'hash', result_ixor_597)
        
        
        # Getting the type of 'self' (line 178)
        self_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Obtaining the member 'hash' of a type (line 178)
        hash_600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_599, 'hash')
        
        # Obtaining the type of the subscript
        # Getting the type of 'color' (line 178)
        color_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'color')
        # Getting the type of 'square' (line 178)
        square_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'square')
        # Obtaining the member 'zobrist_strings' of a type (line 178)
        zobrist_strings_603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 21), square_602, 'zobrist_strings')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 21), zobrist_strings_603, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 178, 21), getitem___604, color_601)
        
        # Applying the binary operator '^=' (line 178)
        result_ixor_606 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 8), '^=', hash_600, subscript_call_result_605)
        # Getting the type of 'self' (line 178)
        self_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Setting the type of the member 'hash' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_607, 'hash', result_ixor_606)
        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_608


    @norecursion
    def add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZobristHash.add.__dict__.__setitem__('stypy_localization', localization)
        ZobristHash.add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZobristHash.add.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZobristHash.add.__dict__.__setitem__('stypy_function_name', 'ZobristHash.add')
        ZobristHash.add.__dict__.__setitem__('stypy_param_names_list', [])
        ZobristHash.add.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZobristHash.add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZobristHash.add.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZobristHash.add.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZobristHash.add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZobristHash.add.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZobristHash.add', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        
        # Call to add(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'self' (line 181)
        self_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'self', False)
        # Obtaining the member 'hash' of a type (line 181)
        hash_613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 26), self_612, 'hash')
        # Processing the call keyword arguments (line 181)
        kwargs_614 = {}
        # Getting the type of 'self' (line 181)
        self_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member 'hash_set' of a type (line 181)
        hash_set_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_609, 'hash_set')
        # Obtaining the member 'add' of a type (line 181)
        add_611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), hash_set_610, 'add')
        # Calling add(args, kwargs) (line 181)
        add_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), add_611, *[hash_613], **kwargs_614)
        
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_616)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_616


    @norecursion
    def dupe(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dupe'
        module_type_store = module_type_store.open_function_context('dupe', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZobristHash.dupe.__dict__.__setitem__('stypy_localization', localization)
        ZobristHash.dupe.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZobristHash.dupe.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZobristHash.dupe.__dict__.__setitem__('stypy_function_name', 'ZobristHash.dupe')
        ZobristHash.dupe.__dict__.__setitem__('stypy_param_names_list', [])
        ZobristHash.dupe.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZobristHash.dupe.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZobristHash.dupe.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZobristHash.dupe.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZobristHash.dupe.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZobristHash.dupe.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZobristHash.dupe', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dupe', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dupe(...)' code ##################

        
        # Getting the type of 'self' (line 184)
        self_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'self')
        # Obtaining the member 'hash' of a type (line 184)
        hash_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 15), self_617, 'hash')
        # Getting the type of 'self' (line 184)
        self_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'self')
        # Obtaining the member 'hash_set' of a type (line 184)
        hash_set_620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), self_619, 'hash_set')
        # Applying the binary operator 'in' (line 184)
        result_contains_621 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), 'in', hash_618, hash_set_620)
        
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', result_contains_621)
        
        # ################# End of 'dupe(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dupe' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dupe'
        return stypy_return_type_622


# Assigning a type to the variable 'ZobristHash' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'ZobristHash', ZobristHash)
# Declaration of the 'Board' class

class Board:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 188, 4, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a ListComp to a Attribute (line 189):
        
        # Assigning a ListComp to a Attribute (line 189):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'SIZE' (line 189)
        SIZE_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 59), 'SIZE', False)
        # Getting the type of 'SIZE' (line 189)
        SIZE_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 66), 'SIZE', False)
        # Applying the binary operator '*' (line 189)
        result_mul_631 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 59), '*', SIZE_629, SIZE_630)
        
        # Processing the call keyword arguments (line 189)
        kwargs_632 = {}
        # Getting the type of 'range' (line 189)
        range_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 53), 'range', False)
        # Calling range(args, kwargs) (line 189)
        range_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 189, 53), range_628, *[result_mul_631], **kwargs_632)
        
        comprehension_634 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 24), range_call_result_633)
        # Assigning a type to the variable 'pos' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'pos', comprehension_634)
        
        # Call to Square(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 31), 'self', False)
        # Getting the type of 'pos' (line 189)
        pos_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'pos', False)
        # Processing the call keyword arguments (line 189)
        kwargs_626 = {}
        # Getting the type of 'Square' (line 189)
        Square_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'Square', False)
        # Calling Square(args, kwargs) (line 189)
        Square_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), Square_623, *[self_624, pos_625], **kwargs_626)
        
        list_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 24), list_635, Square_call_result_627)
        # Getting the type of 'self' (line 189)
        self_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self')
        # Setting the type of the member 'squares' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_636, 'squares', list_635)
        
        # Getting the type of 'self' (line 190)
        self_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'self')
        # Obtaining the member 'squares' of a type (line 190)
        squares_638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 22), self_637, 'squares')
        # Assigning a type to the variable 'squares_638' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'squares_638', squares_638)
        # Testing if the for loop is going to be iterated (line 190)
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), squares_638)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 190, 8), squares_638):
            # Getting the type of the for loop variable (line 190)
            for_loop_var_639 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), squares_638)
            # Assigning a type to the variable 'square' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'square', for_loop_var_639)
            # SSA begins for a for statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to set_neighbours(...): (line 191)
            # Processing the call keyword arguments (line 191)
            kwargs_642 = {}
            # Getting the type of 'square' (line 191)
            square_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'square', False)
            # Obtaining the member 'set_neighbours' of a type (line 191)
            set_neighbours_641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), square_640, 'set_neighbours')
            # Calling set_neighbours(args, kwargs) (line 191)
            set_neighbours_call_result_643 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), set_neighbours_641, *[], **kwargs_642)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to reset(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_646 = {}
        # Getting the type of 'self' (line 192)
        self_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self', False)
        # Obtaining the member 'reset' of a type (line 192)
        reset_645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_644, 'reset')
        # Calling reset(args, kwargs) (line 192)
        reset_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), reset_645, *[], **kwargs_646)
        
        
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
        module_type_store = module_type_store.open_function_context('reset', 194, 4, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.reset.__dict__.__setitem__('stypy_localization', localization)
        Board.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.reset.__dict__.__setitem__('stypy_function_name', 'Board.reset')
        Board.reset.__dict__.__setitem__('stypy_param_names_list', [])
        Board.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.reset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.reset', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 195)
        self_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'self')
        # Obtaining the member 'squares' of a type (line 195)
        squares_649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 22), self_648, 'squares')
        # Assigning a type to the variable 'squares_649' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'squares_649', squares_649)
        # Testing if the for loop is going to be iterated (line 195)
        # Testing the type of a for loop iterable (line 195)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 8), squares_649)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 195, 8), squares_649):
            # Getting the type of the for loop variable (line 195)
            for_loop_var_650 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 8), squares_649)
            # Assigning a type to the variable 'square' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'square', for_loop_var_650)
            # SSA begins for a for statement (line 195)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Attribute (line 196):
            
            # Assigning a Name to a Attribute (line 196):
            # Getting the type of 'EMPTY' (line 196)
            EMPTY_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'EMPTY')
            # Getting the type of 'square' (line 196)
            square_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'square')
            # Setting the type of the member 'color' of a type (line 196)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), square_652, 'color', EMPTY_651)
            
            # Assigning a Name to a Attribute (line 197):
            
            # Assigning a Name to a Attribute (line 197):
            # Getting the type of 'False' (line 197)
            False_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'False')
            # Getting the type of 'square' (line 197)
            square_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'square')
            # Setting the type of the member 'used' of a type (line 197)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), square_654, 'used', False_653)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Attribute (line 198):
        
        # Assigning a Call to a Attribute (line 198):
        
        # Call to EmptySet(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'self' (line 198)
        self_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 33), 'self', False)
        # Processing the call keyword arguments (line 198)
        kwargs_657 = {}
        # Getting the type of 'EmptySet' (line 198)
        EmptySet_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'EmptySet', False)
        # Calling EmptySet(args, kwargs) (line 198)
        EmptySet_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 198, 24), EmptySet_655, *[self_656], **kwargs_657)
        
        # Getting the type of 'self' (line 198)
        self_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member 'emptyset' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_659, 'emptyset', EmptySet_call_result_658)
        
        # Assigning a Call to a Attribute (line 199):
        
        # Assigning a Call to a Attribute (line 199):
        
        # Call to ZobristHash(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'self' (line 199)
        self_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'self', False)
        # Processing the call keyword arguments (line 199)
        kwargs_662 = {}
        # Getting the type of 'ZobristHash' (line 199)
        ZobristHash_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'ZobristHash', False)
        # Calling ZobristHash(args, kwargs) (line 199)
        ZobristHash_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 199, 23), ZobristHash_660, *[self_661], **kwargs_662)
        
        # Getting the type of 'self' (line 199)
        self_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self')
        # Setting the type of the member 'zobrist' of a type (line 199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_664, 'zobrist', ZobristHash_call_result_663)
        
        # Assigning a Name to a Attribute (line 200):
        
        # Assigning a Name to a Attribute (line 200):
        # Getting the type of 'BLACK' (line 200)
        BLACK_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'BLACK')
        # Getting the type of 'self' (line 200)
        self_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member 'color' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_666, 'color', BLACK_665)
        
        # Assigning a Name to a Attribute (line 201):
        
        # Assigning a Name to a Attribute (line 201):
        # Getting the type of 'False' (line 201)
        False_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'False')
        # Getting the type of 'self' (line 201)
        self_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self')
        # Setting the type of the member 'finished' of a type (line 201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_668, 'finished', False_667)
        
        # Assigning a Num to a Attribute (line 202):
        
        # Assigning a Num to a Attribute (line 202):
        int_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 24), 'int')
        # Getting the type of 'self' (line 202)
        self_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member 'lastmove' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_670, 'lastmove', int_669)
        
        # Assigning a List to a Attribute (line 203):
        
        # Assigning a List to a Attribute (line 203):
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        
        # Getting the type of 'self' (line 203)
        self_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self')
        # Setting the type of the member 'history' of a type (line 203)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_672, 'history', list_671)
        
        # Assigning a Name to a Attribute (line 204):
        
        # Assigning a Name to a Attribute (line 204):
        # Getting the type of 'None' (line 204)
        None_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'None')
        # Getting the type of 'self' (line 204)
        self_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member 'atari' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_674, 'atari', None_673)
        
        # Assigning a Num to a Attribute (line 205):
        
        # Assigning a Num to a Attribute (line 205):
        int_675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'int')
        # Getting the type of 'self' (line 205)
        self_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'white_dead' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_676, 'white_dead', int_675)
        
        # Assigning a Num to a Attribute (line 206):
        
        # Assigning a Num to a Attribute (line 206):
        int_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 26), 'int')
        # Getting the type of 'self' (line 206)
        self_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'black_dead' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_678, 'black_dead', int_677)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_679


    @norecursion
    def move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'move'
        module_type_store = module_type_store.open_function_context('move', 208, 4, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.move.__dict__.__setitem__('stypy_localization', localization)
        Board.move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.move.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.move.__dict__.__setitem__('stypy_function_name', 'Board.move')
        Board.move.__dict__.__setitem__('stypy_param_names_list', ['pos'])
        Board.move.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.move.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.move.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.move.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.move', ['pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'move', localization, ['pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'move(...)' code ##################

        
        # Assigning a Subscript to a Name (line 209):
        
        # Assigning a Subscript to a Name (line 209):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 209)
        pos_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'pos')
        # Getting the type of 'self' (line 209)
        self_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'self')
        # Obtaining the member 'squares' of a type (line 209)
        squares_682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 17), self_681, 'squares')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 17), squares_682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 209, 17), getitem___683, pos_680)
        
        # Assigning a type to the variable 'square' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'square', subscript_call_result_684)
        
        # Getting the type of 'pos' (line 210)
        pos_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'pos')
        # Getting the type of 'PASS' (line 210)
        PASS_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 'PASS')
        # Applying the binary operator '!=' (line 210)
        result_ne_687 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), '!=', pos_685, PASS_686)
        
        # Testing if the type of an if condition is none (line 210)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 210, 8), result_ne_687):
            
            # Getting the type of 'self' (line 213)
            self_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'self')
            # Obtaining the member 'lastmove' of a type (line 213)
            lastmove_703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 13), self_702, 'lastmove')
            # Getting the type of 'PASS' (line 213)
            PASS_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'PASS')
            # Applying the binary operator '==' (line 213)
            result_eq_705 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 13), '==', lastmove_703, PASS_704)
            
            # Testing if the type of an if condition is none (line 213)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 13), result_eq_705):
                pass
            else:
                
                # Testing the type of an if condition (line 213)
                if_condition_706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 13), result_eq_705)
                # Assigning a type to the variable 'if_condition_706' (line 213)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'if_condition_706', if_condition_706)
                # SSA begins for if statement (line 213)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 214):
                
                # Assigning a Name to a Attribute (line 214):
                # Getting the type of 'True' (line 214)
                True_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'True')
                # Getting the type of 'self' (line 214)
                self_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'self')
                # Setting the type of the member 'finished' of a type (line 214)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), self_708, 'finished', True_707)
                # SSA join for if statement (line 213)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 210)
            if_condition_688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_ne_687)
            # Assigning a type to the variable 'if_condition_688' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_688', if_condition_688)
            # SSA begins for if statement (line 210)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to move(...): (line 211)
            # Processing the call arguments (line 211)
            # Getting the type of 'self' (line 211)
            self_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'self', False)
            # Obtaining the member 'color' of a type (line 211)
            color_692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), self_691, 'color')
            # Processing the call keyword arguments (line 211)
            kwargs_693 = {}
            # Getting the type of 'square' (line 211)
            square_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'square', False)
            # Obtaining the member 'move' of a type (line 211)
            move_690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), square_689, 'move')
            # Calling move(args, kwargs) (line 211)
            move_call_result_694 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), move_690, *[color_692], **kwargs_693)
            
            
            # Call to remove(...): (line 212)
            # Processing the call arguments (line 212)
            # Getting the type of 'square' (line 212)
            square_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'square', False)
            # Obtaining the member 'pos' of a type (line 212)
            pos_699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 33), square_698, 'pos')
            # Processing the call keyword arguments (line 212)
            kwargs_700 = {}
            # Getting the type of 'self' (line 212)
            self_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'self', False)
            # Obtaining the member 'emptyset' of a type (line 212)
            emptyset_696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), self_695, 'emptyset')
            # Obtaining the member 'remove' of a type (line 212)
            remove_697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), emptyset_696, 'remove')
            # Calling remove(args, kwargs) (line 212)
            remove_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), remove_697, *[pos_699], **kwargs_700)
            
            # SSA branch for the else part of an if statement (line 210)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 213)
            self_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'self')
            # Obtaining the member 'lastmove' of a type (line 213)
            lastmove_703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 13), self_702, 'lastmove')
            # Getting the type of 'PASS' (line 213)
            PASS_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'PASS')
            # Applying the binary operator '==' (line 213)
            result_eq_705 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 13), '==', lastmove_703, PASS_704)
            
            # Testing if the type of an if condition is none (line 213)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 13), result_eq_705):
                pass
            else:
                
                # Testing the type of an if condition (line 213)
                if_condition_706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 13), result_eq_705)
                # Assigning a type to the variable 'if_condition_706' (line 213)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'if_condition_706', if_condition_706)
                # SSA begins for if statement (line 213)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 214):
                
                # Assigning a Name to a Attribute (line 214):
                # Getting the type of 'True' (line 214)
                True_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'True')
                # Getting the type of 'self' (line 214)
                self_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'self')
                # Setting the type of the member 'finished' of a type (line 214)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), self_708, 'finished', True_707)
                # SSA join for if statement (line 213)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 210)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 215)
        self_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'self')
        # Obtaining the member 'color' of a type (line 215)
        color_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 11), self_709, 'color')
        # Getting the type of 'BLACK' (line 215)
        BLACK_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 25), 'BLACK')
        # Applying the binary operator '==' (line 215)
        result_eq_712 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), '==', color_710, BLACK_711)
        
        # Testing if the type of an if condition is none (line 215)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 215, 8), result_eq_712):
            
            # Assigning a Name to a Attribute (line 218):
            
            # Assigning a Name to a Attribute (line 218):
            # Getting the type of 'BLACK' (line 218)
            BLACK_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'BLACK')
            # Getting the type of 'self' (line 218)
            self_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self')
            # Setting the type of the member 'color' of a type (line 218)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_717, 'color', BLACK_716)
        else:
            
            # Testing the type of an if condition (line 215)
            if_condition_713 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_eq_712)
            # Assigning a type to the variable 'if_condition_713' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_713', if_condition_713)
            # SSA begins for if statement (line 215)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 216):
            
            # Assigning a Name to a Attribute (line 216):
            # Getting the type of 'WHITE' (line 216)
            WHITE_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'WHITE')
            # Getting the type of 'self' (line 216)
            self_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'self')
            # Setting the type of the member 'color' of a type (line 216)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), self_715, 'color', WHITE_714)
            # SSA branch for the else part of an if statement (line 215)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 218):
            
            # Assigning a Name to a Attribute (line 218):
            # Getting the type of 'BLACK' (line 218)
            BLACK_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'BLACK')
            # Getting the type of 'self' (line 218)
            self_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self')
            # Setting the type of the member 'color' of a type (line 218)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_717, 'color', BLACK_716)
            # SSA join for if statement (line 215)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 219):
        
        # Assigning a Name to a Attribute (line 219):
        # Getting the type of 'pos' (line 219)
        pos_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'pos')
        # Getting the type of 'self' (line 219)
        self_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self')
        # Setting the type of the member 'lastmove' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_719, 'lastmove', pos_718)
        
        # Call to append(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'pos' (line 220)
        pos_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'pos', False)
        # Processing the call keyword arguments (line 220)
        kwargs_724 = {}
        # Getting the type of 'self' (line 220)
        self_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self', False)
        # Obtaining the member 'history' of a type (line 220)
        history_721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_720, 'history')
        # Obtaining the member 'append' of a type (line 220)
        append_722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), history_721, 'append')
        # Calling append(args, kwargs) (line 220)
        append_call_result_725 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), append_722, *[pos_723], **kwargs_724)
        
        
        # ################# End of 'move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move'
        return stypy_return_type_726


    @norecursion
    def random_move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'random_move'
        module_type_store = module_type_store.open_function_context('random_move', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.random_move.__dict__.__setitem__('stypy_localization', localization)
        Board.random_move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.random_move.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.random_move.__dict__.__setitem__('stypy_function_name', 'Board.random_move')
        Board.random_move.__dict__.__setitem__('stypy_param_names_list', [])
        Board.random_move.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.random_move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.random_move.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.random_move.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.random_move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.random_move.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.random_move', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'random_move', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'random_move(...)' code ##################

        
        # Call to random_choice(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_730 = {}
        # Getting the type of 'self' (line 223)
        self_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'self', False)
        # Obtaining the member 'emptyset' of a type (line 223)
        emptyset_728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), self_727, 'emptyset')
        # Obtaining the member 'random_choice' of a type (line 223)
        random_choice_729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), emptyset_728, 'random_choice')
        # Calling random_choice(args, kwargs) (line 223)
        random_choice_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), random_choice_729, *[], **kwargs_730)
        
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', random_choice_call_result_731)
        
        # ################# End of 'random_move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'random_move' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'random_move'
        return stypy_return_type_732


    @norecursion
    def useful_fast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'useful_fast'
        module_type_store = module_type_store.open_function_context('useful_fast', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.useful_fast.__dict__.__setitem__('stypy_localization', localization)
        Board.useful_fast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.useful_fast.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.useful_fast.__dict__.__setitem__('stypy_function_name', 'Board.useful_fast')
        Board.useful_fast.__dict__.__setitem__('stypy_param_names_list', ['square'])
        Board.useful_fast.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.useful_fast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.useful_fast.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.useful_fast.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.useful_fast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.useful_fast.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.useful_fast', ['square'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'useful_fast', localization, ['square'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'useful_fast(...)' code ##################

        
        # Getting the type of 'square' (line 226)
        square_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'square')
        # Obtaining the member 'used' of a type (line 226)
        used_734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 15), square_733, 'used')
        # Applying the 'not' unary operator (line 226)
        result_not__735 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 11), 'not', used_734)
        
        # Testing if the type of an if condition is none (line 226)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 8), result_not__735):
            pass
        else:
            
            # Testing the type of an if condition (line 226)
            if_condition_736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), result_not__735)
            # Assigning a type to the variable 'if_condition_736' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_736', if_condition_736)
            # SSA begins for if statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'square' (line 227)
            square_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'square')
            # Obtaining the member 'neighbours' of a type (line 227)
            neighbours_738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 29), square_737, 'neighbours')
            # Assigning a type to the variable 'neighbours_738' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'neighbours_738', neighbours_738)
            # Testing if the for loop is going to be iterated (line 227)
            # Testing the type of a for loop iterable (line 227)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 227, 12), neighbours_738)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 227, 12), neighbours_738):
                # Getting the type of the for loop variable (line 227)
                for_loop_var_739 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 227, 12), neighbours_738)
                # Assigning a type to the variable 'neighbour' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'neighbour', for_loop_var_739)
                # SSA begins for a for statement (line 227)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'neighbour' (line 228)
                neighbour_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'neighbour')
                # Obtaining the member 'color' of a type (line 228)
                color_741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), neighbour_740, 'color')
                # Getting the type of 'EMPTY' (line 228)
                EMPTY_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'EMPTY')
                # Applying the binary operator '==' (line 228)
                result_eq_743 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 19), '==', color_741, EMPTY_742)
                
                # Testing if the type of an if condition is none (line 228)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 228, 16), result_eq_743):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 228)
                    if_condition_744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 16), result_eq_743)
                    # Assigning a type to the variable 'if_condition_744' (line 228)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'if_condition_744', if_condition_744)
                    # SSA begins for if statement (line 228)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'True' (line 229)
                    True_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'True')
                    # Assigning a type to the variable 'stypy_return_type' (line 229)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'stypy_return_type', True_745)
                    # SSA join for if statement (line 228)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 226)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 230)
        False_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', False_746)
        
        # ################# End of 'useful_fast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'useful_fast' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'useful_fast'
        return stypy_return_type_747


    @norecursion
    def useful(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'useful'
        module_type_store = module_type_store.open_function_context('useful', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.useful.__dict__.__setitem__('stypy_localization', localization)
        Board.useful.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.useful.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.useful.__dict__.__setitem__('stypy_function_name', 'Board.useful')
        Board.useful.__dict__.__setitem__('stypy_param_names_list', ['pos'])
        Board.useful.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.useful.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.useful.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.useful.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.useful.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.useful.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.useful', ['pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'useful', localization, ['pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'useful(...)' code ##################

        # Marking variables as global (line 233)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 233, 8), 'TIMESTAMP')
        
        # Getting the type of 'TIMESTAMP' (line 234)
        TIMESTAMP_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'TIMESTAMP')
        int_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'int')
        # Applying the binary operator '+=' (line 234)
        result_iadd_750 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 8), '+=', TIMESTAMP_748, int_749)
        # Assigning a type to the variable 'TIMESTAMP' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'TIMESTAMP', result_iadd_750)
        
        
        # Assigning a Subscript to a Name (line 235):
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 235)
        pos_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'pos')
        # Getting the type of 'self' (line 235)
        self_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'self')
        # Obtaining the member 'squares' of a type (line 235)
        squares_753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), self_752, 'squares')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), squares_753, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_755 = invoke(stypy.reporting.localization.Localization(__file__, 235, 17), getitem___754, pos_751)
        
        # Assigning a type to the variable 'square' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'square', subscript_call_result_755)
        
        # Call to useful_fast(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'square' (line 236)
        square_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'square', False)
        # Processing the call keyword arguments (line 236)
        kwargs_759 = {}
        # Getting the type of 'self' (line 236)
        self_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'self', False)
        # Obtaining the member 'useful_fast' of a type (line 236)
        useful_fast_757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 11), self_756, 'useful_fast')
        # Calling useful_fast(args, kwargs) (line 236)
        useful_fast_call_result_760 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), useful_fast_757, *[square_758], **kwargs_759)
        
        # Testing if the type of an if condition is none (line 236)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 236, 8), useful_fast_call_result_760):
            pass
        else:
            
            # Testing the type of an if condition (line 236)
            if_condition_761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), useful_fast_call_result_760)
            # Assigning a type to the variable 'if_condition_761' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_761', if_condition_761)
            # SSA begins for if statement (line 236)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 237)
            True_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 237)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'stypy_return_type', True_762)
            # SSA join for if statement (line 236)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Attribute to a Name (line 238):
        
        # Assigning a Attribute to a Name (line 238):
        # Getting the type of 'self' (line 238)
        self_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 19), 'self')
        # Obtaining the member 'zobrist' of a type (line 238)
        zobrist_764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 19), self_763, 'zobrist')
        # Obtaining the member 'hash' of a type (line 238)
        hash_765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 19), zobrist_764, 'hash')
        # Assigning a type to the variable 'old_hash' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'old_hash', hash_765)
        
        # Call to update(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'square' (line 239)
        square_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'square', False)
        # Getting the type of 'self' (line 239)
        self_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'self', False)
        # Obtaining the member 'color' of a type (line 239)
        color_771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 36), self_770, 'color')
        # Processing the call keyword arguments (line 239)
        kwargs_772 = {}
        # Getting the type of 'self' (line 239)
        self_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self', False)
        # Obtaining the member 'zobrist' of a type (line 239)
        zobrist_767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_766, 'zobrist')
        # Obtaining the member 'update' of a type (line 239)
        update_768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), zobrist_767, 'update')
        # Calling update(args, kwargs) (line 239)
        update_call_result_773 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), update_768, *[square_769, color_771], **kwargs_772)
        
        
        # Multiple assignment of 5 elements.
        
        # Assigning a Num to a Name (line 240):
        int_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 74), 'int')
        # Assigning a type to the variable 'weak_neighs' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 60), 'weak_neighs', int_774)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'weak_neighs' (line 240)
        weak_neighs_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 60), 'weak_neighs')
        # Assigning a type to the variable 'strong_neighs' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'strong_neighs', weak_neighs_775)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'strong_neighs' (line 240)
        strong_neighs_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'strong_neighs')
        # Assigning a type to the variable 'weak_opps' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'weak_opps', strong_neighs_776)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'weak_opps' (line 240)
        weak_opps_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'weak_opps')
        # Assigning a type to the variable 'strong_opps' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'strong_opps', weak_opps_777)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'strong_opps' (line 240)
        strong_opps_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'strong_opps')
        # Assigning a type to the variable 'empties' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'empties', strong_opps_778)
        
        # Getting the type of 'square' (line 241)
        square_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'square')
        # Obtaining the member 'neighbours' of a type (line 241)
        neighbours_780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 25), square_779, 'neighbours')
        # Assigning a type to the variable 'neighbours_780' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'neighbours_780', neighbours_780)
        # Testing if the for loop is going to be iterated (line 241)
        # Testing the type of a for loop iterable (line 241)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 8), neighbours_780)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 241, 8), neighbours_780):
            # Getting the type of the for loop variable (line 241)
            for_loop_var_781 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 8), neighbours_780)
            # Assigning a type to the variable 'neighbour' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'neighbour', for_loop_var_781)
            # SSA begins for a for statement (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'neighbour' (line 242)
            neighbour_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'neighbour')
            # Obtaining the member 'color' of a type (line 242)
            color_783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 15), neighbour_782, 'color')
            # Getting the type of 'EMPTY' (line 242)
            EMPTY_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'EMPTY')
            # Applying the binary operator '==' (line 242)
            result_eq_785 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 15), '==', color_783, EMPTY_784)
            
            # Testing if the type of an if condition is none (line 242)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 242, 12), result_eq_785):
                
                # Assigning a Call to a Name (line 245):
                
                # Assigning a Call to a Name (line 245):
                
                # Call to find(...): (line 245)
                # Processing the call keyword arguments (line 245)
                kwargs_792 = {}
                # Getting the type of 'neighbour' (line 245)
                neighbour_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 32), 'neighbour', False)
                # Obtaining the member 'find' of a type (line 245)
                find_791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 32), neighbour_790, 'find')
                # Calling find(args, kwargs) (line 245)
                find_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 245, 32), find_791, *[], **kwargs_792)
                
                # Assigning a type to the variable 'neighbour_ref' (line 245)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'neighbour_ref', find_call_result_793)
                
                # Getting the type of 'neighbour_ref' (line 246)
                neighbour_ref_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'neighbour_ref')
                # Obtaining the member 'timestamp' of a type (line 246)
                timestamp_795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), neighbour_ref_794, 'timestamp')
                # Getting the type of 'TIMESTAMP' (line 246)
                TIMESTAMP_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 46), 'TIMESTAMP')
                # Applying the binary operator '!=' (line 246)
                result_ne_797 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 19), '!=', timestamp_795, TIMESTAMP_796)
                
                # Testing if the type of an if condition is none (line 246)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 246, 16), result_ne_797):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 246)
                    if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 16), result_ne_797)
                    # Assigning a type to the variable 'if_condition_798' (line 246)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'if_condition_798', if_condition_798)
                    # SSA begins for if statement (line 246)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Attribute (line 247):
                    
                    # Assigning a Name to a Attribute (line 247):
                    # Getting the type of 'TIMESTAMP' (line 247)
                    TIMESTAMP_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 46), 'TIMESTAMP')
                    # Getting the type of 'neighbour_ref' (line 247)
                    neighbour_ref_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'neighbour_ref')
                    # Setting the type of the member 'timestamp' of a type (line 247)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), neighbour_ref_800, 'timestamp', TIMESTAMP_799)
                    
                    # Assigning a Compare to a Name (line 248):
                    
                    # Assigning a Compare to a Name (line 248):
                    
                    # Getting the type of 'neighbour_ref' (line 248)
                    neighbour_ref_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'neighbour_ref')
                    # Obtaining the member 'liberties' of a type (line 248)
                    liberties_802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 28), neighbour_ref_801, 'liberties')
                    int_803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 55), 'int')
                    # Applying the binary operator '==' (line 248)
                    result_eq_804 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 28), '==', liberties_802, int_803)
                    
                    # Assigning a type to the variable 'weak' (line 248)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'weak', result_eq_804)
                    
                    # Getting the type of 'neighbour' (line 249)
                    neighbour_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 23), 'neighbour')
                    # Obtaining the member 'color' of a type (line 249)
                    color_806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 23), neighbour_805, 'color')
                    # Getting the type of 'self' (line 249)
                    self_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'self')
                    # Obtaining the member 'color' of a type (line 249)
                    color_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), self_807, 'color')
                    # Applying the binary operator '==' (line 249)
                    result_eq_809 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 23), '==', color_806, color_808)
                    
                    # Testing if the type of an if condition is none (line 249)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 20), result_eq_809):
                        # Getting the type of 'weak' (line 255)
                        weak_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'weak')
                        # Testing if the type of an if condition is none (line 255)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819):
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                        else:
                            
                            # Testing the type of an if condition (line 255)
                            if_condition_820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819)
                            # Assigning a type to the variable 'if_condition_820' (line 255)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'if_condition_820', if_condition_820)
                            # SSA begins for if statement (line 255)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'weak_opps' (line 256)
                            weak_opps_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps')
                            int_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 41), 'int')
                            # Applying the binary operator '+=' (line 256)
                            result_iadd_823 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 28), '+=', weak_opps_821, int_822)
                            # Assigning a type to the variable 'weak_opps' (line 256)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps', result_iadd_823)
                            
                            
                            # Call to remove(...): (line 257)
                            # Processing the call arguments (line 257)
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 49), 'neighbour_ref', False)
                            # Processing the call keyword arguments (line 257)
                            # Getting the type of 'False' (line 257)
                            False_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 71), 'False', False)
                            keyword_828 = False_827
                            kwargs_829 = {'update': keyword_828}
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'neighbour_ref', False)
                            # Obtaining the member 'remove' of a type (line 257)
                            remove_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 28), neighbour_ref_824, 'remove')
                            # Calling remove(args, kwargs) (line 257)
                            remove_call_result_830 = invoke(stypy.reporting.localization.Localization(__file__, 257, 28), remove_825, *[neighbour_ref_826], **kwargs_829)
                            
                            # SSA branch for the else part of an if statement (line 255)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                            # SSA join for if statement (line 255)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 249)
                        if_condition_810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 20), result_eq_809)
                        # Assigning a type to the variable 'if_condition_810' (line 249)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'if_condition_810', if_condition_810)
                        # SSA begins for if statement (line 249)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'weak' (line 250)
                        weak_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'weak')
                        # Testing if the type of an if condition is none (line 250)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 250, 24), weak_811):
                            
                            # Getting the type of 'strong_neighs' (line 253)
                            strong_neighs_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs')
                            int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'int')
                            # Applying the binary operator '+=' (line 253)
                            result_iadd_818 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 28), '+=', strong_neighs_816, int_817)
                            # Assigning a type to the variable 'strong_neighs' (line 253)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs', result_iadd_818)
                            
                        else:
                            
                            # Testing the type of an if condition (line 250)
                            if_condition_812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 24), weak_811)
                            # Assigning a type to the variable 'if_condition_812' (line 250)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'if_condition_812', if_condition_812)
                            # SSA begins for if statement (line 250)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'weak_neighs' (line 251)
                            weak_neighs_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'weak_neighs')
                            int_814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 43), 'int')
                            # Applying the binary operator '+=' (line 251)
                            result_iadd_815 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 28), '+=', weak_neighs_813, int_814)
                            # Assigning a type to the variable 'weak_neighs' (line 251)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'weak_neighs', result_iadd_815)
                            
                            # SSA branch for the else part of an if statement (line 250)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'strong_neighs' (line 253)
                            strong_neighs_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs')
                            int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'int')
                            # Applying the binary operator '+=' (line 253)
                            result_iadd_818 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 28), '+=', strong_neighs_816, int_817)
                            # Assigning a type to the variable 'strong_neighs' (line 253)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs', result_iadd_818)
                            
                            # SSA join for if statement (line 250)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA branch for the else part of an if statement (line 249)
                        module_type_store.open_ssa_branch('else')
                        # Getting the type of 'weak' (line 255)
                        weak_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'weak')
                        # Testing if the type of an if condition is none (line 255)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819):
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                        else:
                            
                            # Testing the type of an if condition (line 255)
                            if_condition_820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819)
                            # Assigning a type to the variable 'if_condition_820' (line 255)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'if_condition_820', if_condition_820)
                            # SSA begins for if statement (line 255)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'weak_opps' (line 256)
                            weak_opps_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps')
                            int_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 41), 'int')
                            # Applying the binary operator '+=' (line 256)
                            result_iadd_823 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 28), '+=', weak_opps_821, int_822)
                            # Assigning a type to the variable 'weak_opps' (line 256)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps', result_iadd_823)
                            
                            
                            # Call to remove(...): (line 257)
                            # Processing the call arguments (line 257)
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 49), 'neighbour_ref', False)
                            # Processing the call keyword arguments (line 257)
                            # Getting the type of 'False' (line 257)
                            False_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 71), 'False', False)
                            keyword_828 = False_827
                            kwargs_829 = {'update': keyword_828}
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'neighbour_ref', False)
                            # Obtaining the member 'remove' of a type (line 257)
                            remove_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 28), neighbour_ref_824, 'remove')
                            # Calling remove(args, kwargs) (line 257)
                            remove_call_result_830 = invoke(stypy.reporting.localization.Localization(__file__, 257, 28), remove_825, *[neighbour_ref_826], **kwargs_829)
                            
                            # SSA branch for the else part of an if statement (line 255)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                            # SSA join for if statement (line 255)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 249)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 246)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 242)
                if_condition_786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 12), result_eq_785)
                # Assigning a type to the variable 'if_condition_786' (line 242)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'if_condition_786', if_condition_786)
                # SSA begins for if statement (line 242)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'empties' (line 243)
                empties_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'empties')
                int_788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 27), 'int')
                # Applying the binary operator '+=' (line 243)
                result_iadd_789 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 16), '+=', empties_787, int_788)
                # Assigning a type to the variable 'empties' (line 243)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'empties', result_iadd_789)
                
                # SSA branch for the else part of an if statement (line 242)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 245):
                
                # Assigning a Call to a Name (line 245):
                
                # Call to find(...): (line 245)
                # Processing the call keyword arguments (line 245)
                kwargs_792 = {}
                # Getting the type of 'neighbour' (line 245)
                neighbour_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 32), 'neighbour', False)
                # Obtaining the member 'find' of a type (line 245)
                find_791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 32), neighbour_790, 'find')
                # Calling find(args, kwargs) (line 245)
                find_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 245, 32), find_791, *[], **kwargs_792)
                
                # Assigning a type to the variable 'neighbour_ref' (line 245)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'neighbour_ref', find_call_result_793)
                
                # Getting the type of 'neighbour_ref' (line 246)
                neighbour_ref_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'neighbour_ref')
                # Obtaining the member 'timestamp' of a type (line 246)
                timestamp_795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), neighbour_ref_794, 'timestamp')
                # Getting the type of 'TIMESTAMP' (line 246)
                TIMESTAMP_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 46), 'TIMESTAMP')
                # Applying the binary operator '!=' (line 246)
                result_ne_797 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 19), '!=', timestamp_795, TIMESTAMP_796)
                
                # Testing if the type of an if condition is none (line 246)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 246, 16), result_ne_797):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 246)
                    if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 16), result_ne_797)
                    # Assigning a type to the variable 'if_condition_798' (line 246)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'if_condition_798', if_condition_798)
                    # SSA begins for if statement (line 246)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Attribute (line 247):
                    
                    # Assigning a Name to a Attribute (line 247):
                    # Getting the type of 'TIMESTAMP' (line 247)
                    TIMESTAMP_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 46), 'TIMESTAMP')
                    # Getting the type of 'neighbour_ref' (line 247)
                    neighbour_ref_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'neighbour_ref')
                    # Setting the type of the member 'timestamp' of a type (line 247)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), neighbour_ref_800, 'timestamp', TIMESTAMP_799)
                    
                    # Assigning a Compare to a Name (line 248):
                    
                    # Assigning a Compare to a Name (line 248):
                    
                    # Getting the type of 'neighbour_ref' (line 248)
                    neighbour_ref_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'neighbour_ref')
                    # Obtaining the member 'liberties' of a type (line 248)
                    liberties_802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 28), neighbour_ref_801, 'liberties')
                    int_803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 55), 'int')
                    # Applying the binary operator '==' (line 248)
                    result_eq_804 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 28), '==', liberties_802, int_803)
                    
                    # Assigning a type to the variable 'weak' (line 248)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'weak', result_eq_804)
                    
                    # Getting the type of 'neighbour' (line 249)
                    neighbour_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 23), 'neighbour')
                    # Obtaining the member 'color' of a type (line 249)
                    color_806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 23), neighbour_805, 'color')
                    # Getting the type of 'self' (line 249)
                    self_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'self')
                    # Obtaining the member 'color' of a type (line 249)
                    color_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), self_807, 'color')
                    # Applying the binary operator '==' (line 249)
                    result_eq_809 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 23), '==', color_806, color_808)
                    
                    # Testing if the type of an if condition is none (line 249)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 20), result_eq_809):
                        # Getting the type of 'weak' (line 255)
                        weak_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'weak')
                        # Testing if the type of an if condition is none (line 255)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819):
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                        else:
                            
                            # Testing the type of an if condition (line 255)
                            if_condition_820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819)
                            # Assigning a type to the variable 'if_condition_820' (line 255)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'if_condition_820', if_condition_820)
                            # SSA begins for if statement (line 255)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'weak_opps' (line 256)
                            weak_opps_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps')
                            int_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 41), 'int')
                            # Applying the binary operator '+=' (line 256)
                            result_iadd_823 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 28), '+=', weak_opps_821, int_822)
                            # Assigning a type to the variable 'weak_opps' (line 256)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps', result_iadd_823)
                            
                            
                            # Call to remove(...): (line 257)
                            # Processing the call arguments (line 257)
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 49), 'neighbour_ref', False)
                            # Processing the call keyword arguments (line 257)
                            # Getting the type of 'False' (line 257)
                            False_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 71), 'False', False)
                            keyword_828 = False_827
                            kwargs_829 = {'update': keyword_828}
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'neighbour_ref', False)
                            # Obtaining the member 'remove' of a type (line 257)
                            remove_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 28), neighbour_ref_824, 'remove')
                            # Calling remove(args, kwargs) (line 257)
                            remove_call_result_830 = invoke(stypy.reporting.localization.Localization(__file__, 257, 28), remove_825, *[neighbour_ref_826], **kwargs_829)
                            
                            # SSA branch for the else part of an if statement (line 255)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                            # SSA join for if statement (line 255)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 249)
                        if_condition_810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 20), result_eq_809)
                        # Assigning a type to the variable 'if_condition_810' (line 249)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'if_condition_810', if_condition_810)
                        # SSA begins for if statement (line 249)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'weak' (line 250)
                        weak_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'weak')
                        # Testing if the type of an if condition is none (line 250)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 250, 24), weak_811):
                            
                            # Getting the type of 'strong_neighs' (line 253)
                            strong_neighs_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs')
                            int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'int')
                            # Applying the binary operator '+=' (line 253)
                            result_iadd_818 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 28), '+=', strong_neighs_816, int_817)
                            # Assigning a type to the variable 'strong_neighs' (line 253)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs', result_iadd_818)
                            
                        else:
                            
                            # Testing the type of an if condition (line 250)
                            if_condition_812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 24), weak_811)
                            # Assigning a type to the variable 'if_condition_812' (line 250)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'if_condition_812', if_condition_812)
                            # SSA begins for if statement (line 250)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'weak_neighs' (line 251)
                            weak_neighs_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'weak_neighs')
                            int_814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 43), 'int')
                            # Applying the binary operator '+=' (line 251)
                            result_iadd_815 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 28), '+=', weak_neighs_813, int_814)
                            # Assigning a type to the variable 'weak_neighs' (line 251)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'weak_neighs', result_iadd_815)
                            
                            # SSA branch for the else part of an if statement (line 250)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'strong_neighs' (line 253)
                            strong_neighs_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs')
                            int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'int')
                            # Applying the binary operator '+=' (line 253)
                            result_iadd_818 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 28), '+=', strong_neighs_816, int_817)
                            # Assigning a type to the variable 'strong_neighs' (line 253)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs', result_iadd_818)
                            
                            # SSA join for if statement (line 250)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA branch for the else part of an if statement (line 249)
                        module_type_store.open_ssa_branch('else')
                        # Getting the type of 'weak' (line 255)
                        weak_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'weak')
                        # Testing if the type of an if condition is none (line 255)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819):
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                        else:
                            
                            # Testing the type of an if condition (line 255)
                            if_condition_820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 24), weak_819)
                            # Assigning a type to the variable 'if_condition_820' (line 255)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'if_condition_820', if_condition_820)
                            # SSA begins for if statement (line 255)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'weak_opps' (line 256)
                            weak_opps_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps')
                            int_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 41), 'int')
                            # Applying the binary operator '+=' (line 256)
                            result_iadd_823 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 28), '+=', weak_opps_821, int_822)
                            # Assigning a type to the variable 'weak_opps' (line 256)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps', result_iadd_823)
                            
                            
                            # Call to remove(...): (line 257)
                            # Processing the call arguments (line 257)
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 49), 'neighbour_ref', False)
                            # Processing the call keyword arguments (line 257)
                            # Getting the type of 'False' (line 257)
                            False_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 71), 'False', False)
                            keyword_828 = False_827
                            kwargs_829 = {'update': keyword_828}
                            # Getting the type of 'neighbour_ref' (line 257)
                            neighbour_ref_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'neighbour_ref', False)
                            # Obtaining the member 'remove' of a type (line 257)
                            remove_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 28), neighbour_ref_824, 'remove')
                            # Calling remove(args, kwargs) (line 257)
                            remove_call_result_830 = invoke(stypy.reporting.localization.Localization(__file__, 257, 28), remove_825, *[neighbour_ref_826], **kwargs_829)
                            
                            # SSA branch for the else part of an if statement (line 255)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'strong_opps' (line 259)
                            strong_opps_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
                            int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
                            # Applying the binary operator '+=' (line 259)
                            result_iadd_833 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_831, int_832)
                            # Assigning a type to the variable 'strong_opps' (line 259)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_833)
                            
                            # SSA join for if statement (line 255)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 249)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 246)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 242)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to dupe(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_837 = {}
        # Getting the type of 'self' (line 260)
        self_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self', False)
        # Obtaining the member 'zobrist' of a type (line 260)
        zobrist_835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_834, 'zobrist')
        # Obtaining the member 'dupe' of a type (line 260)
        dupe_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), zobrist_835, 'dupe')
        # Calling dupe(args, kwargs) (line 260)
        dupe_call_result_838 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), dupe_836, *[], **kwargs_837)
        
        # Assigning a type to the variable 'dupe' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'dupe', dupe_call_result_838)
        
        # Assigning a Name to a Attribute (line 261):
        
        # Assigning a Name to a Attribute (line 261):
        # Getting the type of 'old_hash' (line 261)
        old_hash_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 28), 'old_hash')
        # Getting the type of 'self' (line 261)
        self_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
        # Obtaining the member 'zobrist' of a type (line 261)
        zobrist_841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_840, 'zobrist')
        # Setting the type of the member 'hash' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), zobrist_841, 'hash', old_hash_839)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'dupe' (line 262)
        dupe_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'dupe')
        # Applying the 'not' unary operator (line 262)
        result_not__843 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), 'not', dupe_842)
        
        
        # Call to bool(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Evaluating a boolean operation
        # Getting the type of 'empties' (line 263)
        empties_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'empties', False)
        # Getting the type of 'weak_opps' (line 263)
        weak_opps_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 31), 'weak_opps', False)
        # Applying the binary operator 'or' (line 263)
        result_or_keyword_847 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 20), 'or', empties_845, weak_opps_846)
        
        # Evaluating a boolean operation
        # Getting the type of 'strong_neighs' (line 263)
        strong_neighs_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'strong_neighs', False)
        
        # Evaluating a boolean operation
        # Getting the type of 'strong_opps' (line 263)
        strong_opps_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 64), 'strong_opps', False)
        # Getting the type of 'weak_neighs' (line 263)
        weak_neighs_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 79), 'weak_neighs', False)
        # Applying the binary operator 'or' (line 263)
        result_or_keyword_851 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 64), 'or', strong_opps_849, weak_neighs_850)
        
        # Applying the binary operator 'and' (line 263)
        result_and_keyword_852 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 45), 'and', strong_neighs_848, result_or_keyword_851)
        
        # Applying the binary operator 'or' (line 263)
        result_or_keyword_853 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 20), 'or', result_or_keyword_847, result_and_keyword_852)
        
        # Processing the call keyword arguments (line 263)
        kwargs_854 = {}
        # Getting the type of 'bool' (line 263)
        bool_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'bool', False)
        # Calling bool(args, kwargs) (line 263)
        bool_call_result_855 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), bool_844, *[result_or_keyword_853], **kwargs_854)
        
        # Applying the binary operator 'and' (line 262)
        result_and_keyword_856 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), 'and', result_not__843, bool_call_result_855)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type', result_and_keyword_856)
        
        # ################# End of 'useful(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'useful' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'useful'
        return stypy_return_type_857


    @norecursion
    def useful_moves(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'useful_moves'
        module_type_store = module_type_store.open_function_context('useful_moves', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.useful_moves.__dict__.__setitem__('stypy_localization', localization)
        Board.useful_moves.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.useful_moves.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.useful_moves.__dict__.__setitem__('stypy_function_name', 'Board.useful_moves')
        Board.useful_moves.__dict__.__setitem__('stypy_param_names_list', [])
        Board.useful_moves.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.useful_moves.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.useful_moves.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.useful_moves.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.useful_moves.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.useful_moves.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.useful_moves', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'useful_moves', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'useful_moves(...)' code ##################

        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 266)
        self_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'self')
        # Obtaining the member 'emptyset' of a type (line 266)
        emptyset_865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 31), self_864, 'emptyset')
        # Obtaining the member 'empties' of a type (line 266)
        empties_866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 31), emptyset_865, 'empties')
        comprehension_867 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 16), empties_866)
        # Assigning a type to the variable 'pos' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'pos', comprehension_867)
        
        # Call to useful(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'pos' (line 266)
        pos_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 68), 'pos', False)
        # Processing the call keyword arguments (line 266)
        kwargs_862 = {}
        # Getting the type of 'self' (line 266)
        self_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 56), 'self', False)
        # Obtaining the member 'useful' of a type (line 266)
        useful_860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 56), self_859, 'useful')
        # Calling useful(args, kwargs) (line 266)
        useful_call_result_863 = invoke(stypy.reporting.localization.Localization(__file__, 266, 56), useful_860, *[pos_861], **kwargs_862)
        
        # Getting the type of 'pos' (line 266)
        pos_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'pos')
        list_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 16), list_868, pos_858)
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type', list_868)
        
        # ################# End of 'useful_moves(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'useful_moves' in the type store
        # Getting the type of 'stypy_return_type' (line 265)
        stypy_return_type_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_869)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'useful_moves'
        return stypy_return_type_869


    @norecursion
    def replay(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'replay'
        module_type_store = module_type_store.open_function_context('replay', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.replay.__dict__.__setitem__('stypy_localization', localization)
        Board.replay.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.replay.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.replay.__dict__.__setitem__('stypy_function_name', 'Board.replay')
        Board.replay.__dict__.__setitem__('stypy_param_names_list', ['history'])
        Board.replay.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.replay.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.replay.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.replay.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.replay.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.replay.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.replay', ['history'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'replay', localization, ['history'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'replay(...)' code ##################

        
        # Getting the type of 'history' (line 269)
        history_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'history')
        # Assigning a type to the variable 'history_870' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'history_870', history_870)
        # Testing if the for loop is going to be iterated (line 269)
        # Testing the type of a for loop iterable (line 269)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 8), history_870)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 269, 8), history_870):
            # Getting the type of the for loop variable (line 269)
            for_loop_var_871 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 8), history_870)
            # Assigning a type to the variable 'pos' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'pos', for_loop_var_871)
            # SSA begins for a for statement (line 269)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to move(...): (line 270)
            # Processing the call arguments (line 270)
            # Getting the type of 'pos' (line 270)
            pos_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'pos', False)
            # Processing the call keyword arguments (line 270)
            kwargs_875 = {}
            # Getting the type of 'self' (line 270)
            self_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'self', False)
            # Obtaining the member 'move' of a type (line 270)
            move_873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), self_872, 'move')
            # Calling move(args, kwargs) (line 270)
            move_call_result_876 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), move_873, *[pos_874], **kwargs_875)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'replay(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'replay' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'replay'
        return stypy_return_type_877


    @norecursion
    def score(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score'
        module_type_store = module_type_store.open_function_context('score', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.score.__dict__.__setitem__('stypy_localization', localization)
        Board.score.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.score.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.score.__dict__.__setitem__('stypy_function_name', 'Board.score')
        Board.score.__dict__.__setitem__('stypy_param_names_list', ['color'])
        Board.score.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.score.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.score.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.score.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.score.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.score.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.score', ['color'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score', localization, ['color'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score(...)' code ##################

        
        # Getting the type of 'color' (line 273)
        color_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'color')
        # Getting the type of 'WHITE' (line 273)
        WHITE_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'WHITE')
        # Applying the binary operator '==' (line 273)
        result_eq_880 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), '==', color_878, WHITE_879)
        
        # Testing if the type of an if condition is none (line 273)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_880):
            
            # Assigning a Attribute to a Name (line 276):
            
            # Assigning a Attribute to a Name (line 276):
            # Getting the type of 'self' (line 276)
            self_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'self')
            # Obtaining the member 'white_dead' of a type (line 276)
            white_dead_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 20), self_886, 'white_dead')
            # Assigning a type to the variable 'count' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'count', white_dead_887)
        else:
            
            # Testing the type of an if condition (line 273)
            if_condition_881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_880)
            # Assigning a type to the variable 'if_condition_881' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_881', if_condition_881)
            # SSA begins for if statement (line 273)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 274):
            
            # Assigning a BinOp to a Name (line 274):
            # Getting the type of 'KOMI' (line 274)
            KOMI_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'KOMI')
            # Getting the type of 'self' (line 274)
            self_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 27), 'self')
            # Obtaining the member 'black_dead' of a type (line 274)
            black_dead_884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 27), self_883, 'black_dead')
            # Applying the binary operator '+' (line 274)
            result_add_885 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 20), '+', KOMI_882, black_dead_884)
            
            # Assigning a type to the variable 'count' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'count', result_add_885)
            # SSA branch for the else part of an if statement (line 273)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 276):
            
            # Assigning a Attribute to a Name (line 276):
            # Getting the type of 'self' (line 276)
            self_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'self')
            # Obtaining the member 'white_dead' of a type (line 276)
            white_dead_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 20), self_886, 'white_dead')
            # Assigning a type to the variable 'count' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'count', white_dead_887)
            # SSA join for if statement (line 273)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 277)
        self_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'self')
        # Obtaining the member 'squares' of a type (line 277)
        squares_889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 22), self_888, 'squares')
        # Assigning a type to the variable 'squares_889' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'squares_889', squares_889)
        # Testing if the for loop is going to be iterated (line 277)
        # Testing the type of a for loop iterable (line 277)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 8), squares_889)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 277, 8), squares_889):
            # Getting the type of the for loop variable (line 277)
            for_loop_var_890 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 8), squares_889)
            # Assigning a type to the variable 'square' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'square', for_loop_var_890)
            # SSA begins for a for statement (line 277)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Attribute to a Name (line 278):
            
            # Assigning a Attribute to a Name (line 278):
            # Getting the type of 'square' (line 278)
            square_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'square')
            # Obtaining the member 'color' of a type (line 278)
            color_892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 26), square_891, 'color')
            # Assigning a type to the variable 'squarecolor' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'squarecolor', color_892)
            
            # Getting the type of 'squarecolor' (line 279)
            squarecolor_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'squarecolor')
            # Getting the type of 'color' (line 279)
            color_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'color')
            # Applying the binary operator '==' (line 279)
            result_eq_895 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 15), '==', squarecolor_893, color_894)
            
            # Testing if the type of an if condition is none (line 279)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 279, 12), result_eq_895):
                
                # Getting the type of 'squarecolor' (line 281)
                squarecolor_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'squarecolor')
                # Getting the type of 'EMPTY' (line 281)
                EMPTY_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'EMPTY')
                # Applying the binary operator '==' (line 281)
                result_eq_902 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 17), '==', squarecolor_900, EMPTY_901)
                
                # Testing if the type of an if condition is none (line 281)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 17), result_eq_902):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 281)
                    if_condition_903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 17), result_eq_902)
                    # Assigning a type to the variable 'if_condition_903' (line 281)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'if_condition_903', if_condition_903)
                    # SSA begins for if statement (line 281)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Num to a Name (line 282):
                    
                    # Assigning a Num to a Name (line 282):
                    int_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 27), 'int')
                    # Assigning a type to the variable 'surround' (line 282)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'surround', int_904)
                    
                    # Getting the type of 'square' (line 283)
                    square_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 33), 'square')
                    # Obtaining the member 'neighbours' of a type (line 283)
                    neighbours_906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 33), square_905, 'neighbours')
                    # Assigning a type to the variable 'neighbours_906' (line 283)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'neighbours_906', neighbours_906)
                    # Testing if the for loop is going to be iterated (line 283)
                    # Testing the type of a for loop iterable (line 283)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_906)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_906):
                        # Getting the type of the for loop variable (line 283)
                        for_loop_var_907 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_906)
                        # Assigning a type to the variable 'neighbour' (line 283)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'neighbour', for_loop_var_907)
                        # SSA begins for a for statement (line 283)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'neighbour' (line 284)
                        neighbour_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'neighbour')
                        # Obtaining the member 'color' of a type (line 284)
                        color_909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), neighbour_908, 'color')
                        # Getting the type of 'color' (line 284)
                        color_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 42), 'color')
                        # Applying the binary operator '==' (line 284)
                        result_eq_911 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 23), '==', color_909, color_910)
                        
                        # Testing if the type of an if condition is none (line 284)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 284, 20), result_eq_911):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 284)
                            if_condition_912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 20), result_eq_911)
                            # Assigning a type to the variable 'if_condition_912' (line 284)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'if_condition_912', if_condition_912)
                            # SSA begins for if statement (line 284)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'surround' (line 285)
                            surround_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'surround')
                            int_914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 36), 'int')
                            # Applying the binary operator '+=' (line 285)
                            result_iadd_915 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 24), '+=', surround_913, int_914)
                            # Assigning a type to the variable 'surround' (line 285)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'surround', result_iadd_915)
                            
                            # SSA join for if statement (line 284)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    # Getting the type of 'surround' (line 286)
                    surround_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'surround')
                    
                    # Call to len(...): (line 286)
                    # Processing the call arguments (line 286)
                    # Getting the type of 'square' (line 286)
                    square_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), 'square', False)
                    # Obtaining the member 'neighbours' of a type (line 286)
                    neighbours_919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 35), square_918, 'neighbours')
                    # Processing the call keyword arguments (line 286)
                    kwargs_920 = {}
                    # Getting the type of 'len' (line 286)
                    len_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'len', False)
                    # Calling len(args, kwargs) (line 286)
                    len_call_result_921 = invoke(stypy.reporting.localization.Localization(__file__, 286, 31), len_917, *[neighbours_919], **kwargs_920)
                    
                    # Applying the binary operator '==' (line 286)
                    result_eq_922 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 19), '==', surround_916, len_call_result_921)
                    
                    # Testing if the type of an if condition is none (line 286)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 286, 16), result_eq_922):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 286)
                        if_condition_923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 16), result_eq_922)
                        # Assigning a type to the variable 'if_condition_923' (line 286)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'if_condition_923', if_condition_923)
                        # SSA begins for if statement (line 286)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'count' (line 287)
                        count_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'count')
                        int_925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 29), 'int')
                        # Applying the binary operator '+=' (line 287)
                        result_iadd_926 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 20), '+=', count_924, int_925)
                        # Assigning a type to the variable 'count' (line 287)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'count', result_iadd_926)
                        
                        # SSA join for if statement (line 286)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 281)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 279)
                if_condition_896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 12), result_eq_895)
                # Assigning a type to the variable 'if_condition_896' (line 279)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'if_condition_896', if_condition_896)
                # SSA begins for if statement (line 279)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'count' (line 280)
                count_897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'count')
                int_898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 25), 'int')
                # Applying the binary operator '+=' (line 280)
                result_iadd_899 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '+=', count_897, int_898)
                # Assigning a type to the variable 'count' (line 280)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'count', result_iadd_899)
                
                # SSA branch for the else part of an if statement (line 279)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'squarecolor' (line 281)
                squarecolor_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'squarecolor')
                # Getting the type of 'EMPTY' (line 281)
                EMPTY_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'EMPTY')
                # Applying the binary operator '==' (line 281)
                result_eq_902 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 17), '==', squarecolor_900, EMPTY_901)
                
                # Testing if the type of an if condition is none (line 281)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 17), result_eq_902):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 281)
                    if_condition_903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 17), result_eq_902)
                    # Assigning a type to the variable 'if_condition_903' (line 281)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'if_condition_903', if_condition_903)
                    # SSA begins for if statement (line 281)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Num to a Name (line 282):
                    
                    # Assigning a Num to a Name (line 282):
                    int_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 27), 'int')
                    # Assigning a type to the variable 'surround' (line 282)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'surround', int_904)
                    
                    # Getting the type of 'square' (line 283)
                    square_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 33), 'square')
                    # Obtaining the member 'neighbours' of a type (line 283)
                    neighbours_906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 33), square_905, 'neighbours')
                    # Assigning a type to the variable 'neighbours_906' (line 283)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'neighbours_906', neighbours_906)
                    # Testing if the for loop is going to be iterated (line 283)
                    # Testing the type of a for loop iterable (line 283)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_906)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_906):
                        # Getting the type of the for loop variable (line 283)
                        for_loop_var_907 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_906)
                        # Assigning a type to the variable 'neighbour' (line 283)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'neighbour', for_loop_var_907)
                        # SSA begins for a for statement (line 283)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'neighbour' (line 284)
                        neighbour_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'neighbour')
                        # Obtaining the member 'color' of a type (line 284)
                        color_909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), neighbour_908, 'color')
                        # Getting the type of 'color' (line 284)
                        color_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 42), 'color')
                        # Applying the binary operator '==' (line 284)
                        result_eq_911 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 23), '==', color_909, color_910)
                        
                        # Testing if the type of an if condition is none (line 284)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 284, 20), result_eq_911):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 284)
                            if_condition_912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 20), result_eq_911)
                            # Assigning a type to the variable 'if_condition_912' (line 284)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'if_condition_912', if_condition_912)
                            # SSA begins for if statement (line 284)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'surround' (line 285)
                            surround_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'surround')
                            int_914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 36), 'int')
                            # Applying the binary operator '+=' (line 285)
                            result_iadd_915 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 24), '+=', surround_913, int_914)
                            # Assigning a type to the variable 'surround' (line 285)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'surround', result_iadd_915)
                            
                            # SSA join for if statement (line 284)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    # Getting the type of 'surround' (line 286)
                    surround_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'surround')
                    
                    # Call to len(...): (line 286)
                    # Processing the call arguments (line 286)
                    # Getting the type of 'square' (line 286)
                    square_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), 'square', False)
                    # Obtaining the member 'neighbours' of a type (line 286)
                    neighbours_919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 35), square_918, 'neighbours')
                    # Processing the call keyword arguments (line 286)
                    kwargs_920 = {}
                    # Getting the type of 'len' (line 286)
                    len_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'len', False)
                    # Calling len(args, kwargs) (line 286)
                    len_call_result_921 = invoke(stypy.reporting.localization.Localization(__file__, 286, 31), len_917, *[neighbours_919], **kwargs_920)
                    
                    # Applying the binary operator '==' (line 286)
                    result_eq_922 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 19), '==', surround_916, len_call_result_921)
                    
                    # Testing if the type of an if condition is none (line 286)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 286, 16), result_eq_922):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 286)
                        if_condition_923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 16), result_eq_922)
                        # Assigning a type to the variable 'if_condition_923' (line 286)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'if_condition_923', if_condition_923)
                        # SSA begins for if statement (line 286)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'count' (line 287)
                        count_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'count')
                        int_925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 29), 'int')
                        # Applying the binary operator '+=' (line 287)
                        result_iadd_926 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 20), '+=', count_924, int_925)
                        # Assigning a type to the variable 'count' (line 287)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'count', result_iadd_926)
                        
                        # SSA join for if statement (line 286)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 281)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 279)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'count' (line 288)
        count_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'count')
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', count_927)
        
        # ################# End of 'score(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score'
        return stypy_return_type_928


    @norecursion
    def check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.check.__dict__.__setitem__('stypy_localization', localization)
        Board.check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.check.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.check.__dict__.__setitem__('stypy_function_name', 'Board.check')
        Board.check.__dict__.__setitem__('stypy_param_names_list', [])
        Board.check.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.check.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.check.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.check.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.check', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Getting the type of 'self' (line 291)
        self_929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 22), 'self')
        # Obtaining the member 'squares' of a type (line 291)
        squares_930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 22), self_929, 'squares')
        # Assigning a type to the variable 'squares_930' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'squares_930', squares_930)
        # Testing if the for loop is going to be iterated (line 291)
        # Testing the type of a for loop iterable (line 291)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 291, 8), squares_930)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 291, 8), squares_930):
            # Getting the type of the for loop variable (line 291)
            for_loop_var_931 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 291, 8), squares_930)
            # Assigning a type to the variable 'square' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'square', for_loop_var_931)
            # SSA begins for a for statement (line 291)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'square' (line 292)
            square_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'square')
            # Obtaining the member 'color' of a type (line 292)
            color_933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), square_932, 'color')
            # Getting the type of 'EMPTY' (line 292)
            EMPTY_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'EMPTY')
            # Applying the binary operator '==' (line 292)
            result_eq_935 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), '==', color_933, EMPTY_934)
            
            # Testing if the type of an if condition is none (line 292)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 292, 12), result_eq_935):
                pass
            else:
                
                # Testing the type of an if condition (line 292)
                if_condition_936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 12), result_eq_935)
                # Assigning a type to the variable 'if_condition_936' (line 292)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'if_condition_936', if_condition_936)
                # SSA begins for if statement (line 292)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 292)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 295):
            
            # Assigning a Call to a Name (line 295):
            
            # Call to set(...): (line 295)
            # Processing the call arguments (line 295)
            
            # Obtaining an instance of the builtin type 'list' (line 295)
            list_938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 295)
            # Adding element type (line 295)
            # Getting the type of 'square' (line 295)
            square_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 28), 'square', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 27), list_938, square_939)
            
            # Processing the call keyword arguments (line 295)
            kwargs_940 = {}
            # Getting the type of 'set' (line 295)
            set_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'set', False)
            # Calling set(args, kwargs) (line 295)
            set_call_result_941 = invoke(stypy.reporting.localization.Localization(__file__, 295, 23), set_937, *[list_938], **kwargs_940)
            
            # Assigning a type to the variable 'members1' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'members1', set_call_result_941)
            
            # Assigning a Name to a Name (line 296):
            
            # Assigning a Name to a Name (line 296):
            # Getting the type of 'True' (line 296)
            True_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'True')
            # Assigning a type to the variable 'changed' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'changed', True_942)
            
            # Getting the type of 'changed' (line 297)
            changed_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 18), 'changed')
            # Assigning a type to the variable 'changed_943' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'changed_943', changed_943)
            # Testing if the while is going to be iterated (line 297)
            # Testing the type of an if condition (line 297)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 12), changed_943)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 297, 12), changed_943):
                # SSA begins for while statement (line 297)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a Name to a Name (line 298):
                
                # Assigning a Name to a Name (line 298):
                # Getting the type of 'False' (line 298)
                False_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 26), 'False')
                # Assigning a type to the variable 'changed' (line 298)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'changed', False_944)
                
                
                # Call to copy(...): (line 299)
                # Processing the call keyword arguments (line 299)
                kwargs_947 = {}
                # Getting the type of 'members1' (line 299)
                members1_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 30), 'members1', False)
                # Obtaining the member 'copy' of a type (line 299)
                copy_946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 30), members1_945, 'copy')
                # Calling copy(args, kwargs) (line 299)
                copy_call_result_948 = invoke(stypy.reporting.localization.Localization(__file__, 299, 30), copy_946, *[], **kwargs_947)
                
                # Assigning a type to the variable 'copy_call_result_948' (line 299)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'copy_call_result_948', copy_call_result_948)
                # Testing if the for loop is going to be iterated (line 299)
                # Testing the type of a for loop iterable (line 299)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 16), copy_call_result_948)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 299, 16), copy_call_result_948):
                    # Getting the type of the for loop variable (line 299)
                    for_loop_var_949 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 16), copy_call_result_948)
                    # Assigning a type to the variable 'member' (line 299)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'member', for_loop_var_949)
                    # SSA begins for a for statement (line 299)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'member' (line 300)
                    member_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 37), 'member')
                    # Obtaining the member 'neighbours' of a type (line 300)
                    neighbours_951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 37), member_950, 'neighbours')
                    # Assigning a type to the variable 'neighbours_951' (line 300)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'neighbours_951', neighbours_951)
                    # Testing if the for loop is going to be iterated (line 300)
                    # Testing the type of a for loop iterable (line 300)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 20), neighbours_951)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 300, 20), neighbours_951):
                        # Getting the type of the for loop variable (line 300)
                        for_loop_var_952 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 20), neighbours_951)
                        # Assigning a type to the variable 'neighbour' (line 300)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'neighbour', for_loop_var_952)
                        # SSA begins for a for statement (line 300)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'neighbour' (line 301)
                        neighbour_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 27), 'neighbour')
                        # Obtaining the member 'color' of a type (line 301)
                        color_954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 27), neighbour_953, 'color')
                        # Getting the type of 'square' (line 301)
                        square_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 46), 'square')
                        # Obtaining the member 'color' of a type (line 301)
                        color_956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 46), square_955, 'color')
                        # Applying the binary operator '==' (line 301)
                        result_eq_957 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 27), '==', color_954, color_956)
                        
                        
                        # Getting the type of 'neighbour' (line 301)
                        neighbour_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 63), 'neighbour')
                        # Getting the type of 'members1' (line 301)
                        members1_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 80), 'members1')
                        # Applying the binary operator 'notin' (line 301)
                        result_contains_960 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 63), 'notin', neighbour_958, members1_959)
                        
                        # Applying the binary operator 'and' (line 301)
                        result_and_keyword_961 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 27), 'and', result_eq_957, result_contains_960)
                        
                        # Testing if the type of an if condition is none (line 301)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 301, 24), result_and_keyword_961):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 301)
                            if_condition_962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 24), result_and_keyword_961)
                            # Assigning a type to the variable 'if_condition_962' (line 301)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'if_condition_962', if_condition_962)
                            # SSA begins for if statement (line 301)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 302):
                            
                            # Assigning a Name to a Name (line 302):
                            # Getting the type of 'True' (line 302)
                            True_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'True')
                            # Assigning a type to the variable 'changed' (line 302)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'changed', True_963)
                            
                            # Call to add(...): (line 303)
                            # Processing the call arguments (line 303)
                            # Getting the type of 'neighbour' (line 303)
                            neighbour_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 41), 'neighbour', False)
                            # Processing the call keyword arguments (line 303)
                            kwargs_967 = {}
                            # Getting the type of 'members1' (line 303)
                            members1_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 28), 'members1', False)
                            # Obtaining the member 'add' of a type (line 303)
                            add_965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 28), members1_964, 'add')
                            # Calling add(args, kwargs) (line 303)
                            add_call_result_968 = invoke(stypy.reporting.localization.Localization(__file__, 303, 28), add_965, *[neighbour_966], **kwargs_967)
                            
                            # SSA join for if statement (line 301)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for while statement (line 297)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 304):
            
            # Assigning a Call to a Name (line 304):
            
            # Call to set(...): (line 304)
            # Processing the call keyword arguments (line 304)
            kwargs_970 = {}
            # Getting the type of 'set' (line 304)
            set_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'set', False)
            # Calling set(args, kwargs) (line 304)
            set_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 304, 25), set_969, *[], **kwargs_970)
            
            # Assigning a type to the variable 'liberties1' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'liberties1', set_call_result_971)
            
            # Getting the type of 'members1' (line 305)
            members1_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 26), 'members1')
            # Assigning a type to the variable 'members1_972' (line 305)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'members1_972', members1_972)
            # Testing if the for loop is going to be iterated (line 305)
            # Testing the type of a for loop iterable (line 305)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 305, 12), members1_972)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 305, 12), members1_972):
                # Getting the type of the for loop variable (line 305)
                for_loop_var_973 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 305, 12), members1_972)
                # Assigning a type to the variable 'member' (line 305)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'member', for_loop_var_973)
                # SSA begins for a for statement (line 305)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'member' (line 306)
                member_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 'member')
                # Obtaining the member 'neighbours' of a type (line 306)
                neighbours_975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 33), member_974, 'neighbours')
                # Assigning a type to the variable 'neighbours_975' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'neighbours_975', neighbours_975)
                # Testing if the for loop is going to be iterated (line 306)
                # Testing the type of a for loop iterable (line 306)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 306, 16), neighbours_975)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 306, 16), neighbours_975):
                    # Getting the type of the for loop variable (line 306)
                    for_loop_var_976 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 306, 16), neighbours_975)
                    # Assigning a type to the variable 'neighbour' (line 306)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'neighbour', for_loop_var_976)
                    # SSA begins for a for statement (line 306)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'neighbour' (line 307)
                    neighbour_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'neighbour')
                    # Obtaining the member 'color' of a type (line 307)
                    color_978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 23), neighbour_977, 'color')
                    # Getting the type of 'EMPTY' (line 307)
                    EMPTY_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 42), 'EMPTY')
                    # Applying the binary operator '==' (line 307)
                    result_eq_980 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 23), '==', color_978, EMPTY_979)
                    
                    # Testing if the type of an if condition is none (line 307)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 307, 20), result_eq_980):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 307)
                        if_condition_981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 20), result_eq_980)
                        # Assigning a type to the variable 'if_condition_981' (line 307)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'if_condition_981', if_condition_981)
                        # SSA begins for if statement (line 307)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to add(...): (line 308)
                        # Processing the call arguments (line 308)
                        # Getting the type of 'neighbour' (line 308)
                        neighbour_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'neighbour', False)
                        # Obtaining the member 'pos' of a type (line 308)
                        pos_985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 39), neighbour_984, 'pos')
                        # Processing the call keyword arguments (line 308)
                        kwargs_986 = {}
                        # Getting the type of 'liberties1' (line 308)
                        liberties1_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'liberties1', False)
                        # Obtaining the member 'add' of a type (line 308)
                        add_983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 24), liberties1_982, 'add')
                        # Calling add(args, kwargs) (line 308)
                        add_call_result_987 = invoke(stypy.reporting.localization.Localization(__file__, 308, 24), add_983, *[pos_985], **kwargs_986)
                        
                        # SSA join for if statement (line 307)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 310):
            
            # Assigning a Call to a Name (line 310):
            
            # Call to find(...): (line 310)
            # Processing the call keyword arguments (line 310)
            kwargs_990 = {}
            # Getting the type of 'square' (line 310)
            square_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'square', False)
            # Obtaining the member 'find' of a type (line 310)
            find_989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), square_988, 'find')
            # Calling find(args, kwargs) (line 310)
            find_call_result_991 = invoke(stypy.reporting.localization.Localization(__file__, 310, 19), find_989, *[], **kwargs_990)
            
            # Assigning a type to the variable 'root' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'root', find_call_result_991)
            
            # Assigning a Call to a Name (line 315):
            
            # Assigning a Call to a Name (line 315):
            
            # Call to set(...): (line 315)
            # Processing the call keyword arguments (line 315)
            kwargs_993 = {}
            # Getting the type of 'set' (line 315)
            set_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'set', False)
            # Calling set(args, kwargs) (line 315)
            set_call_result_994 = invoke(stypy.reporting.localization.Localization(__file__, 315, 23), set_992, *[], **kwargs_993)
            
            # Assigning a type to the variable 'members2' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'members2', set_call_result_994)
            
            # Getting the type of 'self' (line 316)
            self_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'self')
            # Obtaining the member 'squares' of a type (line 316)
            squares_996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 27), self_995, 'squares')
            # Assigning a type to the variable 'squares_996' (line 316)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'squares_996', squares_996)
            # Testing if the for loop is going to be iterated (line 316)
            # Testing the type of a for loop iterable (line 316)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 316, 12), squares_996)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 316, 12), squares_996):
                # Getting the type of the for loop variable (line 316)
                for_loop_var_997 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 316, 12), squares_996)
                # Assigning a type to the variable 'square2' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'square2', for_loop_var_997)
                # SSA begins for a for statement (line 316)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'square2' (line 317)
                square2_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'square2')
                # Obtaining the member 'color' of a type (line 317)
                color_999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 19), square2_998, 'color')
                # Getting the type of 'EMPTY' (line 317)
                EMPTY_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 36), 'EMPTY')
                # Applying the binary operator '!=' (line 317)
                result_ne_1001 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 19), '!=', color_999, EMPTY_1000)
                
                
                
                # Call to find(...): (line 317)
                # Processing the call keyword arguments (line 317)
                kwargs_1004 = {}
                # Getting the type of 'square2' (line 317)
                square2_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 46), 'square2', False)
                # Obtaining the member 'find' of a type (line 317)
                find_1003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 46), square2_1002, 'find')
                # Calling find(args, kwargs) (line 317)
                find_call_result_1005 = invoke(stypy.reporting.localization.Localization(__file__, 317, 46), find_1003, *[], **kwargs_1004)
                
                # Getting the type of 'root' (line 317)
                root_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 64), 'root')
                # Applying the binary operator '==' (line 317)
                result_eq_1007 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 46), '==', find_call_result_1005, root_1006)
                
                # Applying the binary operator 'and' (line 317)
                result_and_keyword_1008 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 19), 'and', result_ne_1001, result_eq_1007)
                
                # Testing if the type of an if condition is none (line 317)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 317, 16), result_and_keyword_1008):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 317)
                    if_condition_1009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 16), result_and_keyword_1008)
                    # Assigning a type to the variable 'if_condition_1009' (line 317)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'if_condition_1009', if_condition_1009)
                    # SSA begins for if statement (line 317)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to add(...): (line 318)
                    # Processing the call arguments (line 318)
                    # Getting the type of 'square2' (line 318)
                    square2_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 33), 'square2', False)
                    # Processing the call keyword arguments (line 318)
                    kwargs_1013 = {}
                    # Getting the type of 'members2' (line 318)
                    members2_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'members2', False)
                    # Obtaining the member 'add' of a type (line 318)
                    add_1011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 20), members2_1010, 'add')
                    # Calling add(args, kwargs) (line 318)
                    add_call_result_1014 = invoke(stypy.reporting.localization.Localization(__file__, 318, 20), add_1011, *[square2_1012], **kwargs_1013)
                    
                    # SSA join for if statement (line 317)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Attribute to a Name (line 320):
            
            # Assigning a Attribute to a Name (line 320):
            # Getting the type of 'root' (line 320)
            root_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), 'root')
            # Obtaining the member 'liberties' of a type (line 320)
            liberties_1016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 25), root_1015, 'liberties')
            # Assigning a type to the variable 'liberties2' (line 320)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'liberties2', liberties_1016)
            # Evaluating assert statement condition
            
            # Getting the type of 'members1' (line 324)
            members1_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'members1')
            # Getting the type of 'members2' (line 324)
            members2_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 31), 'members2')
            # Applying the binary operator '==' (line 324)
            result_eq_1019 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 19), '==', members1_1017, members2_1018)
            
            assert_1020 = result_eq_1019
            # Assigning a type to the variable 'assert_1020' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'assert_1020', result_eq_1019)
            # Evaluating assert statement condition
            
            
            # Call to len(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'liberties1' (line 325)
            liberties1_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), 'liberties1', False)
            # Processing the call keyword arguments (line 325)
            kwargs_1023 = {}
            # Getting the type of 'len' (line 325)
            len_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'len', False)
            # Calling len(args, kwargs) (line 325)
            len_call_result_1024 = invoke(stypy.reporting.localization.Localization(__file__, 325, 19), len_1021, *[liberties1_1022], **kwargs_1023)
            
            # Getting the type of 'liberties2' (line 325)
            liberties2_1025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'liberties2')
            # Applying the binary operator '==' (line 325)
            result_eq_1026 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 19), '==', len_call_result_1024, liberties2_1025)
            
            assert_1027 = result_eq_1026
            # Assigning a type to the variable 'assert_1027' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'assert_1027', result_eq_1026)
            
            # Assigning a Call to a Name (line 328):
            
            # Assigning a Call to a Name (line 328):
            
            # Call to set(...): (line 328)
            # Processing the call arguments (line 328)
            # Getting the type of 'self' (line 328)
            self_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'self', False)
            # Obtaining the member 'emptyset' of a type (line 328)
            emptyset_1030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 27), self_1029, 'emptyset')
            # Obtaining the member 'empties' of a type (line 328)
            empties_1031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 27), emptyset_1030, 'empties')
            # Processing the call keyword arguments (line 328)
            kwargs_1032 = {}
            # Getting the type of 'set' (line 328)
            set_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), 'set', False)
            # Calling set(args, kwargs) (line 328)
            set_call_result_1033 = invoke(stypy.reporting.localization.Localization(__file__, 328, 23), set_1028, *[empties_1031], **kwargs_1032)
            
            # Assigning a type to the variable 'empties1' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'empties1', set_call_result_1033)
            
            # Assigning a Call to a Name (line 330):
            
            # Assigning a Call to a Name (line 330):
            
            # Call to set(...): (line 330)
            # Processing the call keyword arguments (line 330)
            kwargs_1035 = {}
            # Getting the type of 'set' (line 330)
            set_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 23), 'set', False)
            # Calling set(args, kwargs) (line 330)
            set_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 330, 23), set_1034, *[], **kwargs_1035)
            
            # Assigning a type to the variable 'empties2' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'empties2', set_call_result_1036)
            
            # Getting the type of 'self' (line 331)
            self_1037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'self')
            # Obtaining the member 'squares' of a type (line 331)
            squares_1038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 26), self_1037, 'squares')
            # Assigning a type to the variable 'squares_1038' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'squares_1038', squares_1038)
            # Testing if the for loop is going to be iterated (line 331)
            # Testing the type of a for loop iterable (line 331)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 331, 12), squares_1038)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 331, 12), squares_1038):
                # Getting the type of the for loop variable (line 331)
                for_loop_var_1039 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 331, 12), squares_1038)
                # Assigning a type to the variable 'square' (line 331)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'square', for_loop_var_1039)
                # SSA begins for a for statement (line 331)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'square' (line 332)
                square_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'square')
                # Obtaining the member 'color' of a type (line 332)
                color_1041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), square_1040, 'color')
                # Getting the type of 'EMPTY' (line 332)
                EMPTY_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'EMPTY')
                # Applying the binary operator '==' (line 332)
                result_eq_1043 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 19), '==', color_1041, EMPTY_1042)
                
                # Testing if the type of an if condition is none (line 332)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 332, 16), result_eq_1043):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 332)
                    if_condition_1044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 16), result_eq_1043)
                    # Assigning a type to the variable 'if_condition_1044' (line 332)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'if_condition_1044', if_condition_1044)
                    # SSA begins for if statement (line 332)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to add(...): (line 333)
                    # Processing the call arguments (line 333)
                    # Getting the type of 'square' (line 333)
                    square_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 33), 'square', False)
                    # Obtaining the member 'pos' of a type (line 333)
                    pos_1048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 33), square_1047, 'pos')
                    # Processing the call keyword arguments (line 333)
                    kwargs_1049 = {}
                    # Getting the type of 'empties2' (line 333)
                    empties2_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'empties2', False)
                    # Obtaining the member 'add' of a type (line 333)
                    add_1046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 20), empties2_1045, 'add')
                    # Calling add(args, kwargs) (line 333)
                    add_call_result_1050 = invoke(stypy.reporting.localization.Localization(__file__, 333, 20), add_1046, *[pos_1048], **kwargs_1049)
                    
                    # SSA join for if statement (line 332)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Evaluating assert statement condition
            
            # Getting the type of 'empties1' (line 335)
            empties1_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'empties1')
            # Getting the type of 'empties2' (line 335)
            empties2_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'empties2')
            # Applying the binary operator '==' (line 335)
            result_eq_1053 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 19), '==', empties1_1051, empties2_1052)
            
            assert_1054 = result_eq_1053
            # Assigning a type to the variable 'assert_1054' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'assert_1054', result_eq_1053)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_1055


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 337, 4, False)
        # Assigning a type to the variable 'self' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Board.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Board.stypy__repr__')
        Board.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Board.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 338):
        
        # Assigning a List to a Name (line 338):
        
        # Obtaining an instance of the builtin type 'list' (line 338)
        list_1056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        
        # Assigning a type to the variable 'result' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'result', list_1056)
        
        
        # Call to range(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'SIZE' (line 339)
        SIZE_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'SIZE', False)
        # Processing the call keyword arguments (line 339)
        kwargs_1059 = {}
        # Getting the type of 'range' (line 339)
        range_1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'range', False)
        # Calling range(args, kwargs) (line 339)
        range_call_result_1060 = invoke(stypy.reporting.localization.Localization(__file__, 339, 17), range_1057, *[SIZE_1058], **kwargs_1059)
        
        # Assigning a type to the variable 'range_call_result_1060' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'range_call_result_1060', range_call_result_1060)
        # Testing if the for loop is going to be iterated (line 339)
        # Testing the type of a for loop iterable (line 339)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_1060)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_1060):
            # Getting the type of the for loop variable (line 339)
            for_loop_var_1061 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_1060)
            # Assigning a type to the variable 'y' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'y', for_loop_var_1061)
            # SSA begins for a for statement (line 339)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 340):
            
            # Assigning a Call to a Name (line 340):
            
            # Call to to_pos(...): (line 340)
            # Processing the call arguments (line 340)
            int_1063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 27), 'int')
            # Getting the type of 'y' (line 340)
            y_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 30), 'y', False)
            # Processing the call keyword arguments (line 340)
            kwargs_1065 = {}
            # Getting the type of 'to_pos' (line 340)
            to_pos_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'to_pos', False)
            # Calling to_pos(args, kwargs) (line 340)
            to_pos_call_result_1066 = invoke(stypy.reporting.localization.Localization(__file__, 340, 20), to_pos_1062, *[int_1063, y_1064], **kwargs_1065)
            
            # Assigning a type to the variable 'start' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'start', to_pos_call_result_1066)
            
            # Call to append(...): (line 341)
            # Processing the call arguments (line 341)
            
            # Call to join(...): (line 341)
            # Processing the call arguments (line 341)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Obtaining the type of the subscript
            # Getting the type of 'start' (line 341)
            start_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 87), 'start', False)
            # Getting the type of 'start' (line 341)
            start_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 93), 'start', False)
            # Getting the type of 'SIZE' (line 341)
            SIZE_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 101), 'SIZE', False)
            # Applying the binary operator '+' (line 341)
            result_add_1081 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 93), '+', start_1079, SIZE_1080)
            
            slice_1082 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 74), start_1078, result_add_1081, None)
            # Getting the type of 'self' (line 341)
            self_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 74), 'self', False)
            # Obtaining the member 'squares' of a type (line 341)
            squares_1084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 74), self_1083, 'squares')
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___1085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 74), squares_1084, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_1086 = invoke(stypy.reporting.localization.Localization(__file__, 341, 74), getitem___1085, slice_1082)
            
            comprehension_1087 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 35), subscript_call_result_1086)
            # Assigning a type to the variable 'square' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'square', comprehension_1087)
            
            # Obtaining the type of the subscript
            # Getting the type of 'square' (line 341)
            square_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 40), 'square', False)
            # Obtaining the member 'color' of a type (line 341)
            color_1072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 40), square_1071, 'color')
            # Getting the type of 'SHOW' (line 341)
            SHOW_1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'SHOW', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___1074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 35), SHOW_1073, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_1075 = invoke(stypy.reporting.localization.Localization(__file__, 341, 35), getitem___1074, color_1072)
            
            str_1076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 56), 'str', ' ')
            # Applying the binary operator '+' (line 341)
            result_add_1077 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 35), '+', subscript_call_result_1075, str_1076)
            
            list_1088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 35), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 35), list_1088, result_add_1077)
            # Processing the call keyword arguments (line 341)
            kwargs_1089 = {}
            str_1069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 26), 'str', '')
            # Obtaining the member 'join' of a type (line 341)
            join_1070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 26), str_1069, 'join')
            # Calling join(args, kwargs) (line 341)
            join_call_result_1090 = invoke(stypy.reporting.localization.Localization(__file__, 341, 26), join_1070, *[list_1088], **kwargs_1089)
            
            # Processing the call keyword arguments (line 341)
            kwargs_1091 = {}
            # Getting the type of 'result' (line 341)
            result_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'result', False)
            # Obtaining the member 'append' of a type (line 341)
            append_1068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), result_1067, 'append')
            # Calling append(args, kwargs) (line 341)
            append_call_result_1092 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), append_1068, *[join_call_result_1090], **kwargs_1091)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to join(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'result' (line 342)
        result_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'result', False)
        # Processing the call keyword arguments (line 342)
        kwargs_1096 = {}
        str_1093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 342)
        join_1094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), str_1093, 'join')
        # Calling join(args, kwargs) (line 342)
        join_call_result_1097 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), join_1094, *[result_1095], **kwargs_1096)
        
        # Assigning a type to the variable 'stypy_return_type' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'stypy_return_type', join_call_result_1097)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1098)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_1098


# Assigning a type to the variable 'Board' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'Board', Board)
# Declaration of the 'UCTNode' class

class UCTNode:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 346, 4, False)
        # Assigning a type to the variable 'self' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 347):
        
        # Assigning a Name to a Attribute (line 347):
        # Getting the type of 'None' (line 347)
        None_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 25), 'None')
        # Getting the type of 'self' (line 347)
        self_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self')
        # Setting the type of the member 'bestchild' of a type (line 347)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_1100, 'bestchild', None_1099)
        
        # Assigning a Num to a Attribute (line 348):
        
        # Assigning a Num to a Attribute (line 348):
        int_1101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 19), 'int')
        # Getting the type of 'self' (line 348)
        self_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 348)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_1102, 'pos', int_1101)
        
        # Assigning a Num to a Attribute (line 349):
        
        # Assigning a Num to a Attribute (line 349):
        int_1103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 20), 'int')
        # Getting the type of 'self' (line 349)
        self_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'self')
        # Setting the type of the member 'wins' of a type (line 349)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), self_1104, 'wins', int_1103)
        
        # Assigning a Num to a Attribute (line 350):
        
        # Assigning a Num to a Attribute (line 350):
        int_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 22), 'int')
        # Getting the type of 'self' (line 350)
        self_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self')
        # Setting the type of the member 'losses' of a type (line 350)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_1106, 'losses', int_1105)
        
        # Assigning a ListComp to a Attribute (line 351):
        
        # Assigning a ListComp to a Attribute (line 351):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'SIZE' (line 351)
        SIZE_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 46), 'SIZE', False)
        # Getting the type of 'SIZE' (line 351)
        SIZE_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 53), 'SIZE', False)
        # Applying the binary operator '*' (line 351)
        result_mul_1111 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 46), '*', SIZE_1109, SIZE_1110)
        
        # Processing the call keyword arguments (line 351)
        kwargs_1112 = {}
        # Getting the type of 'range' (line 351)
        range_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 40), 'range', False)
        # Calling range(args, kwargs) (line 351)
        range_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 351, 40), range_1108, *[result_mul_1111], **kwargs_1112)
        
        comprehension_1114 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 26), range_call_result_1113)
        # Assigning a type to the variable 'x' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'x', comprehension_1114)
        # Getting the type of 'None' (line 351)
        None_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'None')
        list_1115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 26), list_1115, None_1107)
        # Getting the type of 'self' (line 351)
        self_1116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self')
        # Setting the type of the member 'pos_child' of a type (line 351)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_1116, 'pos_child', list_1115)
        
        # Assigning a Num to a Attribute (line 352):
        
        # Assigning a Num to a Attribute (line 352):
        int_1117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 26), 'int')
        # Getting the type of 'self' (line 352)
        self_1118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self')
        # Setting the type of the member 'amafvisits' of a type (line 352)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), self_1118, 'amafvisits', int_1117)
        
        # Assigning a ListComp to a Attribute (line 353):
        
        # Assigning a ListComp to a Attribute (line 353):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'SIZE' (line 353)
        SIZE_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 47), 'SIZE', False)
        # Getting the type of 'SIZE' (line 353)
        SIZE_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 54), 'SIZE', False)
        # Applying the binary operator '*' (line 353)
        result_mul_1123 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 47), '*', SIZE_1121, SIZE_1122)
        
        # Processing the call keyword arguments (line 353)
        kwargs_1124 = {}
        # Getting the type of 'range' (line 353)
        range_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 41), 'range', False)
        # Calling range(args, kwargs) (line 353)
        range_call_result_1125 = invoke(stypy.reporting.localization.Localization(__file__, 353, 41), range_1120, *[result_mul_1123], **kwargs_1124)
        
        comprehension_1126 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 30), range_call_result_1125)
        # Assigning a type to the variable 'x' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'x', comprehension_1126)
        int_1119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 30), 'int')
        list_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 30), list_1127, int_1119)
        # Getting the type of 'self' (line 353)
        self_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self')
        # Setting the type of the member 'pos_amaf_wins' of a type (line 353)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), self_1128, 'pos_amaf_wins', list_1127)
        
        # Assigning a ListComp to a Attribute (line 354):
        
        # Assigning a ListComp to a Attribute (line 354):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'SIZE' (line 354)
        SIZE_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 49), 'SIZE', False)
        # Getting the type of 'SIZE' (line 354)
        SIZE_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 56), 'SIZE', False)
        # Applying the binary operator '*' (line 354)
        result_mul_1133 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 49), '*', SIZE_1131, SIZE_1132)
        
        # Processing the call keyword arguments (line 354)
        kwargs_1134 = {}
        # Getting the type of 'range' (line 354)
        range_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 43), 'range', False)
        # Calling range(args, kwargs) (line 354)
        range_call_result_1135 = invoke(stypy.reporting.localization.Localization(__file__, 354, 43), range_1130, *[result_mul_1133], **kwargs_1134)
        
        comprehension_1136 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 32), range_call_result_1135)
        # Assigning a type to the variable 'x' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 32), 'x', comprehension_1136)
        int_1129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 32), 'int')
        list_1137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 32), list_1137, int_1129)
        # Getting the type of 'self' (line 354)
        self_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'self')
        # Setting the type of the member 'pos_amaf_losses' of a type (line 354)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), self_1138, 'pos_amaf_losses', list_1137)
        
        # Assigning a Name to a Attribute (line 355):
        
        # Assigning a Name to a Attribute (line 355):
        # Getting the type of 'None' (line 355)
        None_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'None')
        # Getting the type of 'self' (line 355)
        self_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self')
        # Setting the type of the member 'parent' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_1140, 'parent', None_1139)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def play(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'play'
        module_type_store = module_type_store.open_function_context('play', 357, 4, False)
        # Assigning a type to the variable 'self' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UCTNode.play.__dict__.__setitem__('stypy_localization', localization)
        UCTNode.play.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UCTNode.play.__dict__.__setitem__('stypy_type_store', module_type_store)
        UCTNode.play.__dict__.__setitem__('stypy_function_name', 'UCTNode.play')
        UCTNode.play.__dict__.__setitem__('stypy_param_names_list', ['board'])
        UCTNode.play.__dict__.__setitem__('stypy_varargs_param_name', None)
        UCTNode.play.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UCTNode.play.__dict__.__setitem__('stypy_call_defaults', defaults)
        UCTNode.play.__dict__.__setitem__('stypy_call_varargs', varargs)
        UCTNode.play.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UCTNode.play.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.play', ['board'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'play', localization, ['board'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'play(...)' code ##################

        str_1141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 8), 'str', ' uct tree search ')
        
        # Assigning a Attribute to a Name (line 359):
        
        # Assigning a Attribute to a Name (line 359):
        # Getting the type of 'board' (line 359)
        board_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'board')
        # Obtaining the member 'color' of a type (line 359)
        color_1143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), board_1142, 'color')
        # Assigning a type to the variable 'color' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'color', color_1143)
        
        # Assigning a Name to a Name (line 360):
        
        # Assigning a Name to a Name (line 360):
        # Getting the type of 'self' (line 360)
        self_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'self')
        # Assigning a type to the variable 'node' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'node', self_1144)
        
        # Assigning a List to a Name (line 361):
        
        # Assigning a List to a Name (line 361):
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_1145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        # Getting the type of 'node' (line 361)
        node_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'node')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 15), list_1145, node_1146)
        
        # Assigning a type to the variable 'path' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'path', list_1145)
        
        # Assigning a Call to a Name (line 362):
        
        # Assigning a Call to a Name (line 362):
        
        # Call to len(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'board' (line 362)
        board_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'board', False)
        # Obtaining the member 'history' of a type (line 362)
        history_1149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 22), board_1148, 'history')
        # Processing the call keyword arguments (line 362)
        kwargs_1150 = {}
        # Getting the type of 'len' (line 362)
        len_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 18), 'len', False)
        # Calling len(args, kwargs) (line 362)
        len_call_result_1151 = invoke(stypy.reporting.localization.Localization(__file__, 362, 18), len_1147, *[history_1149], **kwargs_1150)
        
        # Assigning a type to the variable 'histpos' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'histpos', len_call_result_1151)
        
        # Getting the type of 'True' (line 363)
        True_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 14), 'True')
        # Assigning a type to the variable 'True_1152' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'True_1152', True_1152)
        # Testing if the while is going to be iterated (line 363)
        # Testing the type of an if condition (line 363)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 8), True_1152)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 363, 8), True_1152):
            
            # Assigning a Call to a Name (line 364):
            
            # Assigning a Call to a Name (line 364):
            
            # Call to select(...): (line 364)
            # Processing the call arguments (line 364)
            # Getting the type of 'board' (line 364)
            board_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'board', False)
            # Processing the call keyword arguments (line 364)
            kwargs_1156 = {}
            # Getting the type of 'node' (line 364)
            node_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 18), 'node', False)
            # Obtaining the member 'select' of a type (line 364)
            select_1154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 18), node_1153, 'select')
            # Calling select(args, kwargs) (line 364)
            select_call_result_1157 = invoke(stypy.reporting.localization.Localization(__file__, 364, 18), select_1154, *[board_1155], **kwargs_1156)
            
            # Assigning a type to the variable 'pos' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'pos', select_call_result_1157)
            
            # Getting the type of 'pos' (line 365)
            pos_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'pos')
            # Getting the type of 'PASS' (line 365)
            PASS_1159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'PASS')
            # Applying the binary operator '==' (line 365)
            result_eq_1160 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 15), '==', pos_1158, PASS_1159)
            
            # Testing if the type of an if condition is none (line 365)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 365, 12), result_eq_1160):
                pass
            else:
                
                # Testing the type of an if condition (line 365)
                if_condition_1161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 12), result_eq_1160)
                # Assigning a type to the variable 'if_condition_1161' (line 365)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'if_condition_1161', if_condition_1161)
                # SSA begins for if statement (line 365)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 365)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to move(...): (line 367)
            # Processing the call arguments (line 367)
            # Getting the type of 'pos' (line 367)
            pos_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'pos', False)
            # Processing the call keyword arguments (line 367)
            kwargs_1165 = {}
            # Getting the type of 'board' (line 367)
            board_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'board', False)
            # Obtaining the member 'move' of a type (line 367)
            move_1163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), board_1162, 'move')
            # Calling move(args, kwargs) (line 367)
            move_call_result_1166 = invoke(stypy.reporting.localization.Localization(__file__, 367, 12), move_1163, *[pos_1164], **kwargs_1165)
            
            
            # Assigning a Subscript to a Name (line 368):
            
            # Assigning a Subscript to a Name (line 368):
            
            # Obtaining the type of the subscript
            # Getting the type of 'pos' (line 368)
            pos_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'pos')
            # Getting the type of 'node' (line 368)
            node_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 20), 'node')
            # Obtaining the member 'pos_child' of a type (line 368)
            pos_child_1169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 20), node_1168, 'pos_child')
            # Obtaining the member '__getitem__' of a type (line 368)
            getitem___1170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 20), pos_child_1169, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 368)
            subscript_call_result_1171 = invoke(stypy.reporting.localization.Localization(__file__, 368, 20), getitem___1170, pos_1167)
            
            # Assigning a type to the variable 'child' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'child', subscript_call_result_1171)
            
            # Getting the type of 'child' (line 369)
            child_1172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'child')
            # Applying the 'not' unary operator (line 369)
            result_not__1173 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 15), 'not', child_1172)
            
            # Testing if the type of an if condition is none (line 369)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 369, 12), result_not__1173):
                pass
            else:
                
                # Testing the type of an if condition (line 369)
                if_condition_1174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 12), result_not__1173)
                # Assigning a type to the variable 'if_condition_1174' (line 369)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'if_condition_1174', if_condition_1174)
                # SSA begins for if statement (line 369)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Multiple assignment of 2 elements.
                
                # Assigning a Call to a Subscript (line 370):
                
                # Call to UCTNode(...): (line 370)
                # Processing the call keyword arguments (line 370)
                kwargs_1176 = {}
                # Getting the type of 'UCTNode' (line 370)
                UCTNode_1175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 46), 'UCTNode', False)
                # Calling UCTNode(args, kwargs) (line 370)
                UCTNode_call_result_1177 = invoke(stypy.reporting.localization.Localization(__file__, 370, 46), UCTNode_1175, *[], **kwargs_1176)
                
                # Getting the type of 'node' (line 370)
                node_1178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'node')
                # Obtaining the member 'pos_child' of a type (line 370)
                pos_child_1179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), node_1178, 'pos_child')
                # Getting the type of 'pos' (line 370)
                pos_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 39), 'pos')
                # Storing an element on a container (line 370)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 24), pos_child_1179, (pos_1180, UCTNode_call_result_1177))
                
                # Assigning a Subscript to a Name (line 370):
                
                # Obtaining the type of the subscript
                # Getting the type of 'pos' (line 370)
                pos_1181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 39), 'pos')
                # Getting the type of 'node' (line 370)
                node_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'node')
                # Obtaining the member 'pos_child' of a type (line 370)
                pos_child_1183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), node_1182, 'pos_child')
                # Obtaining the member '__getitem__' of a type (line 370)
                getitem___1184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), pos_child_1183, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 370)
                subscript_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 370, 24), getitem___1184, pos_1181)
                
                # Assigning a type to the variable 'child' (line 370)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'child', subscript_call_result_1185)
                
                # Assigning a Call to a Attribute (line 371):
                
                # Assigning a Call to a Attribute (line 371):
                
                # Call to useful_moves(...): (line 371)
                # Processing the call keyword arguments (line 371)
                kwargs_1188 = {}
                # Getting the type of 'board' (line 371)
                board_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 35), 'board', False)
                # Obtaining the member 'useful_moves' of a type (line 371)
                useful_moves_1187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 35), board_1186, 'useful_moves')
                # Calling useful_moves(args, kwargs) (line 371)
                useful_moves_call_result_1189 = invoke(stypy.reporting.localization.Localization(__file__, 371, 35), useful_moves_1187, *[], **kwargs_1188)
                
                # Getting the type of 'child' (line 371)
                child_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'child')
                # Setting the type of the member 'unexplored' of a type (line 371)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 16), child_1190, 'unexplored', useful_moves_call_result_1189)
                
                # Assigning a Name to a Attribute (line 372):
                
                # Assigning a Name to a Attribute (line 372):
                # Getting the type of 'pos' (line 372)
                pos_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 28), 'pos')
                # Getting the type of 'child' (line 372)
                child_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'child')
                # Setting the type of the member 'pos' of a type (line 372)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), child_1192, 'pos', pos_1191)
                
                # Assigning a Name to a Attribute (line 373):
                
                # Assigning a Name to a Attribute (line 373):
                # Getting the type of 'node' (line 373)
                node_1193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'node')
                # Getting the type of 'child' (line 373)
                child_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'child')
                # Setting the type of the member 'parent' of a type (line 373)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), child_1194, 'parent', node_1193)
                
                # Call to append(...): (line 374)
                # Processing the call arguments (line 374)
                # Getting the type of 'child' (line 374)
                child_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'child', False)
                # Processing the call keyword arguments (line 374)
                kwargs_1198 = {}
                # Getting the type of 'path' (line 374)
                path_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'path', False)
                # Obtaining the member 'append' of a type (line 374)
                append_1196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 16), path_1195, 'append')
                # Calling append(args, kwargs) (line 374)
                append_call_result_1199 = invoke(stypy.reporting.localization.Localization(__file__, 374, 16), append_1196, *[child_1197], **kwargs_1198)
                
                # SSA join for if statement (line 369)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to append(...): (line 376)
            # Processing the call arguments (line 376)
            # Getting the type of 'child' (line 376)
            child_1202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 24), 'child', False)
            # Processing the call keyword arguments (line 376)
            kwargs_1203 = {}
            # Getting the type of 'path' (line 376)
            path_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'path', False)
            # Obtaining the member 'append' of a type (line 376)
            append_1201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), path_1200, 'append')
            # Calling append(args, kwargs) (line 376)
            append_call_result_1204 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), append_1201, *[child_1202], **kwargs_1203)
            
            
            # Assigning a Name to a Name (line 377):
            
            # Assigning a Name to a Name (line 377):
            # Getting the type of 'child' (line 377)
            child_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'child')
            # Assigning a type to the variable 'node' (line 377)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'node', child_1205)

        
        
        # Call to random_playout(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'board' (line 378)
        board_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 28), 'board', False)
        # Processing the call keyword arguments (line 378)
        kwargs_1209 = {}
        # Getting the type of 'self' (line 378)
        self_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self', False)
        # Obtaining the member 'random_playout' of a type (line 378)
        random_playout_1207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_1206, 'random_playout')
        # Calling random_playout(args, kwargs) (line 378)
        random_playout_call_result_1210 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), random_playout_1207, *[board_1208], **kwargs_1209)
        
        
        # Call to update_path(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'board' (line 379)
        board_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'board', False)
        # Getting the type of 'histpos' (line 379)
        histpos_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 32), 'histpos', False)
        # Getting the type of 'color' (line 379)
        color_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 41), 'color', False)
        # Getting the type of 'path' (line 379)
        path_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 48), 'path', False)
        # Processing the call keyword arguments (line 379)
        kwargs_1217 = {}
        # Getting the type of 'self' (line 379)
        self_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'self', False)
        # Obtaining the member 'update_path' of a type (line 379)
        update_path_1212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), self_1211, 'update_path')
        # Calling update_path(args, kwargs) (line 379)
        update_path_call_result_1218 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), update_path_1212, *[board_1213, histpos_1214, color_1215, path_1216], **kwargs_1217)
        
        
        # ################# End of 'play(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'play' in the type store
        # Getting the type of 'stypy_return_type' (line 357)
        stypy_return_type_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'play'
        return stypy_return_type_1219


    @norecursion
    def select(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'select'
        module_type_store = module_type_store.open_function_context('select', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UCTNode.select.__dict__.__setitem__('stypy_localization', localization)
        UCTNode.select.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UCTNode.select.__dict__.__setitem__('stypy_type_store', module_type_store)
        UCTNode.select.__dict__.__setitem__('stypy_function_name', 'UCTNode.select')
        UCTNode.select.__dict__.__setitem__('stypy_param_names_list', ['board'])
        UCTNode.select.__dict__.__setitem__('stypy_varargs_param_name', None)
        UCTNode.select.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UCTNode.select.__dict__.__setitem__('stypy_call_defaults', defaults)
        UCTNode.select.__dict__.__setitem__('stypy_call_varargs', varargs)
        UCTNode.select.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UCTNode.select.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.select', ['board'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'select', localization, ['board'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'select(...)' code ##################

        str_1220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 8), 'str', ' select move; unexplored children first, then according to uct value ')
        # Getting the type of 'self' (line 383)
        self_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'self')
        # Obtaining the member 'unexplored' of a type (line 383)
        unexplored_1222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 11), self_1221, 'unexplored')
        # Testing if the type of an if condition is none (line 383)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 383, 8), unexplored_1222):
            # Getting the type of 'self' (line 389)
            self_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'self')
            # Obtaining the member 'bestchild' of a type (line 389)
            bestchild_1259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 13), self_1258, 'bestchild')
            # Testing if the type of an if condition is none (line 389)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 389, 13), bestchild_1259):
                # Getting the type of 'PASS' (line 392)
                PASS_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'PASS')
                # Assigning a type to the variable 'stypy_return_type' (line 392)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'stypy_return_type', PASS_1264)
            else:
                
                # Testing the type of an if condition (line 389)
                if_condition_1260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 13), bestchild_1259)
                # Assigning a type to the variable 'if_condition_1260' (line 389)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'if_condition_1260', if_condition_1260)
                # SSA begins for if statement (line 389)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 390)
                self_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'self')
                # Obtaining the member 'bestchild' of a type (line 390)
                bestchild_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), self_1261, 'bestchild')
                # Obtaining the member 'pos' of a type (line 390)
                pos_1263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), bestchild_1262, 'pos')
                # Assigning a type to the variable 'stypy_return_type' (line 390)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'stypy_return_type', pos_1263)
                # SSA branch for the else part of an if statement (line 389)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'PASS' (line 392)
                PASS_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'PASS')
                # Assigning a type to the variable 'stypy_return_type' (line 392)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'stypy_return_type', PASS_1264)
                # SSA join for if statement (line 389)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 383)
            if_condition_1223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), unexplored_1222)
            # Assigning a type to the variable 'if_condition_1223' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_1223', if_condition_1223)
            # SSA begins for if statement (line 383)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 384):
            
            # Assigning a Call to a Name (line 384):
            
            # Call to randrange(...): (line 384)
            # Processing the call arguments (line 384)
            
            # Call to len(...): (line 384)
            # Processing the call arguments (line 384)
            # Getting the type of 'self' (line 384)
            self_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 37), 'self', False)
            # Obtaining the member 'unexplored' of a type (line 384)
            unexplored_1228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 37), self_1227, 'unexplored')
            # Processing the call keyword arguments (line 384)
            kwargs_1229 = {}
            # Getting the type of 'len' (line 384)
            len_1226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 33), 'len', False)
            # Calling len(args, kwargs) (line 384)
            len_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 384, 33), len_1226, *[unexplored_1228], **kwargs_1229)
            
            # Processing the call keyword arguments (line 384)
            kwargs_1231 = {}
            # Getting the type of 'random' (line 384)
            random_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'random', False)
            # Obtaining the member 'randrange' of a type (line 384)
            randrange_1225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 16), random_1224, 'randrange')
            # Calling randrange(args, kwargs) (line 384)
            randrange_call_result_1232 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), randrange_1225, *[len_call_result_1230], **kwargs_1231)
            
            # Assigning a type to the variable 'i' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'i', randrange_call_result_1232)
            
            # Assigning a Subscript to a Name (line 385):
            
            # Assigning a Subscript to a Name (line 385):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 385)
            i_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 34), 'i')
            # Getting the type of 'self' (line 385)
            self_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'self')
            # Obtaining the member 'unexplored' of a type (line 385)
            unexplored_1235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), self_1234, 'unexplored')
            # Obtaining the member '__getitem__' of a type (line 385)
            getitem___1236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), unexplored_1235, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 385)
            subscript_call_result_1237 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), getitem___1236, i_1233)
            
            # Assigning a type to the variable 'pos' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'pos', subscript_call_result_1237)
            
            # Assigning a Subscript to a Subscript (line 386):
            
            # Assigning a Subscript to a Subscript (line 386):
            
            # Obtaining the type of the subscript
            
            # Call to len(...): (line 386)
            # Processing the call arguments (line 386)
            # Getting the type of 'self' (line 386)
            self_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 53), 'self', False)
            # Obtaining the member 'unexplored' of a type (line 386)
            unexplored_1240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 53), self_1239, 'unexplored')
            # Processing the call keyword arguments (line 386)
            kwargs_1241 = {}
            # Getting the type of 'len' (line 386)
            len_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), 'len', False)
            # Calling len(args, kwargs) (line 386)
            len_call_result_1242 = invoke(stypy.reporting.localization.Localization(__file__, 386, 49), len_1238, *[unexplored_1240], **kwargs_1241)
            
            int_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 72), 'int')
            # Applying the binary operator '-' (line 386)
            result_sub_1244 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 49), '-', len_call_result_1242, int_1243)
            
            # Getting the type of 'self' (line 386)
            self_1245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 33), 'self')
            # Obtaining the member 'unexplored' of a type (line 386)
            unexplored_1246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 33), self_1245, 'unexplored')
            # Obtaining the member '__getitem__' of a type (line 386)
            getitem___1247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 33), unexplored_1246, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 386)
            subscript_call_result_1248 = invoke(stypy.reporting.localization.Localization(__file__, 386, 33), getitem___1247, result_sub_1244)
            
            # Getting the type of 'self' (line 386)
            self_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'self')
            # Obtaining the member 'unexplored' of a type (line 386)
            unexplored_1250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), self_1249, 'unexplored')
            # Getting the type of 'i' (line 386)
            i_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 28), 'i')
            # Storing an element on a container (line 386)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), unexplored_1250, (i_1251, subscript_call_result_1248))
            
            # Call to pop(...): (line 387)
            # Processing the call keyword arguments (line 387)
            kwargs_1255 = {}
            # Getting the type of 'self' (line 387)
            self_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'self', False)
            # Obtaining the member 'unexplored' of a type (line 387)
            unexplored_1253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), self_1252, 'unexplored')
            # Obtaining the member 'pop' of a type (line 387)
            pop_1254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), unexplored_1253, 'pop')
            # Calling pop(args, kwargs) (line 387)
            pop_call_result_1256 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), pop_1254, *[], **kwargs_1255)
            
            # Getting the type of 'pos' (line 388)
            pos_1257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'pos')
            # Assigning a type to the variable 'stypy_return_type' (line 388)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'stypy_return_type', pos_1257)
            # SSA branch for the else part of an if statement (line 383)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'self' (line 389)
            self_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'self')
            # Obtaining the member 'bestchild' of a type (line 389)
            bestchild_1259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 13), self_1258, 'bestchild')
            # Testing if the type of an if condition is none (line 389)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 389, 13), bestchild_1259):
                # Getting the type of 'PASS' (line 392)
                PASS_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'PASS')
                # Assigning a type to the variable 'stypy_return_type' (line 392)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'stypy_return_type', PASS_1264)
            else:
                
                # Testing the type of an if condition (line 389)
                if_condition_1260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 13), bestchild_1259)
                # Assigning a type to the variable 'if_condition_1260' (line 389)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'if_condition_1260', if_condition_1260)
                # SSA begins for if statement (line 389)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 390)
                self_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'self')
                # Obtaining the member 'bestchild' of a type (line 390)
                bestchild_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), self_1261, 'bestchild')
                # Obtaining the member 'pos' of a type (line 390)
                pos_1263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), bestchild_1262, 'pos')
                # Assigning a type to the variable 'stypy_return_type' (line 390)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'stypy_return_type', pos_1263)
                # SSA branch for the else part of an if statement (line 389)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'PASS' (line 392)
                PASS_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'PASS')
                # Assigning a type to the variable 'stypy_return_type' (line 392)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'stypy_return_type', PASS_1264)
                # SSA join for if statement (line 389)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 383)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'select(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'select' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_1265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'select'
        return stypy_return_type_1265


    @norecursion
    def random_playout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'random_playout'
        module_type_store = module_type_store.open_function_context('random_playout', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UCTNode.random_playout.__dict__.__setitem__('stypy_localization', localization)
        UCTNode.random_playout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UCTNode.random_playout.__dict__.__setitem__('stypy_type_store', module_type_store)
        UCTNode.random_playout.__dict__.__setitem__('stypy_function_name', 'UCTNode.random_playout')
        UCTNode.random_playout.__dict__.__setitem__('stypy_param_names_list', ['board'])
        UCTNode.random_playout.__dict__.__setitem__('stypy_varargs_param_name', None)
        UCTNode.random_playout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UCTNode.random_playout.__dict__.__setitem__('stypy_call_defaults', defaults)
        UCTNode.random_playout.__dict__.__setitem__('stypy_call_varargs', varargs)
        UCTNode.random_playout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UCTNode.random_playout.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.random_playout', ['board'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'random_playout', localization, ['board'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'random_playout(...)' code ##################

        str_1266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'str', ' random play until both players pass ')
        
        
        # Call to range(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'MAXMOVES' (line 396)
        MAXMOVES_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'MAXMOVES', False)
        # Processing the call keyword arguments (line 396)
        kwargs_1269 = {}
        # Getting the type of 'range' (line 396)
        range_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'range', False)
        # Calling range(args, kwargs) (line 396)
        range_call_result_1270 = invoke(stypy.reporting.localization.Localization(__file__, 396, 17), range_1267, *[MAXMOVES_1268], **kwargs_1269)
        
        # Assigning a type to the variable 'range_call_result_1270' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'range_call_result_1270', range_call_result_1270)
        # Testing if the for loop is going to be iterated (line 396)
        # Testing the type of a for loop iterable (line 396)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 396, 8), range_call_result_1270)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 396, 8), range_call_result_1270):
            # Getting the type of the for loop variable (line 396)
            for_loop_var_1271 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 396, 8), range_call_result_1270)
            # Assigning a type to the variable 'x' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'x', for_loop_var_1271)
            # SSA begins for a for statement (line 396)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'board' (line 397)
            board_1272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'board')
            # Obtaining the member 'finished' of a type (line 397)
            finished_1273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 15), board_1272, 'finished')
            # Testing if the type of an if condition is none (line 397)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 397, 12), finished_1273):
                pass
            else:
                
                # Testing the type of an if condition (line 397)
                if_condition_1274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 12), finished_1273)
                # Assigning a type to the variable 'if_condition_1274' (line 397)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'if_condition_1274', if_condition_1274)
                # SSA begins for if statement (line 397)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 397)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Name (line 399):
            
            # Assigning a Name to a Name (line 399):
            # Getting the type of 'PASS' (line 399)
            PASS_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 18), 'PASS')
            # Assigning a type to the variable 'pos' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'pos', PASS_1275)
            # Getting the type of 'board' (line 400)
            board_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'board')
            # Obtaining the member 'atari' of a type (line 400)
            atari_1277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 15), board_1276, 'atari')
            # Testing if the type of an if condition is none (line 400)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 400, 12), atari_1277):
                pass
            else:
                
                # Testing the type of an if condition (line 400)
                if_condition_1278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 12), atari_1277)
                # Assigning a type to the variable 'if_condition_1278' (line 400)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'if_condition_1278', if_condition_1278)
                # SSA begins for if statement (line 400)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 401):
                
                # Assigning a Call to a Name (line 401):
                
                # Call to liberty(...): (line 401)
                # Processing the call keyword arguments (line 401)
                kwargs_1282 = {}
                # Getting the type of 'board' (line 401)
                board_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 26), 'board', False)
                # Obtaining the member 'atari' of a type (line 401)
                atari_1280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 26), board_1279, 'atari')
                # Obtaining the member 'liberty' of a type (line 401)
                liberty_1281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 26), atari_1280, 'liberty')
                # Calling liberty(args, kwargs) (line 401)
                liberty_call_result_1283 = invoke(stypy.reporting.localization.Localization(__file__, 401, 26), liberty_1281, *[], **kwargs_1282)
                
                # Assigning a type to the variable 'liberty' (line 401)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'liberty', liberty_call_result_1283)
                
                # Call to useful(...): (line 402)
                # Processing the call arguments (line 402)
                # Getting the type of 'liberty' (line 402)
                liberty_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 32), 'liberty', False)
                # Obtaining the member 'pos' of a type (line 402)
                pos_1287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 32), liberty_1286, 'pos')
                # Processing the call keyword arguments (line 402)
                kwargs_1288 = {}
                # Getting the type of 'board' (line 402)
                board_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 19), 'board', False)
                # Obtaining the member 'useful' of a type (line 402)
                useful_1285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 19), board_1284, 'useful')
                # Calling useful(args, kwargs) (line 402)
                useful_call_result_1289 = invoke(stypy.reporting.localization.Localization(__file__, 402, 19), useful_1285, *[pos_1287], **kwargs_1288)
                
                # Testing if the type of an if condition is none (line 402)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 402, 16), useful_call_result_1289):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 402)
                    if_condition_1290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 16), useful_call_result_1289)
                    # Assigning a type to the variable 'if_condition_1290' (line 402)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'if_condition_1290', if_condition_1290)
                    # SSA begins for if statement (line 402)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Attribute to a Name (line 403):
                    
                    # Assigning a Attribute to a Name (line 403):
                    # Getting the type of 'liberty' (line 403)
                    liberty_1291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 26), 'liberty')
                    # Obtaining the member 'pos' of a type (line 403)
                    pos_1292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 26), liberty_1291, 'pos')
                    # Assigning a type to the variable 'pos' (line 403)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'pos', pos_1292)
                    # SSA join for if statement (line 402)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 400)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'pos' (line 404)
            pos_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'pos')
            # Getting the type of 'PASS' (line 404)
            PASS_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 22), 'PASS')
            # Applying the binary operator '==' (line 404)
            result_eq_1295 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 15), '==', pos_1293, PASS_1294)
            
            # Testing if the type of an if condition is none (line 404)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 404, 12), result_eq_1295):
                pass
            else:
                
                # Testing the type of an if condition (line 404)
                if_condition_1296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 12), result_eq_1295)
                # Assigning a type to the variable 'if_condition_1296' (line 404)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'if_condition_1296', if_condition_1296)
                # SSA begins for if statement (line 404)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 405):
                
                # Assigning a Call to a Name (line 405):
                
                # Call to random_move(...): (line 405)
                # Processing the call keyword arguments (line 405)
                kwargs_1299 = {}
                # Getting the type of 'board' (line 405)
                board_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 22), 'board', False)
                # Obtaining the member 'random_move' of a type (line 405)
                random_move_1298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 22), board_1297, 'random_move')
                # Calling random_move(args, kwargs) (line 405)
                random_move_call_result_1300 = invoke(stypy.reporting.localization.Localization(__file__, 405, 22), random_move_1298, *[], **kwargs_1299)
                
                # Assigning a type to the variable 'pos' (line 405)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'pos', random_move_call_result_1300)
                # SSA join for if statement (line 404)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to move(...): (line 407)
            # Processing the call arguments (line 407)
            # Getting the type of 'pos' (line 407)
            pos_1303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 23), 'pos', False)
            # Processing the call keyword arguments (line 407)
            kwargs_1304 = {}
            # Getting the type of 'board' (line 407)
            board_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'board', False)
            # Obtaining the member 'move' of a type (line 407)
            move_1302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), board_1301, 'move')
            # Calling move(args, kwargs) (line 407)
            move_call_result_1305 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), move_1302, *[pos_1303], **kwargs_1304)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'random_playout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'random_playout' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'random_playout'
        return stypy_return_type_1306


    @norecursion
    def update_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_path'
        module_type_store = module_type_store.open_function_context('update_path', 414, 4, False)
        # Assigning a type to the variable 'self' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UCTNode.update_path.__dict__.__setitem__('stypy_localization', localization)
        UCTNode.update_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UCTNode.update_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        UCTNode.update_path.__dict__.__setitem__('stypy_function_name', 'UCTNode.update_path')
        UCTNode.update_path.__dict__.__setitem__('stypy_param_names_list', ['board', 'histpos', 'color', 'path'])
        UCTNode.update_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        UCTNode.update_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UCTNode.update_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        UCTNode.update_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        UCTNode.update_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UCTNode.update_path.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.update_path', ['board', 'histpos', 'color', 'path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_path', localization, ['board', 'histpos', 'color', 'path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_path(...)' code ##################

        str_1307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 8), 'str', ' update win/loss count along path ')
        
        # Assigning a Compare to a Name (line 416):
        
        # Assigning a Compare to a Name (line 416):
        
        
        # Call to score(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'BLACK' (line 416)
        BLACK_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 27), 'BLACK', False)
        # Processing the call keyword arguments (line 416)
        kwargs_1311 = {}
        # Getting the type of 'board' (line 416)
        board_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'board', False)
        # Obtaining the member 'score' of a type (line 416)
        score_1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), board_1308, 'score')
        # Calling score(args, kwargs) (line 416)
        score_call_result_1312 = invoke(stypy.reporting.localization.Localization(__file__, 416, 15), score_1309, *[BLACK_1310], **kwargs_1311)
        
        
        # Call to score(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'WHITE' (line 416)
        WHITE_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 49), 'WHITE', False)
        # Processing the call keyword arguments (line 416)
        kwargs_1316 = {}
        # Getting the type of 'board' (line 416)
        board_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 37), 'board', False)
        # Obtaining the member 'score' of a type (line 416)
        score_1314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 37), board_1313, 'score')
        # Calling score(args, kwargs) (line 416)
        score_call_result_1317 = invoke(stypy.reporting.localization.Localization(__file__, 416, 37), score_1314, *[WHITE_1315], **kwargs_1316)
        
        # Applying the binary operator '>=' (line 416)
        result_ge_1318 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 15), '>=', score_call_result_1312, score_call_result_1317)
        
        # Assigning a type to the variable 'wins' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'wins', result_ge_1318)
        
        # Getting the type of 'path' (line 417)
        path_1319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'path')
        # Assigning a type to the variable 'path_1319' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'path_1319', path_1319)
        # Testing if the for loop is going to be iterated (line 417)
        # Testing the type of a for loop iterable (line 417)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 417, 8), path_1319)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 417, 8), path_1319):
            # Getting the type of the for loop variable (line 417)
            for_loop_var_1320 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 417, 8), path_1319)
            # Assigning a type to the variable 'node' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'node', for_loop_var_1320)
            # SSA begins for a for statement (line 417)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'color' (line 418)
            color_1321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'color')
            # Getting the type of 'BLACK' (line 418)
            BLACK_1322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'BLACK')
            # Applying the binary operator '==' (line 418)
            result_eq_1323 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 15), '==', color_1321, BLACK_1322)
            
            # Testing if the type of an if condition is none (line 418)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 418, 12), result_eq_1323):
                
                # Assigning a Name to a Name (line 421):
                
                # Assigning a Name to a Name (line 421):
                # Getting the type of 'BLACK' (line 421)
                BLACK_1326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 24), 'BLACK')
                # Assigning a type to the variable 'color' (line 421)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'color', BLACK_1326)
            else:
                
                # Testing the type of an if condition (line 418)
                if_condition_1324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 12), result_eq_1323)
                # Assigning a type to the variable 'if_condition_1324' (line 418)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'if_condition_1324', if_condition_1324)
                # SSA begins for if statement (line 418)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 419):
                
                # Assigning a Name to a Name (line 419):
                # Getting the type of 'WHITE' (line 419)
                WHITE_1325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'WHITE')
                # Assigning a type to the variable 'color' (line 419)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'color', WHITE_1325)
                # SSA branch for the else part of an if statement (line 418)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 421):
                
                # Assigning a Name to a Name (line 421):
                # Getting the type of 'BLACK' (line 421)
                BLACK_1326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 24), 'BLACK')
                # Assigning a type to the variable 'color' (line 421)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'color', BLACK_1326)
                # SSA join for if statement (line 418)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'wins' (line 422)
            wins_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'wins')
            
            # Getting the type of 'color' (line 422)
            color_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'color')
            # Getting the type of 'BLACK' (line 422)
            BLACK_1329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 33), 'BLACK')
            # Applying the binary operator '==' (line 422)
            result_eq_1330 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 24), '==', color_1328, BLACK_1329)
            
            # Applying the binary operator '==' (line 422)
            result_eq_1331 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 15), '==', wins_1327, result_eq_1330)
            
            # Testing if the type of an if condition is none (line 422)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 422, 12), result_eq_1331):
                
                # Getting the type of 'node' (line 425)
                node_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'node')
                # Obtaining the member 'losses' of a type (line 425)
                losses_1339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), node_1338, 'losses')
                int_1340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 31), 'int')
                # Applying the binary operator '+=' (line 425)
                result_iadd_1341 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 16), '+=', losses_1339, int_1340)
                # Getting the type of 'node' (line 425)
                node_1342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'node')
                # Setting the type of the member 'losses' of a type (line 425)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), node_1342, 'losses', result_iadd_1341)
                
            else:
                
                # Testing the type of an if condition (line 422)
                if_condition_1332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 12), result_eq_1331)
                # Assigning a type to the variable 'if_condition_1332' (line 422)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'if_condition_1332', if_condition_1332)
                # SSA begins for if statement (line 422)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'node' (line 423)
                node_1333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'node')
                # Obtaining the member 'wins' of a type (line 423)
                wins_1334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), node_1333, 'wins')
                int_1335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 29), 'int')
                # Applying the binary operator '+=' (line 423)
                result_iadd_1336 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 16), '+=', wins_1334, int_1335)
                # Getting the type of 'node' (line 423)
                node_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'node')
                # Setting the type of the member 'wins' of a type (line 423)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), node_1337, 'wins', result_iadd_1336)
                
                # SSA branch for the else part of an if statement (line 422)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'node' (line 425)
                node_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'node')
                # Obtaining the member 'losses' of a type (line 425)
                losses_1339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), node_1338, 'losses')
                int_1340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 31), 'int')
                # Applying the binary operator '+=' (line 425)
                result_iadd_1341 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 16), '+=', losses_1339, int_1340)
                # Getting the type of 'node' (line 425)
                node_1342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'node')
                # Setting the type of the member 'losses' of a type (line 425)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), node_1342, 'losses', result_iadd_1341)
                
                # SSA join for if statement (line 422)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'node' (line 426)
            node_1343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'node')
            # Obtaining the member 'parent' of a type (line 426)
            parent_1344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 15), node_1343, 'parent')
            # Testing if the type of an if condition is none (line 426)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 426, 12), parent_1344):
                pass
            else:
                
                # Testing the type of an if condition (line 426)
                if_condition_1345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 12), parent_1344)
                # Assigning a type to the variable 'if_condition_1345' (line 426)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'if_condition_1345', if_condition_1345)
                # SSA begins for if statement (line 426)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to range(...): (line 427)
                # Processing the call arguments (line 427)
                # Getting the type of 'histpos' (line 427)
                histpos_1347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'histpos', False)
                int_1348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 41), 'int')
                # Applying the binary operator '+' (line 427)
                result_add_1349 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 31), '+', histpos_1347, int_1348)
                
                
                # Call to len(...): (line 427)
                # Processing the call arguments (line 427)
                # Getting the type of 'board' (line 427)
                board_1351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 48), 'board', False)
                # Obtaining the member 'history' of a type (line 427)
                history_1352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 48), board_1351, 'history')
                # Processing the call keyword arguments (line 427)
                kwargs_1353 = {}
                # Getting the type of 'len' (line 427)
                len_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 44), 'len', False)
                # Calling len(args, kwargs) (line 427)
                len_call_result_1354 = invoke(stypy.reporting.localization.Localization(__file__, 427, 44), len_1350, *[history_1352], **kwargs_1353)
                
                int_1355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 64), 'int')
                # Processing the call keyword arguments (line 427)
                kwargs_1356 = {}
                # Getting the type of 'range' (line 427)
                range_1346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 25), 'range', False)
                # Calling range(args, kwargs) (line 427)
                range_call_result_1357 = invoke(stypy.reporting.localization.Localization(__file__, 427, 25), range_1346, *[result_add_1349, len_call_result_1354, int_1355], **kwargs_1356)
                
                # Assigning a type to the variable 'range_call_result_1357' (line 427)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'range_call_result_1357', range_call_result_1357)
                # Testing if the for loop is going to be iterated (line 427)
                # Testing the type of a for loop iterable (line 427)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 427, 16), range_call_result_1357)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 427, 16), range_call_result_1357):
                    # Getting the type of the for loop variable (line 427)
                    for_loop_var_1358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 427, 16), range_call_result_1357)
                    # Assigning a type to the variable 'i' (line 427)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'i', for_loop_var_1358)
                    # SSA begins for a for statement (line 427)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Subscript to a Name (line 428):
                    
                    # Assigning a Subscript to a Name (line 428):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 428)
                    i_1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 40), 'i')
                    # Getting the type of 'board' (line 428)
                    board_1360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'board')
                    # Obtaining the member 'history' of a type (line 428)
                    history_1361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 26), board_1360, 'history')
                    # Obtaining the member '__getitem__' of a type (line 428)
                    getitem___1362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 26), history_1361, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
                    subscript_call_result_1363 = invoke(stypy.reporting.localization.Localization(__file__, 428, 26), getitem___1362, i_1359)
                    
                    # Assigning a type to the variable 'pos' (line 428)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'pos', subscript_call_result_1363)
                    
                    # Getting the type of 'pos' (line 429)
                    pos_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'pos')
                    # Getting the type of 'PASS' (line 429)
                    PASS_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'PASS')
                    # Applying the binary operator '==' (line 429)
                    result_eq_1366 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 23), '==', pos_1364, PASS_1365)
                    
                    # Testing if the type of an if condition is none (line 429)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 429, 20), result_eq_1366):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 429)
                        if_condition_1367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 429, 20), result_eq_1366)
                        # Assigning a type to the variable 'if_condition_1367' (line 429)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'if_condition_1367', if_condition_1367)
                        # SSA begins for if statement (line 429)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # SSA join for if statement (line 429)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'wins' (line 431)
                    wins_1368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'wins')
                    
                    # Getting the type of 'color' (line 431)
                    color_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 32), 'color')
                    # Getting the type of 'BLACK' (line 431)
                    BLACK_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 41), 'BLACK')
                    # Applying the binary operator '==' (line 431)
                    result_eq_1371 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 32), '==', color_1369, BLACK_1370)
                    
                    # Applying the binary operator '==' (line 431)
                    result_eq_1372 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 23), '==', wins_1368, result_eq_1371)
                    
                    # Testing if the type of an if condition is none (line 431)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 431, 20), result_eq_1372):
                        
                        # Getting the type of 'node' (line 434)
                        node_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 434)
                        parent_1390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1389, 'parent')
                        # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                        pos_amaf_losses_1391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1390, 'pos_amaf_losses')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'pos' (line 434)
                        pos_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'pos')
                        # Getting the type of 'node' (line 434)
                        node_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 434)
                        parent_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1393, 'parent')
                        # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                        pos_amaf_losses_1395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1394, 'pos_amaf_losses')
                        # Obtaining the member '__getitem__' of a type (line 434)
                        getitem___1396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), pos_amaf_losses_1395, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
                        subscript_call_result_1397 = invoke(stypy.reporting.localization.Localization(__file__, 434, 24), getitem___1396, pos_1392)
                        
                        int_1398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 60), 'int')
                        # Applying the binary operator '+=' (line 434)
                        result_iadd_1399 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 24), '+=', subscript_call_result_1397, int_1398)
                        # Getting the type of 'node' (line 434)
                        node_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 434)
                        parent_1401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1400, 'parent')
                        # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                        pos_amaf_losses_1402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1401, 'pos_amaf_losses')
                        # Getting the type of 'pos' (line 434)
                        pos_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'pos')
                        # Storing an element on a container (line 434)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 24), pos_amaf_losses_1402, (pos_1403, result_iadd_1399))
                        
                    else:
                        
                        # Testing the type of an if condition (line 431)
                        if_condition_1373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 20), result_eq_1372)
                        # Assigning a type to the variable 'if_condition_1373' (line 431)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 20), 'if_condition_1373', if_condition_1373)
                        # SSA begins for if statement (line 431)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'node' (line 432)
                        node_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 432)
                        parent_1375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), node_1374, 'parent')
                        # Obtaining the member 'pos_amaf_wins' of a type (line 432)
                        pos_amaf_wins_1376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), parent_1375, 'pos_amaf_wins')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'pos' (line 432)
                        pos_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'pos')
                        # Getting the type of 'node' (line 432)
                        node_1378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 432)
                        parent_1379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), node_1378, 'parent')
                        # Obtaining the member 'pos_amaf_wins' of a type (line 432)
                        pos_amaf_wins_1380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), parent_1379, 'pos_amaf_wins')
                        # Obtaining the member '__getitem__' of a type (line 432)
                        getitem___1381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), pos_amaf_wins_1380, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 432)
                        subscript_call_result_1382 = invoke(stypy.reporting.localization.Localization(__file__, 432, 24), getitem___1381, pos_1377)
                        
                        int_1383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 58), 'int')
                        # Applying the binary operator '+=' (line 432)
                        result_iadd_1384 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 24), '+=', subscript_call_result_1382, int_1383)
                        # Getting the type of 'node' (line 432)
                        node_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 432)
                        parent_1386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), node_1385, 'parent')
                        # Obtaining the member 'pos_amaf_wins' of a type (line 432)
                        pos_amaf_wins_1387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), parent_1386, 'pos_amaf_wins')
                        # Getting the type of 'pos' (line 432)
                        pos_1388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'pos')
                        # Storing an element on a container (line 432)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 24), pos_amaf_wins_1387, (pos_1388, result_iadd_1384))
                        
                        # SSA branch for the else part of an if statement (line 431)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'node' (line 434)
                        node_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 434)
                        parent_1390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1389, 'parent')
                        # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                        pos_amaf_losses_1391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1390, 'pos_amaf_losses')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'pos' (line 434)
                        pos_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'pos')
                        # Getting the type of 'node' (line 434)
                        node_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 434)
                        parent_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1393, 'parent')
                        # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                        pos_amaf_losses_1395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1394, 'pos_amaf_losses')
                        # Obtaining the member '__getitem__' of a type (line 434)
                        getitem___1396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), pos_amaf_losses_1395, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
                        subscript_call_result_1397 = invoke(stypy.reporting.localization.Localization(__file__, 434, 24), getitem___1396, pos_1392)
                        
                        int_1398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 60), 'int')
                        # Applying the binary operator '+=' (line 434)
                        result_iadd_1399 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 24), '+=', subscript_call_result_1397, int_1398)
                        # Getting the type of 'node' (line 434)
                        node_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                        # Obtaining the member 'parent' of a type (line 434)
                        parent_1401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1400, 'parent')
                        # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                        pos_amaf_losses_1402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1401, 'pos_amaf_losses')
                        # Getting the type of 'pos' (line 434)
                        pos_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'pos')
                        # Storing an element on a container (line 434)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 24), pos_amaf_losses_1402, (pos_1403, result_iadd_1399))
                        
                        # SSA join for if statement (line 431)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'node' (line 435)
                    node_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'node')
                    # Obtaining the member 'parent' of a type (line 435)
                    parent_1405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), node_1404, 'parent')
                    # Obtaining the member 'amafvisits' of a type (line 435)
                    amafvisits_1406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), parent_1405, 'amafvisits')
                    int_1407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 46), 'int')
                    # Applying the binary operator '+=' (line 435)
                    result_iadd_1408 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 20), '+=', amafvisits_1406, int_1407)
                    # Getting the type of 'node' (line 435)
                    node_1409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'node')
                    # Obtaining the member 'parent' of a type (line 435)
                    parent_1410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), node_1409, 'parent')
                    # Setting the type of the member 'amafvisits' of a type (line 435)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), parent_1410, 'amafvisits', result_iadd_1408)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Attribute (line 436):
                
                # Assigning a Call to a Attribute (line 436):
                
                # Call to best_child(...): (line 436)
                # Processing the call keyword arguments (line 436)
                kwargs_1414 = {}
                # Getting the type of 'node' (line 436)
                node_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 40), 'node', False)
                # Obtaining the member 'parent' of a type (line 436)
                parent_1412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 40), node_1411, 'parent')
                # Obtaining the member 'best_child' of a type (line 436)
                best_child_1413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 40), parent_1412, 'best_child')
                # Calling best_child(args, kwargs) (line 436)
                best_child_call_result_1415 = invoke(stypy.reporting.localization.Localization(__file__, 436, 40), best_child_1413, *[], **kwargs_1414)
                
                # Getting the type of 'node' (line 436)
                node_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'node')
                # Obtaining the member 'parent' of a type (line 436)
                parent_1417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), node_1416, 'parent')
                # Setting the type of the member 'bestchild' of a type (line 436)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), parent_1417, 'bestchild', best_child_call_result_1415)
                # SSA join for if statement (line 426)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'update_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_path' in the type store
        # Getting the type of 'stypy_return_type' (line 414)
        stypy_return_type_1418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_path'
        return stypy_return_type_1418


    @norecursion
    def score(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score'
        module_type_store = module_type_store.open_function_context('score', 438, 4, False)
        # Assigning a type to the variable 'self' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UCTNode.score.__dict__.__setitem__('stypy_localization', localization)
        UCTNode.score.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UCTNode.score.__dict__.__setitem__('stypy_type_store', module_type_store)
        UCTNode.score.__dict__.__setitem__('stypy_function_name', 'UCTNode.score')
        UCTNode.score.__dict__.__setitem__('stypy_param_names_list', [])
        UCTNode.score.__dict__.__setitem__('stypy_varargs_param_name', None)
        UCTNode.score.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UCTNode.score.__dict__.__setitem__('stypy_call_defaults', defaults)
        UCTNode.score.__dict__.__setitem__('stypy_call_varargs', varargs)
        UCTNode.score.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UCTNode.score.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.score', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score(...)' code ##################

        
        # Assigning a BinOp to a Name (line 439):
        
        # Assigning a BinOp to a Name (line 439):
        # Getting the type of 'self' (line 439)
        self_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'self')
        # Obtaining the member 'wins' of a type (line 439)
        wins_1420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 18), self_1419, 'wins')
        
        # Call to float(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'self' (line 439)
        self_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 36), 'self', False)
        # Obtaining the member 'wins' of a type (line 439)
        wins_1423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 36), self_1422, 'wins')
        # Getting the type of 'self' (line 439)
        self_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 48), 'self', False)
        # Obtaining the member 'losses' of a type (line 439)
        losses_1425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 48), self_1424, 'losses')
        # Applying the binary operator '+' (line 439)
        result_add_1426 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 36), '+', wins_1423, losses_1425)
        
        # Processing the call keyword arguments (line 439)
        kwargs_1427 = {}
        # Getting the type of 'float' (line 439)
        float_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 30), 'float', False)
        # Calling float(args, kwargs) (line 439)
        float_call_result_1428 = invoke(stypy.reporting.localization.Localization(__file__, 439, 30), float_1421, *[result_add_1426], **kwargs_1427)
        
        # Applying the binary operator 'div' (line 439)
        result_div_1429 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 18), 'div', wins_1420, float_call_result_1428)
        
        # Assigning a type to the variable 'winrate' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'winrate', result_div_1429)
        
        # Assigning a BinOp to a Name (line 440):
        
        # Assigning a BinOp to a Name (line 440):
        # Getting the type of 'self' (line 440)
        self_1430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'self')
        # Obtaining the member 'parent' of a type (line 440)
        parent_1431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), self_1430, 'parent')
        # Obtaining the member 'wins' of a type (line 440)
        wins_1432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), parent_1431, 'wins')
        # Getting the type of 'self' (line 440)
        self_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 42), 'self')
        # Obtaining the member 'parent' of a type (line 440)
        parent_1434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 42), self_1433, 'parent')
        # Obtaining the member 'losses' of a type (line 440)
        losses_1435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 42), parent_1434, 'losses')
        # Applying the binary operator '+' (line 440)
        result_add_1436 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 23), '+', wins_1432, losses_1435)
        
        # Assigning a type to the variable 'parentvisits' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'parentvisits', result_add_1436)
        
        # Getting the type of 'parentvisits' (line 441)
        parentvisits_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 15), 'parentvisits')
        # Applying the 'not' unary operator (line 441)
        result_not__1438 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 11), 'not', parentvisits_1437)
        
        # Testing if the type of an if condition is none (line 441)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 441, 8), result_not__1438):
            pass
        else:
            
            # Testing the type of an if condition (line 441)
            if_condition_1439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 8), result_not__1438)
            # Assigning a type to the variable 'if_condition_1439' (line 441)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'if_condition_1439', if_condition_1439)
            # SSA begins for if statement (line 441)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'winrate' (line 442)
            winrate_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 'winrate')
            # Assigning a type to the variable 'stypy_return_type' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'stypy_return_type', winrate_1440)
            # SSA join for if statement (line 441)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 443):
        
        # Assigning a BinOp to a Name (line 443):
        # Getting the type of 'self' (line 443)
        self_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 21), 'self')
        # Obtaining the member 'wins' of a type (line 443)
        wins_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 21), self_1441, 'wins')
        # Getting the type of 'self' (line 443)
        self_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 33), 'self')
        # Obtaining the member 'losses' of a type (line 443)
        losses_1444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 33), self_1443, 'losses')
        # Applying the binary operator '+' (line 443)
        result_add_1445 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 21), '+', wins_1442, losses_1444)
        
        # Assigning a type to the variable 'nodevisits' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'nodevisits', result_add_1445)
        
        # Assigning a BinOp to a Name (line 444):
        
        # Assigning a BinOp to a Name (line 444):
        # Getting the type of 'winrate' (line 444)
        winrate_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'winrate')
        
        # Call to sqrt(...): (line 444)
        # Processing the call arguments (line 444)
        
        # Call to log(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'parentvisits' (line 444)
        parentvisits_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 50), 'parentvisits', False)
        # Processing the call keyword arguments (line 444)
        kwargs_1452 = {}
        # Getting the type of 'math' (line 444)
        math_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 41), 'math', False)
        # Obtaining the member 'log' of a type (line 444)
        log_1450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 41), math_1449, 'log')
        # Calling log(args, kwargs) (line 444)
        log_call_result_1453 = invoke(stypy.reporting.localization.Localization(__file__, 444, 41), log_1450, *[parentvisits_1451], **kwargs_1452)
        
        int_1454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 68), 'int')
        # Getting the type of 'nodevisits' (line 444)
        nodevisits_1455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 72), 'nodevisits', False)
        # Applying the binary operator '*' (line 444)
        result_mul_1456 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 68), '*', int_1454, nodevisits_1455)
        
        # Applying the binary operator 'div' (line 444)
        result_div_1457 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 40), 'div', log_call_result_1453, result_mul_1456)
        
        # Processing the call keyword arguments (line 444)
        kwargs_1458 = {}
        # Getting the type of 'math' (line 444)
        math_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 30), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 444)
        sqrt_1448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 30), math_1447, 'sqrt')
        # Calling sqrt(args, kwargs) (line 444)
        sqrt_call_result_1459 = invoke(stypy.reporting.localization.Localization(__file__, 444, 30), sqrt_1448, *[result_div_1457], **kwargs_1458)
        
        # Applying the binary operator '+' (line 444)
        result_add_1460 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), '+', winrate_1446, sqrt_call_result_1459)
        
        # Assigning a type to the variable 'uct_score' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'uct_score', result_add_1460)
        
        # Assigning a BinOp to a Name (line 446):
        
        # Assigning a BinOp to a Name (line 446):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 446)
        self_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 47), 'self')
        # Obtaining the member 'pos' of a type (line 446)
        pos_1462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 47), self_1461, 'pos')
        # Getting the type of 'self' (line 446)
        self_1463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 21), 'self')
        # Obtaining the member 'parent' of a type (line 446)
        parent_1464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), self_1463, 'parent')
        # Obtaining the member 'pos_amaf_wins' of a type (line 446)
        pos_amaf_wins_1465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), parent_1464, 'pos_amaf_wins')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___1466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), pos_amaf_wins_1465, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_1467 = invoke(stypy.reporting.localization.Localization(__file__, 446, 21), getitem___1466, pos_1462)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 446)
        self_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 87), 'self')
        # Obtaining the member 'pos' of a type (line 446)
        pos_1469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 87), self_1468, 'pos')
        # Getting the type of 'self' (line 446)
        self_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 59), 'self')
        # Obtaining the member 'parent' of a type (line 446)
        parent_1471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 59), self_1470, 'parent')
        # Obtaining the member 'pos_amaf_losses' of a type (line 446)
        pos_amaf_losses_1472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 59), parent_1471, 'pos_amaf_losses')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___1473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 59), pos_amaf_losses_1472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_1474 = invoke(stypy.reporting.localization.Localization(__file__, 446, 59), getitem___1473, pos_1469)
        
        # Applying the binary operator '+' (line 446)
        result_add_1475 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 21), '+', subscript_call_result_1467, subscript_call_result_1474)
        
        # Assigning a type to the variable 'amafvisits' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'amafvisits', result_add_1475)
        
        # Getting the type of 'amafvisits' (line 447)
        amafvisits_1476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 15), 'amafvisits')
        # Applying the 'not' unary operator (line 447)
        result_not__1477 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 11), 'not', amafvisits_1476)
        
        # Testing if the type of an if condition is none (line 447)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 447, 8), result_not__1477):
            pass
        else:
            
            # Testing the type of an if condition (line 447)
            if_condition_1478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 8), result_not__1477)
            # Assigning a type to the variable 'if_condition_1478' (line 447)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'if_condition_1478', if_condition_1478)
            # SSA begins for if statement (line 447)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'uct_score' (line 448)
            uct_score_1479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 19), 'uct_score')
            # Assigning a type to the variable 'stypy_return_type' (line 448)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'stypy_return_type', uct_score_1479)
            # SSA join for if statement (line 447)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 449):
        
        # Assigning a BinOp to a Name (line 449):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 449)
        self_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 48), 'self')
        # Obtaining the member 'pos' of a type (line 449)
        pos_1481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 48), self_1480, 'pos')
        # Getting the type of 'self' (line 449)
        self_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 22), 'self')
        # Obtaining the member 'parent' of a type (line 449)
        parent_1483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), self_1482, 'parent')
        # Obtaining the member 'pos_amaf_wins' of a type (line 449)
        pos_amaf_wins_1484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), parent_1483, 'pos_amaf_wins')
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___1485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), pos_amaf_wins_1484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_1486 = invoke(stypy.reporting.localization.Localization(__file__, 449, 22), getitem___1485, pos_1481)
        
        
        # Call to float(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'amafvisits' (line 449)
        amafvisits_1488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 66), 'amafvisits', False)
        # Processing the call keyword arguments (line 449)
        kwargs_1489 = {}
        # Getting the type of 'float' (line 449)
        float_1487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 60), 'float', False)
        # Calling float(args, kwargs) (line 449)
        float_call_result_1490 = invoke(stypy.reporting.localization.Localization(__file__, 449, 60), float_1487, *[amafvisits_1488], **kwargs_1489)
        
        # Applying the binary operator 'div' (line 449)
        result_div_1491 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 22), 'div', subscript_call_result_1486, float_call_result_1490)
        
        # Assigning a type to the variable 'amafwinrate' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'amafwinrate', result_div_1491)
        
        # Assigning a BinOp to a Name (line 450):
        
        # Assigning a BinOp to a Name (line 450):
        # Getting the type of 'amafwinrate' (line 450)
        amafwinrate_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'amafwinrate')
        
        # Call to sqrt(...): (line 450)
        # Processing the call arguments (line 450)
        
        # Call to log(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'self' (line 450)
        self_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 53), 'self', False)
        # Obtaining the member 'parent' of a type (line 450)
        parent_1498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 53), self_1497, 'parent')
        # Obtaining the member 'amafvisits' of a type (line 450)
        amafvisits_1499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 53), parent_1498, 'amafvisits')
        # Processing the call keyword arguments (line 450)
        kwargs_1500 = {}
        # Getting the type of 'math' (line 450)
        math_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 44), 'math', False)
        # Obtaining the member 'log' of a type (line 450)
        log_1496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 44), math_1495, 'log')
        # Calling log(args, kwargs) (line 450)
        log_call_result_1501 = invoke(stypy.reporting.localization.Localization(__file__, 450, 44), log_1496, *[amafvisits_1499], **kwargs_1500)
        
        int_1502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 81), 'int')
        # Getting the type of 'amafvisits' (line 450)
        amafvisits_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 85), 'amafvisits', False)
        # Applying the binary operator '*' (line 450)
        result_mul_1504 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 81), '*', int_1502, amafvisits_1503)
        
        # Applying the binary operator 'div' (line 450)
        result_div_1505 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 43), 'div', log_call_result_1501, result_mul_1504)
        
        # Processing the call keyword arguments (line 450)
        kwargs_1506 = {}
        # Getting the type of 'math' (line 450)
        math_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 33), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 450)
        sqrt_1494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 33), math_1493, 'sqrt')
        # Calling sqrt(args, kwargs) (line 450)
        sqrt_call_result_1507 = invoke(stypy.reporting.localization.Localization(__file__, 450, 33), sqrt_1494, *[result_div_1505], **kwargs_1506)
        
        # Applying the binary operator '+' (line 450)
        result_add_1508 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 19), '+', amafwinrate_1492, sqrt_call_result_1507)
        
        # Assigning a type to the variable 'uct_amaf' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'uct_amaf', result_add_1508)
        
        # Assigning a Call to a Name (line 452):
        
        # Assigning a Call to a Name (line 452):
        
        # Call to sqrt(...): (line 452)
        # Processing the call arguments (line 452)
        float_1511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 25), 'float')
        int_1512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 35), 'int')
        # Getting the type of 'parentvisits' (line 452)
        parentvisits_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 39), 'parentvisits', False)
        # Applying the binary operator '*' (line 452)
        result_mul_1514 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 35), '*', int_1512, parentvisits_1513)
        
        float_1515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 54), 'float')
        # Applying the binary operator '+' (line 452)
        result_add_1516 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 35), '+', result_mul_1514, float_1515)
        
        # Applying the binary operator 'div' (line 452)
        result_div_1517 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 25), 'div', float_1511, result_add_1516)
        
        # Processing the call keyword arguments (line 452)
        kwargs_1518 = {}
        # Getting the type of 'math' (line 452)
        math_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 452)
        sqrt_1510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 15), math_1509, 'sqrt')
        # Calling sqrt(args, kwargs) (line 452)
        sqrt_call_result_1519 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), sqrt_1510, *[result_div_1517], **kwargs_1518)
        
        # Assigning a type to the variable 'beta' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'beta', sqrt_call_result_1519)
        # Getting the type of 'beta' (line 453)
        beta_1520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'beta')
        # Getting the type of 'uct_amaf' (line 453)
        uct_amaf_1521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 22), 'uct_amaf')
        # Applying the binary operator '*' (line 453)
        result_mul_1522 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 15), '*', beta_1520, uct_amaf_1521)
        
        int_1523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 34), 'int')
        # Getting the type of 'beta' (line 453)
        beta_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 38), 'beta')
        # Applying the binary operator '-' (line 453)
        result_sub_1525 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 34), '-', int_1523, beta_1524)
        
        # Getting the type of 'uct_score' (line 453)
        uct_score_1526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 46), 'uct_score')
        # Applying the binary operator '*' (line 453)
        result_mul_1527 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 33), '*', result_sub_1525, uct_score_1526)
        
        # Applying the binary operator '+' (line 453)
        result_add_1528 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 15), '+', result_mul_1522, result_mul_1527)
        
        # Assigning a type to the variable 'stypy_return_type' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'stypy_return_type', result_add_1528)
        
        # ################# End of 'score(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score' in the type store
        # Getting the type of 'stypy_return_type' (line 438)
        stypy_return_type_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1529)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score'
        return stypy_return_type_1529


    @norecursion
    def best_child(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'best_child'
        module_type_store = module_type_store.open_function_context('best_child', 455, 4, False)
        # Assigning a type to the variable 'self' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UCTNode.best_child.__dict__.__setitem__('stypy_localization', localization)
        UCTNode.best_child.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UCTNode.best_child.__dict__.__setitem__('stypy_type_store', module_type_store)
        UCTNode.best_child.__dict__.__setitem__('stypy_function_name', 'UCTNode.best_child')
        UCTNode.best_child.__dict__.__setitem__('stypy_param_names_list', [])
        UCTNode.best_child.__dict__.__setitem__('stypy_varargs_param_name', None)
        UCTNode.best_child.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UCTNode.best_child.__dict__.__setitem__('stypy_call_defaults', defaults)
        UCTNode.best_child.__dict__.__setitem__('stypy_call_varargs', varargs)
        UCTNode.best_child.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UCTNode.best_child.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.best_child', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'best_child', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'best_child(...)' code ##################

        
        # Assigning a Num to a Name (line 456):
        
        # Assigning a Num to a Name (line 456):
        int_1530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 19), 'int')
        # Assigning a type to the variable 'maxscore' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'maxscore', int_1530)
        
        # Assigning a Name to a Name (line 457):
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'None' (line 457)
        None_1531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'None')
        # Assigning a type to the variable 'maxchild' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'maxchild', None_1531)
        
        # Getting the type of 'self' (line 458)
        self_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 21), 'self')
        # Obtaining the member 'pos_child' of a type (line 458)
        pos_child_1533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 21), self_1532, 'pos_child')
        # Assigning a type to the variable 'pos_child_1533' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'pos_child_1533', pos_child_1533)
        # Testing if the for loop is going to be iterated (line 458)
        # Testing the type of a for loop iterable (line 458)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 458, 8), pos_child_1533)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 458, 8), pos_child_1533):
            # Getting the type of the for loop variable (line 458)
            for_loop_var_1534 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 458, 8), pos_child_1533)
            # Assigning a type to the variable 'child' (line 458)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'child', for_loop_var_1534)
            # SSA begins for a for statement (line 458)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            # Getting the type of 'child' (line 459)
            child_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'child')
            
            
            # Call to score(...): (line 459)
            # Processing the call keyword arguments (line 459)
            kwargs_1538 = {}
            # Getting the type of 'child' (line 459)
            child_1536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'child', False)
            # Obtaining the member 'score' of a type (line 459)
            score_1537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 25), child_1536, 'score')
            # Calling score(args, kwargs) (line 459)
            score_call_result_1539 = invoke(stypy.reporting.localization.Localization(__file__, 459, 25), score_1537, *[], **kwargs_1538)
            
            # Getting the type of 'maxscore' (line 459)
            maxscore_1540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 41), 'maxscore')
            # Applying the binary operator '>' (line 459)
            result_gt_1541 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 25), '>', score_call_result_1539, maxscore_1540)
            
            # Applying the binary operator 'and' (line 459)
            result_and_keyword_1542 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 15), 'and', child_1535, result_gt_1541)
            
            # Testing if the type of an if condition is none (line 459)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 459, 12), result_and_keyword_1542):
                pass
            else:
                
                # Testing the type of an if condition (line 459)
                if_condition_1543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 12), result_and_keyword_1542)
                # Assigning a type to the variable 'if_condition_1543' (line 459)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'if_condition_1543', if_condition_1543)
                # SSA begins for if statement (line 459)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 460):
                
                # Assigning a Name to a Name (line 460):
                # Getting the type of 'child' (line 460)
                child_1544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 27), 'child')
                # Assigning a type to the variable 'maxchild' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'maxchild', child_1544)
                
                # Assigning a Call to a Name (line 461):
                
                # Assigning a Call to a Name (line 461):
                
                # Call to score(...): (line 461)
                # Processing the call keyword arguments (line 461)
                kwargs_1547 = {}
                # Getting the type of 'child' (line 461)
                child_1545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 27), 'child', False)
                # Obtaining the member 'score' of a type (line 461)
                score_1546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 27), child_1545, 'score')
                # Calling score(args, kwargs) (line 461)
                score_call_result_1548 = invoke(stypy.reporting.localization.Localization(__file__, 461, 27), score_1546, *[], **kwargs_1547)
                
                # Assigning a type to the variable 'maxscore' (line 461)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'maxscore', score_call_result_1548)
                # SSA join for if statement (line 459)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'maxchild' (line 462)
        maxchild_1549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'maxchild')
        # Assigning a type to the variable 'stypy_return_type' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'stypy_return_type', maxchild_1549)
        
        # ################# End of 'best_child(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'best_child' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1550)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'best_child'
        return stypy_return_type_1550


    @norecursion
    def best_visited(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'best_visited'
        module_type_store = module_type_store.open_function_context('best_visited', 464, 4, False)
        # Assigning a type to the variable 'self' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UCTNode.best_visited.__dict__.__setitem__('stypy_localization', localization)
        UCTNode.best_visited.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UCTNode.best_visited.__dict__.__setitem__('stypy_type_store', module_type_store)
        UCTNode.best_visited.__dict__.__setitem__('stypy_function_name', 'UCTNode.best_visited')
        UCTNode.best_visited.__dict__.__setitem__('stypy_param_names_list', [])
        UCTNode.best_visited.__dict__.__setitem__('stypy_varargs_param_name', None)
        UCTNode.best_visited.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UCTNode.best_visited.__dict__.__setitem__('stypy_call_defaults', defaults)
        UCTNode.best_visited.__dict__.__setitem__('stypy_call_varargs', varargs)
        UCTNode.best_visited.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UCTNode.best_visited.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UCTNode.best_visited', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'best_visited', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'best_visited(...)' code ##################

        
        # Assigning a Num to a Name (line 465):
        
        # Assigning a Num to a Name (line 465):
        int_1551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 20), 'int')
        # Assigning a type to the variable 'maxvisits' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'maxvisits', int_1551)
        
        # Assigning a Name to a Name (line 466):
        
        # Assigning a Name to a Name (line 466):
        # Getting the type of 'None' (line 466)
        None_1552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 19), 'None')
        # Assigning a type to the variable 'maxchild' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'maxchild', None_1552)
        
        # Getting the type of 'self' (line 467)
        self_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'self')
        # Obtaining the member 'pos_child' of a type (line 467)
        pos_child_1554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 21), self_1553, 'pos_child')
        # Assigning a type to the variable 'pos_child_1554' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'pos_child_1554', pos_child_1554)
        # Testing if the for loop is going to be iterated (line 467)
        # Testing the type of a for loop iterable (line 467)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 467, 8), pos_child_1554)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 467, 8), pos_child_1554):
            # Getting the type of the for loop variable (line 467)
            for_loop_var_1555 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 467, 8), pos_child_1554)
            # Assigning a type to the variable 'child' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'child', for_loop_var_1555)
            # SSA begins for a for statement (line 467)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            # Getting the type of 'child' (line 470)
            child_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'child')
            
            # Getting the type of 'child' (line 470)
            child_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'child')
            # Obtaining the member 'wins' of a type (line 470)
            wins_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 26), child_1557, 'wins')
            # Getting the type of 'child' (line 470)
            child_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 39), 'child')
            # Obtaining the member 'losses' of a type (line 470)
            losses_1560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 39), child_1559, 'losses')
            # Applying the binary operator '+' (line 470)
            result_add_1561 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 26), '+', wins_1558, losses_1560)
            
            # Getting the type of 'maxvisits' (line 470)
            maxvisits_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 55), 'maxvisits')
            # Applying the binary operator '>' (line 470)
            result_gt_1563 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 25), '>', result_add_1561, maxvisits_1562)
            
            # Applying the binary operator 'and' (line 470)
            result_and_keyword_1564 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 15), 'and', child_1556, result_gt_1563)
            
            # Testing if the type of an if condition is none (line 470)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 470, 12), result_and_keyword_1564):
                pass
            else:
                
                # Testing the type of an if condition (line 470)
                if_condition_1565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 12), result_and_keyword_1564)
                # Assigning a type to the variable 'if_condition_1565' (line 470)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'if_condition_1565', if_condition_1565)
                # SSA begins for if statement (line 470)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Tuple to a Tuple (line 471):
                
                # Assigning a BinOp to a Name (line 471):
                # Getting the type of 'child' (line 471)
                child_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 39), 'child')
                # Obtaining the member 'wins' of a type (line 471)
                wins_1567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 39), child_1566, 'wins')
                # Getting the type of 'child' (line 471)
                child_1568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 52), 'child')
                # Obtaining the member 'losses' of a type (line 471)
                losses_1569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 52), child_1568, 'losses')
                # Applying the binary operator '+' (line 471)
                result_add_1570 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 39), '+', wins_1567, losses_1569)
                
                # Assigning a type to the variable 'tuple_assignment_11' (line 471)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_11', result_add_1570)
                
                # Assigning a Name to a Name (line 471):
                # Getting the type of 'child' (line 471)
                child_1571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 67), 'child')
                # Assigning a type to the variable 'tuple_assignment_12' (line 471)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_12', child_1571)
                
                # Assigning a Name to a Name (line 471):
                # Getting the type of 'tuple_assignment_11' (line 471)
                tuple_assignment_11_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_11')
                # Assigning a type to the variable 'maxvisits' (line 471)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'maxvisits', tuple_assignment_11_1572)
                
                # Assigning a Name to a Name (line 471):
                # Getting the type of 'tuple_assignment_12' (line 471)
                tuple_assignment_12_1573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_12')
                # Assigning a type to the variable 'maxchild' (line 471)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 27), 'maxchild', tuple_assignment_12_1573)
                # SSA join for if statement (line 470)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'maxchild' (line 472)
        maxchild_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'maxchild')
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'stypy_return_type', maxchild_1574)
        
        # ################# End of 'best_visited(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'best_visited' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'best_visited'
        return stypy_return_type_1575


# Assigning a type to the variable 'UCTNode' (line 345)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'UCTNode', UCTNode)

@norecursion
def user_move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'user_move'
    module_type_store = module_type_store.open_function_context('user_move', 475, 0, False)
    
    # Passed parameters checking function
    user_move.stypy_localization = localization
    user_move.stypy_type_of_self = None
    user_move.stypy_type_store = module_type_store
    user_move.stypy_function_name = 'user_move'
    user_move.stypy_param_names_list = ['board']
    user_move.stypy_varargs_param_name = None
    user_move.stypy_kwargs_param_name = None
    user_move.stypy_call_defaults = defaults
    user_move.stypy_call_varargs = varargs
    user_move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'user_move', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'user_move', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'user_move(...)' code ##################

    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to to_pos(...): (line 476)
    # Processing the call arguments (line 476)
    int_1577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 17), 'int')
    int_1578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 20), 'int')
    # Processing the call keyword arguments (line 476)
    kwargs_1579 = {}
    # Getting the type of 'to_pos' (line 476)
    to_pos_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 10), 'to_pos', False)
    # Calling to_pos(args, kwargs) (line 476)
    to_pos_call_result_1580 = invoke(stypy.reporting.localization.Localization(__file__, 476, 10), to_pos_1576, *[int_1577, int_1578], **kwargs_1579)
    
    # Assigning a type to the variable 'pos' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'pos', to_pos_call_result_1580)
    # Getting the type of 'pos' (line 477)
    pos_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'pos')
    # Assigning a type to the variable 'stypy_return_type' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type', pos_1581)
    
    # ################# End of 'user_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'user_move' in the type store
    # Getting the type of 'stypy_return_type' (line 475)
    stypy_return_type_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1582)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'user_move'
    return stypy_return_type_1582

# Assigning a type to the variable 'user_move' (line 475)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'user_move', user_move)

@norecursion
def computer_move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'computer_move'
    module_type_store = module_type_store.open_function_context('computer_move', 496, 0, False)
    
    # Passed parameters checking function
    computer_move.stypy_localization = localization
    computer_move.stypy_type_of_self = None
    computer_move.stypy_type_store = module_type_store
    computer_move.stypy_function_name = 'computer_move'
    computer_move.stypy_param_names_list = ['board']
    computer_move.stypy_varargs_param_name = None
    computer_move.stypy_kwargs_param_name = None
    computer_move.stypy_call_defaults = defaults
    computer_move.stypy_call_varargs = varargs
    computer_move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'computer_move', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'computer_move', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'computer_move(...)' code ##################

    # Marking variables as global (line 497)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 497, 4), 'MOVES')
    
    # Assigning a Call to a Name (line 498):
    
    # Assigning a Call to a Name (line 498):
    
    # Call to random_move(...): (line 498)
    # Processing the call keyword arguments (line 498)
    kwargs_1585 = {}
    # Getting the type of 'board' (line 498)
    board_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 10), 'board', False)
    # Obtaining the member 'random_move' of a type (line 498)
    random_move_1584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 10), board_1583, 'random_move')
    # Calling random_move(args, kwargs) (line 498)
    random_move_call_result_1586 = invoke(stypy.reporting.localization.Localization(__file__, 498, 10), random_move_1584, *[], **kwargs_1585)
    
    # Assigning a type to the variable 'pos' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'pos', random_move_call_result_1586)
    
    # Getting the type of 'pos' (line 499)
    pos_1587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 7), 'pos')
    # Getting the type of 'PASS' (line 499)
    PASS_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 14), 'PASS')
    # Applying the binary operator '==' (line 499)
    result_eq_1589 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 7), '==', pos_1587, PASS_1588)
    
    # Testing if the type of an if condition is none (line 499)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 499, 4), result_eq_1589):
        pass
    else:
        
        # Testing the type of an if condition (line 499)
        if_condition_1590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 4), result_eq_1589)
        # Assigning a type to the variable 'if_condition_1590' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'if_condition_1590', if_condition_1590)
        # SSA begins for if statement (line 499)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'PASS' (line 500)
        PASS_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'PASS')
        # Assigning a type to the variable 'stypy_return_type' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'stypy_return_type', PASS_1591)
        # SSA join for if statement (line 499)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 501):
    
    # Assigning a Call to a Name (line 501):
    
    # Call to UCTNode(...): (line 501)
    # Processing the call keyword arguments (line 501)
    kwargs_1593 = {}
    # Getting the type of 'UCTNode' (line 501)
    UCTNode_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 11), 'UCTNode', False)
    # Calling UCTNode(args, kwargs) (line 501)
    UCTNode_call_result_1594 = invoke(stypy.reporting.localization.Localization(__file__, 501, 11), UCTNode_1592, *[], **kwargs_1593)
    
    # Assigning a type to the variable 'tree' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tree', UCTNode_call_result_1594)
    
    # Assigning a Call to a Attribute (line 502):
    
    # Assigning a Call to a Attribute (line 502):
    
    # Call to useful_moves(...): (line 502)
    # Processing the call keyword arguments (line 502)
    kwargs_1597 = {}
    # Getting the type of 'board' (line 502)
    board_1595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'board', False)
    # Obtaining the member 'useful_moves' of a type (line 502)
    useful_moves_1596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 22), board_1595, 'useful_moves')
    # Calling useful_moves(args, kwargs) (line 502)
    useful_moves_call_result_1598 = invoke(stypy.reporting.localization.Localization(__file__, 502, 22), useful_moves_1596, *[], **kwargs_1597)
    
    # Getting the type of 'tree' (line 502)
    tree_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'tree')
    # Setting the type of the member 'unexplored' of a type (line 502)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 4), tree_1599, 'unexplored', useful_moves_call_result_1598)
    
    # Assigning a Call to a Name (line 503):
    
    # Assigning a Call to a Name (line 503):
    
    # Call to Board(...): (line 503)
    # Processing the call keyword arguments (line 503)
    kwargs_1601 = {}
    # Getting the type of 'Board' (line 503)
    Board_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 13), 'Board', False)
    # Calling Board(args, kwargs) (line 503)
    Board_call_result_1602 = invoke(stypy.reporting.localization.Localization(__file__, 503, 13), Board_1600, *[], **kwargs_1601)
    
    # Assigning a type to the variable 'nboard' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'nboard', Board_call_result_1602)
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to min(...): (line 504)
    # Processing the call arguments (line 504)
    int_1604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 16), 'int')
    int_1605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 25), 'int')
    
    # Call to len(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'board' (line 504)
    board_1607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'board', False)
    # Obtaining the member 'history' of a type (line 504)
    history_1608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 36), board_1607, 'history')
    # Processing the call keyword arguments (line 504)
    kwargs_1609 = {}
    # Getting the type of 'len' (line 504)
    len_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'len', False)
    # Calling len(args, kwargs) (line 504)
    len_call_result_1610 = invoke(stypy.reporting.localization.Localization(__file__, 504, 32), len_1606, *[history_1608], **kwargs_1609)
    
    # Applying the binary operator '*' (line 504)
    result_mul_1611 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 25), '*', int_1605, len_call_result_1610)
    
    int_1612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 54), 'int')
    # Applying the binary operator 'div' (line 504)
    result_div_1613 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 24), 'div', result_mul_1611, int_1612)
    
    # Applying the binary operator '-' (line 504)
    result_sub_1614 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 16), '-', int_1604, result_div_1613)
    
    int_1615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 57), 'int')
    # Processing the call keyword arguments (line 504)
    kwargs_1616 = {}
    # Getting the type of 'min' (line 504)
    min_1603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'min', False)
    # Calling min(args, kwargs) (line 504)
    min_call_result_1617 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), min_1603, *[result_sub_1614, int_1615], **kwargs_1616)
    
    # Assigning a type to the variable 'GAMES' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'GAMES', min_call_result_1617)
    
    
    # Call to range(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'GAMES' (line 506)
    GAMES_1619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 22), 'GAMES', False)
    # Processing the call keyword arguments (line 506)
    kwargs_1620 = {}
    # Getting the type of 'range' (line 506)
    range_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'range', False)
    # Calling range(args, kwargs) (line 506)
    range_call_result_1621 = invoke(stypy.reporting.localization.Localization(__file__, 506, 16), range_1618, *[GAMES_1619], **kwargs_1620)
    
    # Assigning a type to the variable 'range_call_result_1621' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'range_call_result_1621', range_call_result_1621)
    # Testing if the for loop is going to be iterated (line 506)
    # Testing the type of a for loop iterable (line 506)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 506, 4), range_call_result_1621)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 506, 4), range_call_result_1621):
        # Getting the type of the for loop variable (line 506)
        for_loop_var_1622 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 506, 4), range_call_result_1621)
        # Assigning a type to the variable 'game' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'game', for_loop_var_1622)
        # SSA begins for a for statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 507):
        
        # Assigning a Name to a Name (line 507):
        # Getting the type of 'tree' (line 507)
        tree_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 15), 'tree')
        # Assigning a type to the variable 'node' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'node', tree_1623)
        
        # Call to reset(...): (line 508)
        # Processing the call keyword arguments (line 508)
        kwargs_1626 = {}
        # Getting the type of 'nboard' (line 508)
        nboard_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'nboard', False)
        # Obtaining the member 'reset' of a type (line 508)
        reset_1625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), nboard_1624, 'reset')
        # Calling reset(args, kwargs) (line 508)
        reset_call_result_1627 = invoke(stypy.reporting.localization.Localization(__file__, 508, 8), reset_1625, *[], **kwargs_1626)
        
        
        # Call to replay(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'board' (line 509)
        board_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 22), 'board', False)
        # Obtaining the member 'history' of a type (line 509)
        history_1631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 22), board_1630, 'history')
        # Processing the call keyword arguments (line 509)
        kwargs_1632 = {}
        # Getting the type of 'nboard' (line 509)
        nboard_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'nboard', False)
        # Obtaining the member 'replay' of a type (line 509)
        replay_1629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), nboard_1628, 'replay')
        # Calling replay(args, kwargs) (line 509)
        replay_call_result_1633 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), replay_1629, *[history_1631], **kwargs_1632)
        
        
        # Call to play(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'nboard' (line 510)
        nboard_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 18), 'nboard', False)
        # Processing the call keyword arguments (line 510)
        kwargs_1637 = {}
        # Getting the type of 'node' (line 510)
        node_1634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'node', False)
        # Obtaining the member 'play' of a type (line 510)
        play_1635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), node_1634, 'play')
        # Calling play(args, kwargs) (line 510)
        play_call_result_1638 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), play_1635, *[nboard_1636], **kwargs_1637)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to best_visited(...): (line 514)
    # Processing the call keyword arguments (line 514)
    kwargs_1641 = {}
    # Getting the type of 'tree' (line 514)
    tree_1639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'tree', False)
    # Obtaining the member 'best_visited' of a type (line 514)
    best_visited_1640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 11), tree_1639, 'best_visited')
    # Calling best_visited(args, kwargs) (line 514)
    best_visited_call_result_1642 = invoke(stypy.reporting.localization.Localization(__file__, 514, 11), best_visited_1640, *[], **kwargs_1641)
    
    # Obtaining the member 'pos' of a type (line 514)
    pos_1643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 11), best_visited_call_result_1642, 'pos')
    # Assigning a type to the variable 'stypy_return_type' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type', pos_1643)
    
    # ################# End of 'computer_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'computer_move' in the type store
    # Getting the type of 'stypy_return_type' (line 496)
    stypy_return_type_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1644)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'computer_move'
    return stypy_return_type_1644

# Assigning a type to the variable 'computer_move' (line 496)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'computer_move', computer_move)

@norecursion
def versus_cpu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'versus_cpu'
    module_type_store = module_type_store.open_function_context('versus_cpu', 517, 0, False)
    
    # Passed parameters checking function
    versus_cpu.stypy_localization = localization
    versus_cpu.stypy_type_of_self = None
    versus_cpu.stypy_type_store = module_type_store
    versus_cpu.stypy_function_name = 'versus_cpu'
    versus_cpu.stypy_param_names_list = []
    versus_cpu.stypy_varargs_param_name = None
    versus_cpu.stypy_kwargs_param_name = None
    versus_cpu.stypy_call_defaults = defaults
    versus_cpu.stypy_call_varargs = varargs
    versus_cpu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'versus_cpu', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'versus_cpu', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'versus_cpu(...)' code ##################

    
    # Assigning a Call to a Name (line 518):
    
    # Assigning a Call to a Name (line 518):
    
    # Call to Board(...): (line 518)
    # Processing the call keyword arguments (line 518)
    kwargs_1646 = {}
    # Getting the type of 'Board' (line 518)
    Board_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'Board', False)
    # Calling Board(args, kwargs) (line 518)
    Board_call_result_1647 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), Board_1645, *[], **kwargs_1646)
    
    # Assigning a type to the variable 'board' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'board', Board_call_result_1647)
    
    # Assigning a Num to a Name (line 519):
    
    # Assigning a Num to a Name (line 519):
    int_1648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 15), 'int')
    # Assigning a type to the variable 'maxturns' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'maxturns', int_1648)
    
    # Assigning a Num to a Name (line 520):
    
    # Assigning a Num to a Name (line 520):
    int_1649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
    # Assigning a type to the variable 'turns' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'turns', int_1649)
    
    
    # Getting the type of 'turns' (line 521)
    turns_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 10), 'turns')
    # Getting the type of 'maxturns' (line 521)
    maxturns_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 18), 'maxturns')
    # Applying the binary operator '<' (line 521)
    result_lt_1652 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 10), '<', turns_1650, maxturns_1651)
    
    # Assigning a type to the variable 'result_lt_1652' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'result_lt_1652', result_lt_1652)
    # Testing if the while is going to be iterated (line 521)
    # Testing the type of an if condition (line 521)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 4), result_lt_1652)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 521, 4), result_lt_1652):
        # SSA begins for while statement (line 521)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'board' (line 522)
        board_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 11), 'board')
        # Obtaining the member 'lastmove' of a type (line 522)
        lastmove_1654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 11), board_1653, 'lastmove')
        # Getting the type of 'PASS' (line 522)
        PASS_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 29), 'PASS')
        # Applying the binary operator '!=' (line 522)
        result_ne_1656 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 11), '!=', lastmove_1654, PASS_1655)
        
        # Testing if the type of an if condition is none (line 522)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 522, 8), result_ne_1656):
            pass
        else:
            
            # Testing the type of an if condition (line 522)
            if_condition_1657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 8), result_ne_1656)
            # Assigning a type to the variable 'if_condition_1657' (line 522)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'if_condition_1657', if_condition_1657)
            # SSA begins for if statement (line 522)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 522)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 525):
        
        # Assigning a Call to a Name (line 525):
        
        # Call to computer_move(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'board' (line 525)
        board_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 28), 'board', False)
        # Processing the call keyword arguments (line 525)
        kwargs_1660 = {}
        # Getting the type of 'computer_move' (line 525)
        computer_move_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 14), 'computer_move', False)
        # Calling computer_move(args, kwargs) (line 525)
        computer_move_call_result_1661 = invoke(stypy.reporting.localization.Localization(__file__, 525, 14), computer_move_1658, *[board_1659], **kwargs_1660)
        
        # Assigning a type to the variable 'pos' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'pos', computer_move_call_result_1661)
        
        # Getting the type of 'pos' (line 526)
        pos_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 11), 'pos')
        # Getting the type of 'PASS' (line 526)
        PASS_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 18), 'PASS')
        # Applying the binary operator '==' (line 526)
        result_eq_1664 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 11), '==', pos_1662, PASS_1663)
        
        # Testing if the type of an if condition is none (line 526)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 526, 8), result_eq_1664):
            
            # Call to to_xy(...): (line 530)
            # Processing the call arguments (line 530)
            # Getting the type of 'pos' (line 530)
            pos_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 18), 'pos', False)
            # Processing the call keyword arguments (line 530)
            kwargs_1668 = {}
            # Getting the type of 'to_xy' (line 530)
            to_xy_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'to_xy', False)
            # Calling to_xy(args, kwargs) (line 530)
            to_xy_call_result_1669 = invoke(stypy.reporting.localization.Localization(__file__, 530, 12), to_xy_1666, *[pos_1667], **kwargs_1668)
            
        else:
            
            # Testing the type of an if condition (line 526)
            if_condition_1665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 526, 8), result_eq_1664)
            # Assigning a type to the variable 'if_condition_1665' (line 526)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'if_condition_1665', if_condition_1665)
            # SSA begins for if statement (line 526)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 526)
            module_type_store.open_ssa_branch('else')
            
            # Call to to_xy(...): (line 530)
            # Processing the call arguments (line 530)
            # Getting the type of 'pos' (line 530)
            pos_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 18), 'pos', False)
            # Processing the call keyword arguments (line 530)
            kwargs_1668 = {}
            # Getting the type of 'to_xy' (line 530)
            to_xy_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'to_xy', False)
            # Calling to_xy(args, kwargs) (line 530)
            to_xy_call_result_1669 = invoke(stypy.reporting.localization.Localization(__file__, 530, 12), to_xy_1666, *[pos_1667], **kwargs_1668)
            
            # SSA join for if statement (line 526)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to move(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'pos' (line 531)
        pos_1672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 19), 'pos', False)
        # Processing the call keyword arguments (line 531)
        kwargs_1673 = {}
        # Getting the type of 'board' (line 531)
        board_1670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'board', False)
        # Obtaining the member 'move' of a type (line 531)
        move_1671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), board_1670, 'move')
        # Calling move(args, kwargs) (line 531)
        move_call_result_1674 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), move_1671, *[pos_1672], **kwargs_1673)
        
        # Getting the type of 'board' (line 534)
        board_1675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 11), 'board')
        # Obtaining the member 'finished' of a type (line 534)
        finished_1676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 11), board_1675, 'finished')
        # Testing if the type of an if condition is none (line 534)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 534, 8), finished_1676):
            pass
        else:
            
            # Testing the type of an if condition (line 534)
            if_condition_1677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 8), finished_1676)
            # Assigning a type to the variable 'if_condition_1677' (line 534)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'if_condition_1677', if_condition_1677)
            # SSA begins for if statement (line 534)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 534)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'board' (line 536)
        board_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'board')
        # Obtaining the member 'lastmove' of a type (line 536)
        lastmove_1679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 11), board_1678, 'lastmove')
        # Getting the type of 'PASS' (line 536)
        PASS_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 29), 'PASS')
        # Applying the binary operator '!=' (line 536)
        result_ne_1681 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), '!=', lastmove_1679, PASS_1680)
        
        # Testing if the type of an if condition is none (line 536)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 536, 8), result_ne_1681):
            pass
        else:
            
            # Testing the type of an if condition (line 536)
            if_condition_1682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), result_ne_1681)
            # Assigning a type to the variable 'if_condition_1682' (line 536)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_1682', if_condition_1682)
            # SSA begins for if statement (line 536)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 536)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 538):
        
        # Assigning a Call to a Name (line 538):
        
        # Call to user_move(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'board' (line 538)
        board_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 24), 'board', False)
        # Processing the call keyword arguments (line 538)
        kwargs_1685 = {}
        # Getting the type of 'user_move' (line 538)
        user_move_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 14), 'user_move', False)
        # Calling user_move(args, kwargs) (line 538)
        user_move_call_result_1686 = invoke(stypy.reporting.localization.Localization(__file__, 538, 14), user_move_1683, *[board_1684], **kwargs_1685)
        
        # Assigning a type to the variable 'pos' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'pos', user_move_call_result_1686)
        
        # Call to move(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'pos' (line 539)
        pos_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 19), 'pos', False)
        # Processing the call keyword arguments (line 539)
        kwargs_1690 = {}
        # Getting the type of 'board' (line 539)
        board_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'board', False)
        # Obtaining the member 'move' of a type (line 539)
        move_1688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 8), board_1687, 'move')
        # Calling move(args, kwargs) (line 539)
        move_call_result_1691 = invoke(stypy.reporting.localization.Localization(__file__, 539, 8), move_1688, *[pos_1689], **kwargs_1690)
        
        # Getting the type of 'board' (line 541)
        board_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 11), 'board')
        # Obtaining the member 'finished' of a type (line 541)
        finished_1693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 11), board_1692, 'finished')
        # Testing if the type of an if condition is none (line 541)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 541, 8), finished_1693):
            pass
        else:
            
            # Testing the type of an if condition (line 541)
            if_condition_1694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 8), finished_1693)
            # Assigning a type to the variable 'if_condition_1694' (line 541)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'if_condition_1694', if_condition_1694)
            # SSA begins for if statement (line 541)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 541)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'turns' (line 543)
        turns_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'turns')
        int_1696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 17), 'int')
        # Applying the binary operator '+=' (line 543)
        result_iadd_1697 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 8), '+=', turns_1695, int_1696)
        # Assigning a type to the variable 'turns' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'turns', result_iadd_1697)
        
        # SSA join for while statement (line 521)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to score(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'WHITE' (line 545)
    WHITE_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'WHITE', False)
    # Processing the call keyword arguments (line 545)
    kwargs_1701 = {}
    # Getting the type of 'board' (line 545)
    board_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'board', False)
    # Obtaining the member 'score' of a type (line 545)
    score_1699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 4), board_1698, 'score')
    # Calling score(args, kwargs) (line 545)
    score_call_result_1702 = invoke(stypy.reporting.localization.Localization(__file__, 545, 4), score_1699, *[WHITE_1700], **kwargs_1701)
    
    
    # Call to score(...): (line 547)
    # Processing the call arguments (line 547)
    # Getting the type of 'BLACK' (line 547)
    BLACK_1705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'BLACK', False)
    # Processing the call keyword arguments (line 547)
    kwargs_1706 = {}
    # Getting the type of 'board' (line 547)
    board_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'board', False)
    # Obtaining the member 'score' of a type (line 547)
    score_1704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 4), board_1703, 'score')
    # Calling score(args, kwargs) (line 547)
    score_call_result_1707 = invoke(stypy.reporting.localization.Localization(__file__, 547, 4), score_1704, *[BLACK_1705], **kwargs_1706)
    
    
    # ################# End of 'versus_cpu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'versus_cpu' in the type store
    # Getting the type of 'stypy_return_type' (line 517)
    stypy_return_type_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1708)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'versus_cpu'
    return stypy_return_type_1708

# Assigning a type to the variable 'versus_cpu' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'versus_cpu', versus_cpu)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 550, 0, False)
    
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

    
    # Call to seed(...): (line 551)
    # Processing the call arguments (line 551)
    int_1711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 16), 'int')
    # Processing the call keyword arguments (line 551)
    kwargs_1712 = {}
    # Getting the type of 'random' (line 551)
    random_1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'random', False)
    # Obtaining the member 'seed' of a type (line 551)
    seed_1710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 4), random_1709, 'seed')
    # Calling seed(args, kwargs) (line 551)
    seed_call_result_1713 = invoke(stypy.reporting.localization.Localization(__file__, 551, 4), seed_1710, *[int_1711], **kwargs_1712)
    
    
    
    # SSA begins for try-except statement (line 552)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to versus_cpu(...): (line 553)
    # Processing the call keyword arguments (line 553)
    kwargs_1715 = {}
    # Getting the type of 'versus_cpu' (line 553)
    versus_cpu_1714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'versus_cpu', False)
    # Calling versus_cpu(args, kwargs) (line 553)
    versus_cpu_call_result_1716 = invoke(stypy.reporting.localization.Localization(__file__, 553, 8), versus_cpu_1714, *[], **kwargs_1715)
    
    # SSA branch for the except part of a try statement (line 552)
    # SSA branch for the except 'EOFError' branch of a try statement (line 552)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 552)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 556)
    True_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'stypy_return_type', True_1717)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 550)
    stypy_return_type_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1718

# Assigning a type to the variable 'run' (line 550)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'run', run)

# Call to run(...): (line 559)
# Processing the call keyword arguments (line 559)
kwargs_1720 = {}
# Getting the type of 'run' (line 559)
run_1719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 0), 'run', False)
# Calling run(args, kwargs) (line 559)
run_call_result_1721 = invoke(stypy.reporting.localization.Localization(__file__, 559, 0), run_1719, *[], **kwargs_1720)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

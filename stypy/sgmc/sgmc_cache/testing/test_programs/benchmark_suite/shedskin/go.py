
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
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'int')
    # Processing the call keyword arguments
    kwargs_52 = {}
    # Getting the type of 'call_assignment_4' (line 21)
    call_assignment_4_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_4', False)
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), call_assignment_4_49, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___50, *[int_51], **kwargs_52)
    
    # Assigning a type to the variable 'call_assignment_5' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_5', getitem___call_result_53)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'call_assignment_5' (line 21)
    call_assignment_5_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_5')
    # Assigning a type to the variable 'y' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'y', call_assignment_5_54)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'int')
    # Processing the call keyword arguments
    kwargs_58 = {}
    # Getting the type of 'call_assignment_4' (line 21)
    call_assignment_4_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_4', False)
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), call_assignment_4_55, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_59 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56, *[int_57], **kwargs_58)
    
    # Assigning a type to the variable 'call_assignment_6' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_6', getitem___call_result_59)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'call_assignment_6' (line 21)
    call_assignment_6_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'call_assignment_6')
    # Assigning a type to the variable 'x' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'x', call_assignment_6_60)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    # Getting the type of 'x' (line 22)
    x_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_61, x_62)
    # Adding element type (line 22)
    # Getting the type of 'y' (line 22)
    y_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_61, y_63)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', tuple_61)
    
    # ################# End of 'to_xy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_xy' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_64)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_xy'
    return stypy_return_type_64

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
        board_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'board')
        # Getting the type of 'self' (line 27)
        self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'board' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_66, 'board', board_65)
        
        # Assigning a Name to a Attribute (line 28):
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'pos' (line 28)
        pos_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'pos')
        # Getting the type of 'self' (line 28)
        self_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_68, 'pos', pos_67)
        
        # Assigning a Num to a Attribute (line 29):
        
        # Assigning a Num to a Attribute (line 29):
        int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'int')
        # Getting the type of 'self' (line 29)
        self_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'liberties' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_70, 'liberties', int_69)
        
        # Assigning a Name to a Attribute (line 30):
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'TIMESTAMP' (line 30)
        TIMESTAMP_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 30)
        self_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'timestamp' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_72, 'timestamp', TIMESTAMP_71)
        
        # Assigning a Name to a Attribute (line 31):
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'TIMESTAMP' (line 31)
        TIMESTAMP_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'TIMESTAMP')
        # Getting the type of 'self' (line 31)
        self_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'timestamp2' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_74, 'timestamp2', TIMESTAMP_73)
        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'TIMESTAMP' (line 32)
        TIMESTAMP_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 32)
        self_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'findstamp' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_76, 'findstamp', TIMESTAMP_75)
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'REMOVESTAMP' (line 33)
        REMOVESTAMP_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'REMOVESTAMP')
        # Getting the type of 'self' (line 33)
        self_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'removestamp' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_78, 'removestamp', REMOVESTAMP_77)
        
        # Assigning a ListComp to a Attribute (line 34):
        
        # Assigning a ListComp to a Attribute (line 34):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 34)
        # Processing the call arguments (line 34)
        int_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 76), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_87 = {}
        # Getting the type of 'range' (line 34)
        range_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 70), 'range', False)
        # Calling range(args, kwargs) (line 34)
        range_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 34, 70), range_85, *[int_86], **kwargs_87)
        
        comprehension_89 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), range_call_result_88)
        # Assigning a type to the variable 'i' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'i', comprehension_89)
        
        # Call to randrange(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'sys' (line 34)
        sys_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 49), 'sys', False)
        # Obtaining the member 'maxint' of a type (line 34)
        maxint_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 49), sys_81, 'maxint')
        # Processing the call keyword arguments (line 34)
        kwargs_83 = {}
        # Getting the type of 'random' (line 34)
        random_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'random', False)
        # Obtaining the member 'randrange' of a type (line 34)
        randrange_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 32), random_79, 'randrange')
        # Calling randrange(args, kwargs) (line 34)
        randrange_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 34, 32), randrange_80, *[maxint_82], **kwargs_83)
        
        list_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), list_90, randrange_call_result_84)
        # Getting the type of 'self' (line 34)
        self_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'zobrist_strings' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_91, 'zobrist_strings', list_90)
        
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
        self_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self')
        # Obtaining the member 'pos' of a type (line 37)
        pos_93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), self_92, 'pos')
        # Getting the type of 'SIZE' (line 37)
        SIZE_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'SIZE')
        # Applying the binary operator '%' (line 37)
        result_mod_95 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 15), '%', pos_93, SIZE_94)
        
        # Assigning a type to the variable 'tuple_assignment_7' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_7', result_mod_95)
        
        # Assigning a BinOp to a Name (line 37):
        # Getting the type of 'self' (line 37)
        self_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 32), 'self')
        # Obtaining the member 'pos' of a type (line 37)
        pos_97 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 32), self_96, 'pos')
        # Getting the type of 'SIZE' (line 37)
        SIZE_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 43), 'SIZE')
        # Applying the binary operator 'div' (line 37)
        result_div_99 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 32), 'div', pos_97, SIZE_98)
        
        # Assigning a type to the variable 'tuple_assignment_8' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_8', result_div_99)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'tuple_assignment_7' (line 37)
        tuple_assignment_7_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_7')
        # Assigning a type to the variable 'x' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'x', tuple_assignment_7_100)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'tuple_assignment_8' (line 37)
        tuple_assignment_8_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_8')
        # Assigning a type to the variable 'y' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'y', tuple_assignment_8_101)
        
        # Assigning a List to a Attribute (line 38):
        
        # Assigning a List to a Attribute (line 38):
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        
        # Getting the type of 'self' (line 38)
        self_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'neighbours' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_103, 'neighbours', list_102)
        
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 24), tuple_105, int_106)
        # Adding element type (line 39)
        int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 24), tuple_105, int_107)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_104, tuple_105)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 33), tuple_108, int_109)
        # Adding element type (line 39)
        int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 33), tuple_108, int_110)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_104, tuple_108)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 41), tuple_111, int_112)
        # Adding element type (line 39)
        int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 41), tuple_111, int_113)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_104, tuple_111)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 50), tuple_114, int_115)
        # Adding element type (line 39)
        int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 50), tuple_114, int_116)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_104, tuple_114)
        
        # Testing if the loop is going to be iterated (line 39)
        # Testing the type of a for loop iterable (line 39)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 8), list_104)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 39, 8), list_104):
            # Getting the type of the for loop variable (line 39)
            for_loop_var_117 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 8), list_104)
            # Assigning a type to the variable 'dx' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'dx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 8), for_loop_var_117))
            # Assigning a type to the variable 'dy' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'dy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 8), for_loop_var_117))
            # SSA begins for a for statement (line 39)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Tuple to a Tuple (line 40):
            
            # Assigning a BinOp to a Name (line 40):
            # Getting the type of 'x' (line 40)
            x_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'x')
            # Getting the type of 'dx' (line 40)
            dx_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'dx')
            # Applying the binary operator '+' (line 40)
            result_add_120 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 25), '+', x_118, dx_119)
            
            # Assigning a type to the variable 'tuple_assignment_9' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_9', result_add_120)
            
            # Assigning a BinOp to a Name (line 40):
            # Getting the type of 'y' (line 40)
            y_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'y')
            # Getting the type of 'dy' (line 40)
            dy_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 37), 'dy')
            # Applying the binary operator '+' (line 40)
            result_add_123 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 33), '+', y_121, dy_122)
            
            # Assigning a type to the variable 'tuple_assignment_10' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_10', result_add_123)
            
            # Assigning a Name to a Name (line 40):
            # Getting the type of 'tuple_assignment_9' (line 40)
            tuple_assignment_9_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_9')
            # Assigning a type to the variable 'newx' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'newx', tuple_assignment_9_124)
            
            # Assigning a Name to a Name (line 40):
            # Getting the type of 'tuple_assignment_10' (line 40)
            tuple_assignment_10_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple_assignment_10')
            # Assigning a type to the variable 'newy' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'newy', tuple_assignment_10_125)
            
            
            # Evaluating a boolean operation
            
            int_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'int')
            # Getting the type of 'newx' (line 41)
            newx_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'newx')
            # Applying the binary operator '<=' (line 41)
            result_le_128 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '<=', int_126, newx_127)
            # Getting the type of 'SIZE' (line 41)
            SIZE_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'SIZE')
            # Applying the binary operator '<' (line 41)
            result_lt_130 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '<', newx_127, SIZE_129)
            # Applying the binary operator '&' (line 41)
            result_and__131 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '&', result_le_128, result_lt_130)
            
            
            int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'int')
            # Getting the type of 'newy' (line 41)
            newy_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), 'newy')
            # Applying the binary operator '<=' (line 41)
            result_le_134 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '<=', int_132, newy_133)
            # Getting the type of 'SIZE' (line 41)
            SIZE_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'SIZE')
            # Applying the binary operator '<' (line 41)
            result_lt_136 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '<', newy_133, SIZE_135)
            # Applying the binary operator '&' (line 41)
            result_and__137 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '&', result_le_134, result_lt_136)
            
            # Applying the binary operator 'and' (line 41)
            result_and_keyword_138 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), 'and', result_and__131, result_and__137)
            
            # Testing the type of an if condition (line 41)
            if_condition_139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), result_and_keyword_138)
            # Assigning a type to the variable 'if_condition_139' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_139', if_condition_139)
            # SSA begins for if statement (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 42)
            # Processing the call arguments (line 42)
            
            # Obtaining the type of the subscript
            
            # Call to to_pos(...): (line 42)
            # Processing the call arguments (line 42)
            # Getting the type of 'newx' (line 42)
            newx_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 65), 'newx', False)
            # Getting the type of 'newy' (line 42)
            newy_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 71), 'newy', False)
            # Processing the call keyword arguments (line 42)
            kwargs_146 = {}
            # Getting the type of 'to_pos' (line 42)
            to_pos_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 58), 'to_pos', False)
            # Calling to_pos(args, kwargs) (line 42)
            to_pos_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 42, 58), to_pos_143, *[newx_144, newy_145], **kwargs_146)
            
            # Getting the type of 'self' (line 42)
            self_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'self', False)
            # Obtaining the member 'board' of a type (line 42)
            board_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), self_148, 'board')
            # Obtaining the member 'squares' of a type (line 42)
            squares_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), board_149, 'squares')
            # Obtaining the member '__getitem__' of a type (line 42)
            getitem___151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), squares_150, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 42)
            subscript_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 42, 39), getitem___151, to_pos_call_result_147)
            
            # Processing the call keyword arguments (line 42)
            kwargs_153 = {}
            # Getting the type of 'self' (line 42)
            self_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'self', False)
            # Obtaining the member 'neighbours' of a type (line 42)
            neighbours_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), self_140, 'neighbours')
            # Obtaining the member 'append' of a type (line 42)
            append_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), neighbours_141, 'append')
            # Calling append(args, kwargs) (line 42)
            append_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), append_142, *[subscript_call_result_152], **kwargs_153)
            
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'set_neighbours(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_neighbours' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_neighbours'
        return stypy_return_type_155


    @norecursion
    def count_liberties(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 44)
        None_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'None')
        defaults = [None_156]
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
        reference_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'reference')
        # Applying the 'not' unary operator (line 45)
        result_not__158 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), 'not', reference_157)
        
        # Testing the type of an if condition (line 45)
        if_condition_159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_not__158)
        # Assigning a type to the variable 'if_condition_159' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_159', if_condition_159)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 46):
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'self' (line 46)
        self_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'self')
        # Assigning a type to the variable 'reference' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'reference', self_160)
        
        # Assigning a Num to a Attribute (line 47):
        
        # Assigning a Num to a Attribute (line 47):
        int_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'int')
        # Getting the type of 'self' (line 47)
        self_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self')
        # Setting the type of the member 'liberties' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_162, 'liberties', int_161)
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'TIMESTAMP' (line 48)
        TIMESTAMP_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 48)
        self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'timestamp' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_164, 'timestamp', TIMESTAMP_163)
        
        # Getting the type of 'self' (line 49)
        self_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 49)
        neighbours_166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), self_165, 'neighbours')
        # Testing if the loop is going to be iterated (line 49)
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), neighbours_166)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 49, 8), neighbours_166):
            # Getting the type of the for loop variable (line 49)
            for_loop_var_167 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), neighbours_166)
            # Assigning a type to the variable 'neighbour' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'neighbour', for_loop_var_167)
            # SSA begins for a for statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'neighbour' (line 50)
            neighbour_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'neighbour')
            # Obtaining the member 'timestamp' of a type (line 50)
            timestamp_169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), neighbour_168, 'timestamp')
            # Getting the type of 'TIMESTAMP' (line 50)
            TIMESTAMP_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'TIMESTAMP')
            # Applying the binary operator '!=' (line 50)
            result_ne_171 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '!=', timestamp_169, TIMESTAMP_170)
            
            # Testing the type of an if condition (line 50)
            if_condition_172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), result_ne_171)
            # Assigning a type to the variable 'if_condition_172' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_condition_172', if_condition_172)
            # SSA begins for if statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 51):
            
            # Assigning a Name to a Attribute (line 51):
            # Getting the type of 'TIMESTAMP' (line 51)
            TIMESTAMP_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'TIMESTAMP')
            # Getting the type of 'neighbour' (line 51)
            neighbour_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'neighbour')
            # Setting the type of the member 'timestamp' of a type (line 51)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), neighbour_174, 'timestamp', TIMESTAMP_173)
            
            
            # Getting the type of 'neighbour' (line 52)
            neighbour_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'neighbour')
            # Obtaining the member 'color' of a type (line 52)
            color_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), neighbour_175, 'color')
            # Getting the type of 'EMPTY' (line 52)
            EMPTY_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'EMPTY')
            # Applying the binary operator '==' (line 52)
            result_eq_178 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 19), '==', color_176, EMPTY_177)
            
            # Testing the type of an if condition (line 52)
            if_condition_179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 16), result_eq_178)
            # Assigning a type to the variable 'if_condition_179' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'if_condition_179', if_condition_179)
            # SSA begins for if statement (line 52)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'reference' (line 53)
            reference_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'reference')
            # Obtaining the member 'liberties' of a type (line 53)
            liberties_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), reference_180, 'liberties')
            int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 43), 'int')
            # Applying the binary operator '+=' (line 53)
            result_iadd_183 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 20), '+=', liberties_181, int_182)
            # Getting the type of 'reference' (line 53)
            reference_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'reference')
            # Setting the type of the member 'liberties' of a type (line 53)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), reference_184, 'liberties', result_iadd_183)
            
            # SSA branch for the else part of an if statement (line 52)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'neighbour' (line 54)
            neighbour_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'neighbour')
            # Obtaining the member 'color' of a type (line 54)
            color_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), neighbour_185, 'color')
            # Getting the type of 'self' (line 54)
            self_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'self')
            # Obtaining the member 'color' of a type (line 54)
            color_188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 40), self_187, 'color')
            # Applying the binary operator '==' (line 54)
            result_eq_189 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 21), '==', color_186, color_188)
            
            # Testing the type of an if condition (line 54)
            if_condition_190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 21), result_eq_189)
            # Assigning a type to the variable 'if_condition_190' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'if_condition_190', if_condition_190)
            # SSA begins for if statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to count_liberties(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'reference' (line 55)
            reference_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'reference', False)
            # Processing the call keyword arguments (line 55)
            kwargs_194 = {}
            # Getting the type of 'neighbour' (line 55)
            neighbour_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'neighbour', False)
            # Obtaining the member 'count_liberties' of a type (line 55)
            count_liberties_192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), neighbour_191, 'count_liberties')
            # Calling count_liberties(args, kwargs) (line 55)
            count_liberties_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), count_liberties_192, *[reference_193], **kwargs_194)
            
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
        stypy_return_type_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_liberties'
        return stypy_return_type_196


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
        TIMESTAMP_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'TIMESTAMP')
        # Getting the type of 'self' (line 58)
        self_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'findstamp' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_198, 'findstamp', TIMESTAMP_197)
        
        # Getting the type of 'self' (line 59)
        self_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 59)
        neighbours_200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 25), self_199, 'neighbours')
        # Testing if the loop is going to be iterated (line 59)
        # Testing the type of a for loop iterable (line 59)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 8), neighbours_200)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 59, 8), neighbours_200):
            # Getting the type of the for loop variable (line 59)
            for_loop_var_201 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 8), neighbours_200)
            # Assigning a type to the variable 'neighbour' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'neighbour', for_loop_var_201)
            # SSA begins for a for statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'neighbour' (line 60)
            neighbour_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'neighbour')
            # Obtaining the member 'findstamp' of a type (line 60)
            findstamp_203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), neighbour_202, 'findstamp')
            # Getting the type of 'TIMESTAMP' (line 60)
            TIMESTAMP_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'TIMESTAMP')
            # Applying the binary operator '!=' (line 60)
            result_ne_205 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 15), '!=', findstamp_203, TIMESTAMP_204)
            
            # Testing the type of an if condition (line 60)
            if_condition_206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 12), result_ne_205)
            # Assigning a type to the variable 'if_condition_206' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'if_condition_206', if_condition_206)
            # SSA begins for if statement (line 60)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 61):
            
            # Assigning a Name to a Attribute (line 61):
            # Getting the type of 'TIMESTAMP' (line 61)
            TIMESTAMP_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'TIMESTAMP')
            # Getting the type of 'neighbour' (line 61)
            neighbour_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'neighbour')
            # Setting the type of the member 'findstamp' of a type (line 61)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), neighbour_208, 'findstamp', TIMESTAMP_207)
            
            
            # Getting the type of 'neighbour' (line 62)
            neighbour_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'neighbour')
            # Obtaining the member 'color' of a type (line 62)
            color_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), neighbour_209, 'color')
            # Getting the type of 'EMPTY' (line 62)
            EMPTY_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'EMPTY')
            # Applying the binary operator '==' (line 62)
            result_eq_212 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 19), '==', color_210, EMPTY_211)
            
            # Testing the type of an if condition (line 62)
            if_condition_213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 16), result_eq_212)
            # Assigning a type to the variable 'if_condition_213' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'if_condition_213', if_condition_213)
            # SSA begins for if statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'neighbour' (line 63)
            neighbour_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'neighbour')
            # Assigning a type to the variable 'stypy_return_type' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'stypy_return_type', neighbour_214)
            # SSA branch for the else part of an if statement (line 62)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'neighbour' (line 64)
            neighbour_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'neighbour')
            # Obtaining the member 'color' of a type (line 64)
            color_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), neighbour_215, 'color')
            # Getting the type of 'self' (line 64)
            self_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'self')
            # Obtaining the member 'color' of a type (line 64)
            color_218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), self_217, 'color')
            # Applying the binary operator '==' (line 64)
            result_eq_219 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 21), '==', color_216, color_218)
            
            # Testing the type of an if condition (line 64)
            if_condition_220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 21), result_eq_219)
            # Assigning a type to the variable 'if_condition_220' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'if_condition_220', if_condition_220)
            # SSA begins for if statement (line 64)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 65):
            
            # Assigning a Call to a Name (line 65):
            
            # Call to liberty(...): (line 65)
            # Processing the call keyword arguments (line 65)
            kwargs_223 = {}
            # Getting the type of 'neighbour' (line 65)
            neighbour_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'neighbour', False)
            # Obtaining the member 'liberty' of a type (line 65)
            liberty_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), neighbour_221, 'liberty')
            # Calling liberty(args, kwargs) (line 65)
            liberty_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), liberty_222, *[], **kwargs_223)
            
            # Assigning a type to the variable 'liberty' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'liberty', liberty_call_result_224)
            
            # Getting the type of 'liberty' (line 66)
            liberty_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'liberty')
            # Testing the type of an if condition (line 66)
            if_condition_226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 20), liberty_225)
            # Assigning a type to the variable 'if_condition_226' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'if_condition_226', if_condition_226)
            # SSA begins for if statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'liberty' (line 67)
            liberty_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'liberty')
            # Assigning a type to the variable 'stypy_return_type' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'stypy_return_type', liberty_227)
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
        stypy_return_type_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'liberty'
        return stypy_return_type_228


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
        TIMESTAMP_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'TIMESTAMP')
        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'int')
        # Applying the binary operator '+=' (line 71)
        result_iadd_231 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 8), '+=', TIMESTAMP_229, int_230)
        # Assigning a type to the variable 'TIMESTAMP' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'TIMESTAMP', result_iadd_231)
        
        
        # Getting the type of 'MOVES' (line 72)
        MOVES_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'MOVES')
        int_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'int')
        # Applying the binary operator '+=' (line 72)
        result_iadd_234 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 8), '+=', MOVES_232, int_233)
        # Assigning a type to the variable 'MOVES' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'MOVES', result_iadd_234)
        
        
        # Call to update(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'self', False)
        # Getting the type of 'color' (line 73)
        color_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'color', False)
        # Processing the call keyword arguments (line 73)
        kwargs_241 = {}
        # Getting the type of 'self' (line 73)
        self_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'board' of a type (line 73)
        board_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_235, 'board')
        # Obtaining the member 'zobrist' of a type (line 73)
        zobrist_237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), board_236, 'zobrist')
        # Obtaining the member 'update' of a type (line 73)
        update_238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), zobrist_237, 'update')
        # Calling update(args, kwargs) (line 73)
        update_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), update_238, *[self_239, color_240], **kwargs_241)
        
        
        # Assigning a Name to a Attribute (line 74):
        
        # Assigning a Name to a Attribute (line 74):
        # Getting the type of 'color' (line 74)
        color_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'color')
        # Getting the type of 'self' (line 74)
        self_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'color' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_244, 'color', color_243)
        
        # Assigning a Name to a Attribute (line 75):
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'self' (line 75)
        self_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'self')
        # Getting the type of 'self' (line 75)
        self_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'reference' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_246, 'reference', self_245)
        
        # Assigning a Num to a Attribute (line 76):
        
        # Assigning a Num to a Attribute (line 76):
        int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'int')
        # Getting the type of 'self' (line 76)
        self_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'members' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_248, 'members', int_247)
        
        # Assigning a Name to a Attribute (line 77):
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'True' (line 77)
        True_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'True')
        # Getting the type of 'self' (line 77)
        self_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'used' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_250, 'used', True_249)
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'None' (line 78)
        None_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'None')
        # Getting the type of 'self' (line 78)
        self_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Obtaining the member 'board' of a type (line 78)
        board_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_252, 'board')
        # Setting the type of the member 'atari' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), board_253, 'atari', None_251)
        
        # Getting the type of 'self' (line 79)
        self_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 79)
        neighbours_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), self_254, 'neighbours')
        # Testing if the loop is going to be iterated (line 79)
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), neighbours_255)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 79, 8), neighbours_255):
            # Getting the type of the for loop variable (line 79)
            for_loop_var_256 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), neighbours_255)
            # Assigning a type to the variable 'neighbour' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'neighbour', for_loop_var_256)
            # SSA begins for a for statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'neighbour' (line 80)
            neighbour_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'neighbour')
            # Obtaining the member 'color' of a type (line 80)
            color_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), neighbour_257, 'color')
            # Getting the type of 'EMPTY' (line 80)
            EMPTY_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'EMPTY')
            # Applying the binary operator '!=' (line 80)
            result_ne_260 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), '!=', color_258, EMPTY_259)
            
            # Testing the type of an if condition (line 80)
            if_condition_261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), result_ne_260)
            # Assigning a type to the variable 'if_condition_261' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_261', if_condition_261)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 81):
            
            # Assigning a Call to a Name (line 81):
            
            # Call to find(...): (line 81)
            # Processing the call keyword arguments (line 81)
            # Getting the type of 'True' (line 81)
            True_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 54), 'True', False)
            keyword_265 = True_264
            kwargs_266 = {'update': keyword_265}
            # Getting the type of 'neighbour' (line 81)
            neighbour_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 32), 'neighbour', False)
            # Obtaining the member 'find' of a type (line 81)
            find_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 32), neighbour_262, 'find')
            # Calling find(args, kwargs) (line 81)
            find_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 81, 32), find_263, *[], **kwargs_266)
            
            # Assigning a type to the variable 'neighbour_ref' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'neighbour_ref', find_call_result_267)
            
            
            # Getting the type of 'neighbour_ref' (line 82)
            neighbour_ref_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'neighbour_ref')
            # Obtaining the member 'timestamp' of a type (line 82)
            timestamp_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 19), neighbour_ref_268, 'timestamp')
            # Getting the type of 'TIMESTAMP' (line 82)
            TIMESTAMP_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 46), 'TIMESTAMP')
            # Applying the binary operator '!=' (line 82)
            result_ne_271 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '!=', timestamp_269, TIMESTAMP_270)
            
            # Testing the type of an if condition (line 82)
            if_condition_272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 16), result_ne_271)
            # Assigning a type to the variable 'if_condition_272' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'if_condition_272', if_condition_272)
            # SSA begins for if statement (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 83):
            
            # Assigning a Name to a Attribute (line 83):
            # Getting the type of 'TIMESTAMP' (line 83)
            TIMESTAMP_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'TIMESTAMP')
            # Getting the type of 'neighbour_ref' (line 83)
            neighbour_ref_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'neighbour_ref')
            # Setting the type of the member 'timestamp' of a type (line 83)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), neighbour_ref_274, 'timestamp', TIMESTAMP_273)
            
            
            # Getting the type of 'neighbour' (line 84)
            neighbour_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'neighbour')
            # Obtaining the member 'color' of a type (line 84)
            color_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), neighbour_275, 'color')
            # Getting the type of 'color' (line 84)
            color_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'color')
            # Applying the binary operator '==' (line 84)
            result_eq_278 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 23), '==', color_276, color_277)
            
            # Testing the type of an if condition (line 84)
            if_condition_279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 20), result_eq_278)
            # Assigning a type to the variable 'if_condition_279' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'if_condition_279', if_condition_279)
            # SSA begins for if statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 85):
            
            # Assigning a Name to a Attribute (line 85):
            # Getting the type of 'self' (line 85)
            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 50), 'self')
            # Getting the type of 'neighbour_ref' (line 85)
            neighbour_ref_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'neighbour_ref')
            # Setting the type of the member 'reference' of a type (line 85)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), neighbour_ref_281, 'reference', self_280)
            
            # Getting the type of 'self' (line 86)
            self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'self')
            # Obtaining the member 'members' of a type (line 86)
            members_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), self_282, 'members')
            # Getting the type of 'neighbour_ref' (line 86)
            neighbour_ref_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 40), 'neighbour_ref')
            # Obtaining the member 'members' of a type (line 86)
            members_285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 40), neighbour_ref_284, 'members')
            # Applying the binary operator '+=' (line 86)
            result_iadd_286 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 24), '+=', members_283, members_285)
            # Getting the type of 'self' (line 86)
            self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'self')
            # Setting the type of the member 'members' of a type (line 86)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), self_287, 'members', result_iadd_286)
            
            # SSA branch for the else part of an if statement (line 84)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'neighbour_ref' (line 88)
            neighbour_ref_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'neighbour_ref')
            # Obtaining the member 'liberties' of a type (line 88)
            liberties_289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), neighbour_ref_288, 'liberties')
            int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 51), 'int')
            # Applying the binary operator '-=' (line 88)
            result_isub_291 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 24), '-=', liberties_289, int_290)
            # Getting the type of 'neighbour_ref' (line 88)
            neighbour_ref_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'neighbour_ref')
            # Setting the type of the member 'liberties' of a type (line 88)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), neighbour_ref_292, 'liberties', result_isub_291)
            
            
            
            # Getting the type of 'neighbour_ref' (line 89)
            neighbour_ref_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'neighbour_ref')
            # Obtaining the member 'liberties' of a type (line 89)
            liberties_294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), neighbour_ref_293, 'liberties')
            int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 54), 'int')
            # Applying the binary operator '==' (line 89)
            result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 27), '==', liberties_294, int_295)
            
            # Testing the type of an if condition (line 89)
            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 24), result_eq_296)
            # Assigning a type to the variable 'if_condition_297' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'if_condition_297', if_condition_297)
            # SSA begins for if statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 90)
            # Processing the call arguments (line 90)
            # Getting the type of 'neighbour_ref' (line 90)
            neighbour_ref_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'neighbour_ref', False)
            # Processing the call keyword arguments (line 90)
            # Getting the type of 'True' (line 90)
            True_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 71), 'True', False)
            keyword_302 = True_301
            kwargs_303 = {'update': keyword_302}
            # Getting the type of 'neighbour_ref' (line 90)
            neighbour_ref_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'neighbour_ref', False)
            # Obtaining the member 'remove' of a type (line 90)
            remove_299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), neighbour_ref_298, 'remove')
            # Calling remove(args, kwargs) (line 90)
            remove_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 90, 28), remove_299, *[neighbour_ref_300], **kwargs_303)
            
            # SSA branch for the else part of an if statement (line 89)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'neighbour_ref' (line 91)
            neighbour_ref_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'neighbour_ref')
            # Obtaining the member 'liberties' of a type (line 91)
            liberties_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), neighbour_ref_305, 'liberties')
            int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'int')
            # Applying the binary operator '==' (line 91)
            result_eq_308 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '==', liberties_306, int_307)
            
            # Testing the type of an if condition (line 91)
            if_condition_309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 29), result_eq_308)
            # Assigning a type to the variable 'if_condition_309' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'if_condition_309', if_condition_309)
            # SSA begins for if statement (line 91)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 92):
            
            # Assigning a Name to a Attribute (line 92):
            # Getting the type of 'neighbour_ref' (line 92)
            neighbour_ref_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'neighbour_ref')
            # Getting the type of 'self' (line 92)
            self_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'self')
            # Obtaining the member 'board' of a type (line 92)
            board_312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), self_311, 'board')
            # Setting the type of the member 'atari' of a type (line 92)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), board_312, 'atari', neighbour_ref_310)
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
        TIMESTAMP_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'TIMESTAMP')
        int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'int')
        # Applying the binary operator '+=' (line 93)
        result_iadd_315 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 8), '+=', TIMESTAMP_313, int_314)
        # Assigning a type to the variable 'TIMESTAMP' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'TIMESTAMP', result_iadd_315)
        
        
        # Call to count_liberties(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_318 = {}
        # Getting the type of 'self' (line 94)
        self_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'count_liberties' of a type (line 94)
        count_liberties_317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_316, 'count_liberties')
        # Calling count_liberties(args, kwargs) (line 94)
        count_liberties_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), count_liberties_317, *[], **kwargs_318)
        
        
        # Call to add(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_324 = {}
        # Getting the type of 'self' (line 95)
        self_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self', False)
        # Obtaining the member 'board' of a type (line 95)
        board_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_320, 'board')
        # Obtaining the member 'zobrist' of a type (line 95)
        zobrist_322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), board_321, 'zobrist')
        # Obtaining the member 'add' of a type (line 95)
        add_323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), zobrist_322, 'add')
        # Calling add(args, kwargs) (line 95)
        add_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), add_323, *[], **kwargs_324)
        
        
        # ################# End of 'move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move'
        return stypy_return_type_326


    @norecursion
    def remove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 97)
        True_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'True')
        defaults = [True_327]
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
        REMOVESTAMP_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'REMOVESTAMP')
        int_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'int')
        # Applying the binary operator '+=' (line 99)
        result_iadd_330 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 8), '+=', REMOVESTAMP_328, int_329)
        # Assigning a type to the variable 'REMOVESTAMP' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'REMOVESTAMP', result_iadd_330)
        
        
        # Assigning a Name to a Name (line 100):
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'REMOVESTAMP' (line 100)
        REMOVESTAMP_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'REMOVESTAMP')
        # Assigning a type to the variable 'removestamp' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'removestamp', REMOVESTAMP_331)
        
        # Call to update(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 34), 'self', False)
        # Getting the type of 'EMPTY' (line 101)
        EMPTY_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'EMPTY', False)
        # Processing the call keyword arguments (line 101)
        kwargs_338 = {}
        # Getting the type of 'self' (line 101)
        self_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self', False)
        # Obtaining the member 'board' of a type (line 101)
        board_333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_332, 'board')
        # Obtaining the member 'zobrist' of a type (line 101)
        zobrist_334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), board_333, 'zobrist')
        # Obtaining the member 'update' of a type (line 101)
        update_335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), zobrist_334, 'update')
        # Calling update(args, kwargs) (line 101)
        update_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), update_335, *[self_336, EMPTY_337], **kwargs_338)
        
        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'TIMESTAMP' (line 102)
        TIMESTAMP_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 'TIMESTAMP')
        # Getting the type of 'self' (line 102)
        self_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'timestamp2' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_341, 'timestamp2', TIMESTAMP_340)
        
        # Getting the type of 'update' (line 103)
        update_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'update')
        # Testing the type of an if condition (line 103)
        if_condition_343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), update_342)
        # Assigning a type to the variable 'if_condition_343' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_343', if_condition_343)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 104):
        
        # Assigning a Name to a Attribute (line 104):
        # Getting the type of 'EMPTY' (line 104)
        EMPTY_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'EMPTY')
        # Getting the type of 'self' (line 104)
        self_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self')
        # Setting the type of the member 'color' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_345, 'color', EMPTY_344)
        
        # Call to add(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'self' (line 105)
        self_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 36), 'self', False)
        # Obtaining the member 'pos' of a type (line 105)
        pos_351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 36), self_350, 'pos')
        # Processing the call keyword arguments (line 105)
        kwargs_352 = {}
        # Getting the type of 'self' (line 105)
        self_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self', False)
        # Obtaining the member 'board' of a type (line 105)
        board_347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_346, 'board')
        # Obtaining the member 'emptyset' of a type (line 105)
        emptyset_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), board_347, 'emptyset')
        # Obtaining the member 'add' of a type (line 105)
        add_349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), emptyset_348, 'add')
        # Calling add(args, kwargs) (line 105)
        add_call_result_353 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), add_349, *[pos_351], **kwargs_352)
        
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'update' (line 110)
        update_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'update')
        # Testing the type of an if condition (line 110)
        if_condition_355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), update_354)
        # Assigning a type to the variable 'if_condition_355' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_355', if_condition_355)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 111)
        self_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'self')
        # Obtaining the member 'neighbours' of a type (line 111)
        neighbours_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 29), self_356, 'neighbours')
        # Testing if the loop is going to be iterated (line 111)
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), neighbours_357)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 12), neighbours_357):
            # Getting the type of the for loop variable (line 111)
            for_loop_var_358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), neighbours_357)
            # Assigning a type to the variable 'neighbour' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'neighbour', for_loop_var_358)
            # SSA begins for a for statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'neighbour' (line 112)
            neighbour_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'neighbour')
            # Obtaining the member 'color' of a type (line 112)
            color_360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), neighbour_359, 'color')
            # Getting the type of 'EMPTY' (line 112)
            EMPTY_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'EMPTY')
            # Applying the binary operator '!=' (line 112)
            result_ne_362 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), '!=', color_360, EMPTY_361)
            
            # Testing the type of an if condition (line 112)
            if_condition_363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), result_ne_362)
            # Assigning a type to the variable 'if_condition_363' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_363', if_condition_363)
            # SSA begins for if statement (line 112)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 113):
            
            # Assigning a Call to a Name (line 113):
            
            # Call to find(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'update' (line 113)
            update_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 51), 'update', False)
            # Processing the call keyword arguments (line 113)
            kwargs_367 = {}
            # Getting the type of 'neighbour' (line 113)
            neighbour_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 36), 'neighbour', False)
            # Obtaining the member 'find' of a type (line 113)
            find_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 36), neighbour_364, 'find')
            # Calling find(args, kwargs) (line 113)
            find_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 113, 36), find_365, *[update_366], **kwargs_367)
            
            # Assigning a type to the variable 'neighbour_ref' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'neighbour_ref', find_call_result_368)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'neighbour_ref' (line 114)
            neighbour_ref_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'neighbour_ref')
            # Obtaining the member 'pos' of a type (line 114)
            pos_370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 23), neighbour_ref_369, 'pos')
            # Getting the type of 'self' (line 114)
            self_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 44), 'self')
            # Obtaining the member 'pos' of a type (line 114)
            pos_372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 44), self_371, 'pos')
            # Applying the binary operator '!=' (line 114)
            result_ne_373 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 23), '!=', pos_370, pos_372)
            
            
            # Getting the type of 'neighbour_ref' (line 114)
            neighbour_ref_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 57), 'neighbour_ref')
            # Obtaining the member 'removestamp' of a type (line 114)
            removestamp_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 57), neighbour_ref_374, 'removestamp')
            # Getting the type of 'removestamp' (line 114)
            removestamp_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 86), 'removestamp')
            # Applying the binary operator '!=' (line 114)
            result_ne_377 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 57), '!=', removestamp_375, removestamp_376)
            
            # Applying the binary operator 'and' (line 114)
            result_and_keyword_378 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 23), 'and', result_ne_373, result_ne_377)
            
            # Testing the type of an if condition (line 114)
            if_condition_379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 20), result_and_keyword_378)
            # Assigning a type to the variable 'if_condition_379' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'if_condition_379', if_condition_379)
            # SSA begins for if statement (line 114)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 115):
            
            # Assigning a Name to a Attribute (line 115):
            # Getting the type of 'removestamp' (line 115)
            removestamp_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 52), 'removestamp')
            # Getting the type of 'neighbour_ref' (line 115)
            neighbour_ref_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'neighbour_ref')
            # Setting the type of the member 'removestamp' of a type (line 115)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 24), neighbour_ref_381, 'removestamp', removestamp_380)
            
            # Getting the type of 'neighbour_ref' (line 116)
            neighbour_ref_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'neighbour_ref')
            # Obtaining the member 'liberties' of a type (line 116)
            liberties_383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), neighbour_ref_382, 'liberties')
            int_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 51), 'int')
            # Applying the binary operator '+=' (line 116)
            result_iadd_385 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 24), '+=', liberties_383, int_384)
            # Getting the type of 'neighbour_ref' (line 116)
            neighbour_ref_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'neighbour_ref')
            # Setting the type of the member 'liberties' of a type (line 116)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), neighbour_ref_386, 'liberties', result_iadd_385)
            
            # SSA join for if statement (line 114)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 112)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 117)
        self_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'self')
        # Obtaining the member 'neighbours' of a type (line 117)
        neighbours_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), self_387, 'neighbours')
        # Testing if the loop is going to be iterated (line 117)
        # Testing the type of a for loop iterable (line 117)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 8), neighbours_388)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 117, 8), neighbours_388):
            # Getting the type of the for loop variable (line 117)
            for_loop_var_389 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 8), neighbours_388)
            # Assigning a type to the variable 'neighbour' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'neighbour', for_loop_var_389)
            # SSA begins for a for statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'neighbour' (line 118)
            neighbour_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'neighbour')
            # Obtaining the member 'color' of a type (line 118)
            color_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), neighbour_390, 'color')
            # Getting the type of 'EMPTY' (line 118)
            EMPTY_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'EMPTY')
            # Applying the binary operator '!=' (line 118)
            result_ne_393 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '!=', color_391, EMPTY_392)
            
            # Testing the type of an if condition (line 118)
            if_condition_394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 12), result_ne_393)
            # Assigning a type to the variable 'if_condition_394' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'if_condition_394', if_condition_394)
            # SSA begins for if statement (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 119):
            
            # Assigning a Call to a Name (line 119):
            
            # Call to find(...): (line 119)
            # Processing the call arguments (line 119)
            # Getting the type of 'update' (line 119)
            update_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'update', False)
            # Processing the call keyword arguments (line 119)
            kwargs_398 = {}
            # Getting the type of 'neighbour' (line 119)
            neighbour_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 32), 'neighbour', False)
            # Obtaining the member 'find' of a type (line 119)
            find_396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 32), neighbour_395, 'find')
            # Calling find(args, kwargs) (line 119)
            find_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 119, 32), find_396, *[update_397], **kwargs_398)
            
            # Assigning a type to the variable 'neighbour_ref' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'neighbour_ref', find_call_result_399)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'neighbour_ref' (line 120)
            neighbour_ref_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'neighbour_ref')
            # Obtaining the member 'pos' of a type (line 120)
            pos_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), neighbour_ref_400, 'pos')
            # Getting the type of 'reference' (line 120)
            reference_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'reference')
            # Obtaining the member 'pos' of a type (line 120)
            pos_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 40), reference_402, 'pos')
            # Applying the binary operator '==' (line 120)
            result_eq_404 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 19), '==', pos_401, pos_403)
            
            
            # Getting the type of 'neighbour' (line 120)
            neighbour_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 58), 'neighbour')
            # Obtaining the member 'timestamp2' of a type (line 120)
            timestamp2_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 58), neighbour_405, 'timestamp2')
            # Getting the type of 'TIMESTAMP' (line 120)
            TIMESTAMP_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 82), 'TIMESTAMP')
            # Applying the binary operator '!=' (line 120)
            result_ne_408 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 58), '!=', timestamp2_406, TIMESTAMP_407)
            
            # Applying the binary operator 'and' (line 120)
            result_and_keyword_409 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 19), 'and', result_eq_404, result_ne_408)
            
            # Testing the type of an if condition (line 120)
            if_condition_410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 16), result_and_keyword_409)
            # Assigning a type to the variable 'if_condition_410' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'if_condition_410', if_condition_410)
            # SSA begins for if statement (line 120)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 121)
            # Processing the call arguments (line 121)
            # Getting the type of 'reference' (line 121)
            reference_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 37), 'reference', False)
            # Getting the type of 'update' (line 121)
            update_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'update', False)
            # Processing the call keyword arguments (line 121)
            kwargs_415 = {}
            # Getting the type of 'neighbour' (line 121)
            neighbour_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'neighbour', False)
            # Obtaining the member 'remove' of a type (line 121)
            remove_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 20), neighbour_411, 'remove')
            # Calling remove(args, kwargs) (line 121)
            remove_call_result_416 = invoke(stypy.reporting.localization.Localization(__file__, 121, 20), remove_412, *[reference_413, update_414], **kwargs_415)
            
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
        stypy_return_type_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_417


    @norecursion
    def find(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 123)
        False_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'False')
        defaults = [False_418]
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
        self_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'self')
        # Obtaining the member 'reference' of a type (line 124)
        reference_420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), self_419, 'reference')
        # Assigning a type to the variable 'reference' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'reference', reference_420)
        
        
        # Getting the type of 'reference' (line 125)
        reference_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'reference')
        # Obtaining the member 'pos' of a type (line 125)
        pos_422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), reference_421, 'pos')
        # Getting the type of 'self' (line 125)
        self_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'self')
        # Obtaining the member 'pos' of a type (line 125)
        pos_424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 28), self_423, 'pos')
        # Applying the binary operator '!=' (line 125)
        result_ne_425 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), '!=', pos_422, pos_424)
        
        # Testing the type of an if condition (line 125)
        if_condition_426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_ne_425)
        # Assigning a type to the variable 'if_condition_426' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_426', if_condition_426)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to find(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'update' (line 126)
        update_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'update', False)
        # Processing the call keyword arguments (line 126)
        kwargs_430 = {}
        # Getting the type of 'reference' (line 126)
        reference_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'reference', False)
        # Obtaining the member 'find' of a type (line 126)
        find_428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), reference_427, 'find')
        # Calling find(args, kwargs) (line 126)
        find_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), find_428, *[update_429], **kwargs_430)
        
        # Assigning a type to the variable 'reference' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'reference', find_call_result_431)
        
        # Getting the type of 'update' (line 127)
        update_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'update')
        # Testing the type of an if condition (line 127)
        if_condition_433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), update_432)
        # Assigning a type to the variable 'if_condition_433' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_433', if_condition_433)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 128):
        
        # Assigning a Name to a Attribute (line 128):
        # Getting the type of 'reference' (line 128)
        reference_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 33), 'reference')
        # Getting the type of 'self' (line 128)
        self_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'self')
        # Setting the type of the member 'reference' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), self_435, 'reference', reference_434)
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'reference' (line 129)
        reference_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'reference')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', reference_436)
        
        # ################# End of 'find(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_437)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find'
        return stypy_return_type_437


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
        self_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 26), 'self', False)
        # Obtaining the member 'pos' of a type (line 132)
        pos_441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 26), self_440, 'pos')
        # Processing the call keyword arguments (line 132)
        kwargs_442 = {}
        # Getting the type of 'to_xy' (line 132)
        to_xy_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'to_xy', False)
        # Calling to_xy(args, kwargs) (line 132)
        to_xy_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 132, 20), to_xy_439, *[pos_441], **kwargs_442)
        
        # Processing the call keyword arguments (line 132)
        kwargs_444 = {}
        # Getting the type of 'repr' (line 132)
        repr_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'repr', False)
        # Calling repr(args, kwargs) (line 132)
        repr_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), repr_438, *[to_xy_call_result_443], **kwargs_444)
        
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', repr_call_result_445)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_446)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_446


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
        board_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'board')
        # Getting the type of 'self' (line 137)
        self_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member 'board' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_448, 'board', board_447)
        
        # Assigning a Call to a Attribute (line 138):
        
        # Assigning a Call to a Attribute (line 138):
        
        # Call to range(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'SIZE' (line 138)
        SIZE_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'SIZE', False)
        # Getting the type of 'SIZE' (line 138)
        SIZE_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 36), 'SIZE', False)
        # Applying the binary operator '*' (line 138)
        result_mul_452 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 29), '*', SIZE_450, SIZE_451)
        
        # Processing the call keyword arguments (line 138)
        kwargs_453 = {}
        # Getting the type of 'range' (line 138)
        range_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'range', False)
        # Calling range(args, kwargs) (line 138)
        range_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), range_449, *[result_mul_452], **kwargs_453)
        
        # Getting the type of 'self' (line 138)
        self_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
        # Setting the type of the member 'empties' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_455, 'empties', range_call_result_454)
        
        # Assigning a Call to a Attribute (line 139):
        
        # Assigning a Call to a Attribute (line 139):
        
        # Call to range(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'SIZE' (line 139)
        SIZE_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'SIZE', False)
        # Getting the type of 'SIZE' (line 139)
        SIZE_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'SIZE', False)
        # Applying the binary operator '*' (line 139)
        result_mul_459 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), '*', SIZE_457, SIZE_458)
        
        # Processing the call keyword arguments (line 139)
        kwargs_460 = {}
        # Getting the type of 'range' (line 139)
        range_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'range', False)
        # Calling range(args, kwargs) (line 139)
        range_call_result_461 = invoke(stypy.reporting.localization.Localization(__file__, 139, 25), range_456, *[result_mul_459], **kwargs_460)
        
        # Getting the type of 'self' (line 139)
        self_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member 'empty_pos' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_462, 'empty_pos', range_call_result_461)
        
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
        self_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'self', False)
        # Obtaining the member 'empties' of a type (line 142)
        empties_465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 22), self_464, 'empties')
        # Processing the call keyword arguments (line 142)
        kwargs_466 = {}
        # Getting the type of 'len' (line 142)
        len_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'len', False)
        # Calling len(args, kwargs) (line 142)
        len_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), len_463, *[empties_465], **kwargs_466)
        
        # Assigning a type to the variable 'choices' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'choices', len_call_result_467)
        
        # Getting the type of 'choices' (line 143)
        choices_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), 'choices')
        # Testing the type of an if condition (line 143)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 8), choices_468)
        # SSA begins for while statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to int(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to random(...): (line 144)
        # Processing the call keyword arguments (line 144)
        kwargs_472 = {}
        # Getting the type of 'random' (line 144)
        random_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'random', False)
        # Obtaining the member 'random' of a type (line 144)
        random_471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), random_470, 'random')
        # Calling random(args, kwargs) (line 144)
        random_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), random_471, *[], **kwargs_472)
        
        # Getting the type of 'choices' (line 144)
        choices_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 38), 'choices', False)
        # Applying the binary operator '*' (line 144)
        result_mul_475 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), '*', random_call_result_473, choices_474)
        
        # Processing the call keyword arguments (line 144)
        kwargs_476 = {}
        # Getting the type of 'int' (line 144)
        int_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'int', False)
        # Calling int(args, kwargs) (line 144)
        int_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), int_469, *[result_mul_475], **kwargs_476)
        
        # Assigning a type to the variable 'i' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'i', int_call_result_477)
        
        # Assigning a Subscript to a Name (line 145):
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 145)
        i_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 31), 'i')
        # Getting the type of 'self' (line 145)
        self_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'self')
        # Obtaining the member 'empties' of a type (line 145)
        empties_480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), self_479, 'empties')
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), empties_480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 145, 18), getitem___481, i_478)
        
        # Assigning a type to the variable 'pos' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'pos', subscript_call_result_482)
        
        
        # Call to useful(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'pos' (line 146)
        pos_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'pos', False)
        # Processing the call keyword arguments (line 146)
        kwargs_487 = {}
        # Getting the type of 'self' (line 146)
        self_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'self', False)
        # Obtaining the member 'board' of a type (line 146)
        board_484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), self_483, 'board')
        # Obtaining the member 'useful' of a type (line 146)
        useful_485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), board_484, 'useful')
        # Calling useful(args, kwargs) (line 146)
        useful_call_result_488 = invoke(stypy.reporting.localization.Localization(__file__, 146, 15), useful_485, *[pos_486], **kwargs_487)
        
        # Testing the type of an if condition (line 146)
        if_condition_489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 12), useful_call_result_488)
        # Assigning a type to the variable 'if_condition_489' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'if_condition_489', if_condition_489)
        # SSA begins for if statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'pos' (line 147)
        pos_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'pos')
        # Assigning a type to the variable 'stypy_return_type' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'stypy_return_type', pos_490)
        # SSA join for if statement (line 146)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'choices' (line 148)
        choices_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'choices')
        int_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'int')
        # Applying the binary operator '-=' (line 148)
        result_isub_493 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 12), '-=', choices_491, int_492)
        # Assigning a type to the variable 'choices' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'choices', result_isub_493)
        
        
        # Call to set(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'i' (line 149)
        i_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'i', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'choices' (line 149)
        choices_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'choices', False)
        # Getting the type of 'self' (line 149)
        self_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'self', False)
        # Obtaining the member 'empties' of a type (line 149)
        empties_499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), self_498, 'empties')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), empties_499, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), getitem___500, choices_497)
        
        # Processing the call keyword arguments (line 149)
        kwargs_502 = {}
        # Getting the type of 'self' (line 149)
        self_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self', False)
        # Obtaining the member 'set' of a type (line 149)
        set_495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_494, 'set')
        # Calling set(args, kwargs) (line 149)
        set_call_result_503 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), set_495, *[i_496, subscript_call_result_501], **kwargs_502)
        
        
        # Call to set(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'choices' (line 150)
        choices_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'choices', False)
        # Getting the type of 'pos' (line 150)
        pos_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'pos', False)
        # Processing the call keyword arguments (line 150)
        kwargs_508 = {}
        # Getting the type of 'self' (line 150)
        self_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'self', False)
        # Obtaining the member 'set' of a type (line 150)
        set_505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), self_504, 'set')
        # Calling set(args, kwargs) (line 150)
        set_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), set_505, *[choices_506, pos_507], **kwargs_508)
        
        # SSA join for while statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'PASS' (line 151)
        PASS_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'PASS')
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', PASS_510)
        
        # ################# End of 'random_choice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'random_choice' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'random_choice'
        return stypy_return_type_511


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
        self_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'self', False)
        # Obtaining the member 'empties' of a type (line 154)
        empties_514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 34), self_513, 'empties')
        # Processing the call keyword arguments (line 154)
        kwargs_515 = {}
        # Getting the type of 'len' (line 154)
        len_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'len', False)
        # Calling len(args, kwargs) (line 154)
        len_call_result_516 = invoke(stypy.reporting.localization.Localization(__file__, 154, 30), len_512, *[empties_514], **kwargs_515)
        
        # Getting the type of 'self' (line 154)
        self_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Obtaining the member 'empty_pos' of a type (line 154)
        empty_pos_518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_517, 'empty_pos')
        # Getting the type of 'pos' (line 154)
        pos_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'pos')
        # Storing an element on a container (line 154)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 8), empty_pos_518, (pos_519, len_call_result_516))
        
        # Call to append(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'pos' (line 155)
        pos_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'pos', False)
        # Processing the call keyword arguments (line 155)
        kwargs_524 = {}
        # Getting the type of 'self' (line 155)
        self_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'empties' of a type (line 155)
        empties_521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_520, 'empties')
        # Obtaining the member 'append' of a type (line 155)
        append_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), empties_521, 'append')
        # Calling append(args, kwargs) (line 155)
        append_call_result_525 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), append_522, *[pos_523], **kwargs_524)
        
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_526


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
        pos_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'pos', False)
        # Getting the type of 'self' (line 158)
        self_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'self', False)
        # Obtaining the member 'empty_pos' of a type (line 158)
        empty_pos_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 17), self_530, 'empty_pos')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 17), empty_pos_531, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 158, 17), getitem___532, pos_529)
        
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'self', False)
        # Obtaining the member 'empties' of a type (line 158)
        empties_536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 55), self_535, 'empties')
        # Processing the call keyword arguments (line 158)
        kwargs_537 = {}
        # Getting the type of 'len' (line 158)
        len_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 51), 'len', False)
        # Calling len(args, kwargs) (line 158)
        len_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 158, 51), len_534, *[empties_536], **kwargs_537)
        
        int_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 71), 'int')
        # Applying the binary operator '-' (line 158)
        result_sub_540 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 51), '-', len_call_result_538, int_539)
        
        # Getting the type of 'self' (line 158)
        self_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'self', False)
        # Obtaining the member 'empties' of a type (line 158)
        empties_542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 38), self_541, 'empties')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 38), empties_542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 158, 38), getitem___543, result_sub_540)
        
        # Processing the call keyword arguments (line 158)
        kwargs_545 = {}
        # Getting the type of 'self' (line 158)
        self_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self', False)
        # Obtaining the member 'set' of a type (line 158)
        set_528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_527, 'set')
        # Calling set(args, kwargs) (line 158)
        set_call_result_546 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), set_528, *[subscript_call_result_533, subscript_call_result_544], **kwargs_545)
        
        
        # Call to pop(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_550 = {}
        # Getting the type of 'self' (line 159)
        self_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self', False)
        # Obtaining the member 'empties' of a type (line 159)
        empties_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_547, 'empties')
        # Obtaining the member 'pop' of a type (line 159)
        pop_549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), empties_548, 'pop')
        # Calling pop(args, kwargs) (line 159)
        pop_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), pop_549, *[], **kwargs_550)
        
        
        # ################# End of 'remove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_552


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
        pos_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'pos')
        # Getting the type of 'self' (line 162)
        self_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Obtaining the member 'empties' of a type (line 162)
        empties_555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_554, 'empties')
        # Getting the type of 'i' (line 162)
        i_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'i')
        # Storing an element on a container (line 162)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), empties_555, (i_556, pos_553))
        
        # Assigning a Name to a Subscript (line 163):
        
        # Assigning a Name to a Subscript (line 163):
        # Getting the type of 'i' (line 163)
        i_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'i')
        # Getting the type of 'self' (line 163)
        self_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Obtaining the member 'empty_pos' of a type (line 163)
        empty_pos_559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_558, 'empty_pos')
        # Getting the type of 'pos' (line 163)
        pos_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'pos')
        # Storing an element on a container (line 163)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 8), empty_pos_559, (pos_560, i_557))
        
        # ################# End of 'set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set'
        return stypy_return_type_561


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
        board_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'board')
        # Getting the type of 'self' (line 168)
        self_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member 'board' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_563, 'board', board_562)
        
        # Assigning a Call to a Attribute (line 169):
        
        # Assigning a Call to a Attribute (line 169):
        
        # Call to set(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_565 = {}
        # Getting the type of 'set' (line 169)
        set_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'set', False)
        # Calling set(args, kwargs) (line 169)
        set_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), set_564, *[], **kwargs_565)
        
        # Getting the type of 'self' (line 169)
        self_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member 'hash_set' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_567, 'hash_set', set_call_result_566)
        
        # Assigning a Num to a Attribute (line 170):
        
        # Assigning a Num to a Attribute (line 170):
        int_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 20), 'int')
        # Getting the type of 'self' (line 170)
        self_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'hash' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_569, 'hash', int_568)
        
        # Getting the type of 'self' (line 171)
        self_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'self')
        # Obtaining the member 'board' of a type (line 171)
        board_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), self_570, 'board')
        # Obtaining the member 'squares' of a type (line 171)
        squares_572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), board_571, 'squares')
        # Testing if the loop is going to be iterated (line 171)
        # Testing the type of a for loop iterable (line 171)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 8), squares_572)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 171, 8), squares_572):
            # Getting the type of the for loop variable (line 171)
            for_loop_var_573 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 8), squares_572)
            # Assigning a type to the variable 'square' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'square', for_loop_var_573)
            # SSA begins for a for statement (line 171)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'self' (line 172)
            self_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'self')
            # Obtaining the member 'hash' of a type (line 172)
            hash_575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), self_574, 'hash')
            
            # Obtaining the type of the subscript
            # Getting the type of 'EMPTY' (line 172)
            EMPTY_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 48), 'EMPTY')
            # Getting the type of 'square' (line 172)
            square_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'square')
            # Obtaining the member 'zobrist_strings' of a type (line 172)
            zobrist_strings_578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), square_577, 'zobrist_strings')
            # Obtaining the member '__getitem__' of a type (line 172)
            getitem___579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), zobrist_strings_578, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 172)
            subscript_call_result_580 = invoke(stypy.reporting.localization.Localization(__file__, 172, 25), getitem___579, EMPTY_576)
            
            # Applying the binary operator '^=' (line 172)
            result_ixor_581 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 12), '^=', hash_575, subscript_call_result_580)
            # Getting the type of 'self' (line 172)
            self_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'self')
            # Setting the type of the member 'hash' of a type (line 172)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), self_582, 'hash', result_ixor_581)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to clear(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_586 = {}
        # Getting the type of 'self' (line 173)
        self_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'hash_set' of a type (line 173)
        hash_set_584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_583, 'hash_set')
        # Obtaining the member 'clear' of a type (line 173)
        clear_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), hash_set_584, 'clear')
        # Calling clear(args, kwargs) (line 173)
        clear_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), clear_585, *[], **kwargs_586)
        
        
        # Call to add(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'self' (line 174)
        self_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'self', False)
        # Obtaining the member 'hash' of a type (line 174)
        hash_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), self_591, 'hash')
        # Processing the call keyword arguments (line 174)
        kwargs_593 = {}
        # Getting the type of 'self' (line 174)
        self_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self', False)
        # Obtaining the member 'hash_set' of a type (line 174)
        hash_set_589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_588, 'hash_set')
        # Obtaining the member 'add' of a type (line 174)
        add_590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), hash_set_589, 'add')
        # Calling add(args, kwargs) (line 174)
        add_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), add_590, *[hash_592], **kwargs_593)
        
        
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
        self_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Obtaining the member 'hash' of a type (line 177)
        hash_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_595, 'hash')
        
        # Obtaining the type of the subscript
        # Getting the type of 'square' (line 177)
        square_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 44), 'square')
        # Obtaining the member 'color' of a type (line 177)
        color_598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 44), square_597, 'color')
        # Getting the type of 'square' (line 177)
        square_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'square')
        # Obtaining the member 'zobrist_strings' of a type (line 177)
        zobrist_strings_600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), square_599, 'zobrist_strings')
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), zobrist_strings_600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 177, 21), getitem___601, color_598)
        
        # Applying the binary operator '^=' (line 177)
        result_ixor_603 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 8), '^=', hash_596, subscript_call_result_602)
        # Getting the type of 'self' (line 177)
        self_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member 'hash' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_604, 'hash', result_ixor_603)
        
        
        # Getting the type of 'self' (line 178)
        self_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Obtaining the member 'hash' of a type (line 178)
        hash_606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_605, 'hash')
        
        # Obtaining the type of the subscript
        # Getting the type of 'color' (line 178)
        color_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'color')
        # Getting the type of 'square' (line 178)
        square_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'square')
        # Obtaining the member 'zobrist_strings' of a type (line 178)
        zobrist_strings_609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 21), square_608, 'zobrist_strings')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 21), zobrist_strings_609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 178, 21), getitem___610, color_607)
        
        # Applying the binary operator '^=' (line 178)
        result_ixor_612 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 8), '^=', hash_606, subscript_call_result_611)
        # Getting the type of 'self' (line 178)
        self_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Setting the type of the member 'hash' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_613, 'hash', result_ixor_612)
        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_614


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
        self_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'self', False)
        # Obtaining the member 'hash' of a type (line 181)
        hash_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 26), self_618, 'hash')
        # Processing the call keyword arguments (line 181)
        kwargs_620 = {}
        # Getting the type of 'self' (line 181)
        self_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member 'hash_set' of a type (line 181)
        hash_set_616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_615, 'hash_set')
        # Obtaining the member 'add' of a type (line 181)
        add_617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), hash_set_616, 'add')
        # Calling add(args, kwargs) (line 181)
        add_call_result_621 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), add_617, *[hash_619], **kwargs_620)
        
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_622


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
        self_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'self')
        # Obtaining the member 'hash' of a type (line 184)
        hash_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 15), self_623, 'hash')
        # Getting the type of 'self' (line 184)
        self_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'self')
        # Obtaining the member 'hash_set' of a type (line 184)
        hash_set_626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), self_625, 'hash_set')
        # Applying the binary operator 'in' (line 184)
        result_contains_627 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), 'in', hash_624, hash_set_626)
        
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', result_contains_627)
        
        # ################# End of 'dupe(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dupe' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_628)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dupe'
        return stypy_return_type_628


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
        SIZE_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 59), 'SIZE', False)
        # Getting the type of 'SIZE' (line 189)
        SIZE_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 66), 'SIZE', False)
        # Applying the binary operator '*' (line 189)
        result_mul_637 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 59), '*', SIZE_635, SIZE_636)
        
        # Processing the call keyword arguments (line 189)
        kwargs_638 = {}
        # Getting the type of 'range' (line 189)
        range_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 53), 'range', False)
        # Calling range(args, kwargs) (line 189)
        range_call_result_639 = invoke(stypy.reporting.localization.Localization(__file__, 189, 53), range_634, *[result_mul_637], **kwargs_638)
        
        comprehension_640 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 24), range_call_result_639)
        # Assigning a type to the variable 'pos' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'pos', comprehension_640)
        
        # Call to Square(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 31), 'self', False)
        # Getting the type of 'pos' (line 189)
        pos_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'pos', False)
        # Processing the call keyword arguments (line 189)
        kwargs_632 = {}
        # Getting the type of 'Square' (line 189)
        Square_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'Square', False)
        # Calling Square(args, kwargs) (line 189)
        Square_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), Square_629, *[self_630, pos_631], **kwargs_632)
        
        list_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 24), list_641, Square_call_result_633)
        # Getting the type of 'self' (line 189)
        self_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self')
        # Setting the type of the member 'squares' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_642, 'squares', list_641)
        
        # Getting the type of 'self' (line 190)
        self_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'self')
        # Obtaining the member 'squares' of a type (line 190)
        squares_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 22), self_643, 'squares')
        # Testing if the loop is going to be iterated (line 190)
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), squares_644)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 190, 8), squares_644):
            # Getting the type of the for loop variable (line 190)
            for_loop_var_645 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), squares_644)
            # Assigning a type to the variable 'square' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'square', for_loop_var_645)
            # SSA begins for a for statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to set_neighbours(...): (line 191)
            # Processing the call keyword arguments (line 191)
            kwargs_648 = {}
            # Getting the type of 'square' (line 191)
            square_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'square', False)
            # Obtaining the member 'set_neighbours' of a type (line 191)
            set_neighbours_647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), square_646, 'set_neighbours')
            # Calling set_neighbours(args, kwargs) (line 191)
            set_neighbours_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), set_neighbours_647, *[], **kwargs_648)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to reset(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_652 = {}
        # Getting the type of 'self' (line 192)
        self_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self', False)
        # Obtaining the member 'reset' of a type (line 192)
        reset_651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_650, 'reset')
        # Calling reset(args, kwargs) (line 192)
        reset_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), reset_651, *[], **kwargs_652)
        
        
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
        self_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'self')
        # Obtaining the member 'squares' of a type (line 195)
        squares_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 22), self_654, 'squares')
        # Testing if the loop is going to be iterated (line 195)
        # Testing the type of a for loop iterable (line 195)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 8), squares_655)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 195, 8), squares_655):
            # Getting the type of the for loop variable (line 195)
            for_loop_var_656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 8), squares_655)
            # Assigning a type to the variable 'square' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'square', for_loop_var_656)
            # SSA begins for a for statement (line 195)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Attribute (line 196):
            
            # Assigning a Name to a Attribute (line 196):
            # Getting the type of 'EMPTY' (line 196)
            EMPTY_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'EMPTY')
            # Getting the type of 'square' (line 196)
            square_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'square')
            # Setting the type of the member 'color' of a type (line 196)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), square_658, 'color', EMPTY_657)
            
            # Assigning a Name to a Attribute (line 197):
            
            # Assigning a Name to a Attribute (line 197):
            # Getting the type of 'False' (line 197)
            False_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'False')
            # Getting the type of 'square' (line 197)
            square_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'square')
            # Setting the type of the member 'used' of a type (line 197)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), square_660, 'used', False_659)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Attribute (line 198):
        
        # Assigning a Call to a Attribute (line 198):
        
        # Call to EmptySet(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'self' (line 198)
        self_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 33), 'self', False)
        # Processing the call keyword arguments (line 198)
        kwargs_663 = {}
        # Getting the type of 'EmptySet' (line 198)
        EmptySet_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'EmptySet', False)
        # Calling EmptySet(args, kwargs) (line 198)
        EmptySet_call_result_664 = invoke(stypy.reporting.localization.Localization(__file__, 198, 24), EmptySet_661, *[self_662], **kwargs_663)
        
        # Getting the type of 'self' (line 198)
        self_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member 'emptyset' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_665, 'emptyset', EmptySet_call_result_664)
        
        # Assigning a Call to a Attribute (line 199):
        
        # Assigning a Call to a Attribute (line 199):
        
        # Call to ZobristHash(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'self' (line 199)
        self_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'self', False)
        # Processing the call keyword arguments (line 199)
        kwargs_668 = {}
        # Getting the type of 'ZobristHash' (line 199)
        ZobristHash_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'ZobristHash', False)
        # Calling ZobristHash(args, kwargs) (line 199)
        ZobristHash_call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 199, 23), ZobristHash_666, *[self_667], **kwargs_668)
        
        # Getting the type of 'self' (line 199)
        self_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self')
        # Setting the type of the member 'zobrist' of a type (line 199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_670, 'zobrist', ZobristHash_call_result_669)
        
        # Assigning a Name to a Attribute (line 200):
        
        # Assigning a Name to a Attribute (line 200):
        # Getting the type of 'BLACK' (line 200)
        BLACK_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'BLACK')
        # Getting the type of 'self' (line 200)
        self_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member 'color' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_672, 'color', BLACK_671)
        
        # Assigning a Name to a Attribute (line 201):
        
        # Assigning a Name to a Attribute (line 201):
        # Getting the type of 'False' (line 201)
        False_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'False')
        # Getting the type of 'self' (line 201)
        self_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self')
        # Setting the type of the member 'finished' of a type (line 201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_674, 'finished', False_673)
        
        # Assigning a Num to a Attribute (line 202):
        
        # Assigning a Num to a Attribute (line 202):
        int_675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 24), 'int')
        # Getting the type of 'self' (line 202)
        self_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member 'lastmove' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_676, 'lastmove', int_675)
        
        # Assigning a List to a Attribute (line 203):
        
        # Assigning a List to a Attribute (line 203):
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        
        # Getting the type of 'self' (line 203)
        self_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self')
        # Setting the type of the member 'history' of a type (line 203)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_678, 'history', list_677)
        
        # Assigning a Name to a Attribute (line 204):
        
        # Assigning a Name to a Attribute (line 204):
        # Getting the type of 'None' (line 204)
        None_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'None')
        # Getting the type of 'self' (line 204)
        self_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member 'atari' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_680, 'atari', None_679)
        
        # Assigning a Num to a Attribute (line 205):
        
        # Assigning a Num to a Attribute (line 205):
        int_681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'int')
        # Getting the type of 'self' (line 205)
        self_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'white_dead' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_682, 'white_dead', int_681)
        
        # Assigning a Num to a Attribute (line 206):
        
        # Assigning a Num to a Attribute (line 206):
        int_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 26), 'int')
        # Getting the type of 'self' (line 206)
        self_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'black_dead' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_684, 'black_dead', int_683)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_685


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
        pos_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'pos')
        # Getting the type of 'self' (line 209)
        self_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'self')
        # Obtaining the member 'squares' of a type (line 209)
        squares_688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 17), self_687, 'squares')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 17), squares_688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_690 = invoke(stypy.reporting.localization.Localization(__file__, 209, 17), getitem___689, pos_686)
        
        # Assigning a type to the variable 'square' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'square', subscript_call_result_690)
        
        
        # Getting the type of 'pos' (line 210)
        pos_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'pos')
        # Getting the type of 'PASS' (line 210)
        PASS_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 'PASS')
        # Applying the binary operator '!=' (line 210)
        result_ne_693 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), '!=', pos_691, PASS_692)
        
        # Testing the type of an if condition (line 210)
        if_condition_694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_ne_693)
        # Assigning a type to the variable 'if_condition_694' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_694', if_condition_694)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to move(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'self' (line 211)
        self_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'self', False)
        # Obtaining the member 'color' of a type (line 211)
        color_698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), self_697, 'color')
        # Processing the call keyword arguments (line 211)
        kwargs_699 = {}
        # Getting the type of 'square' (line 211)
        square_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'square', False)
        # Obtaining the member 'move' of a type (line 211)
        move_696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), square_695, 'move')
        # Calling move(args, kwargs) (line 211)
        move_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), move_696, *[color_698], **kwargs_699)
        
        
        # Call to remove(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'square' (line 212)
        square_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'square', False)
        # Obtaining the member 'pos' of a type (line 212)
        pos_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 33), square_704, 'pos')
        # Processing the call keyword arguments (line 212)
        kwargs_706 = {}
        # Getting the type of 'self' (line 212)
        self_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'self', False)
        # Obtaining the member 'emptyset' of a type (line 212)
        emptyset_702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), self_701, 'emptyset')
        # Obtaining the member 'remove' of a type (line 212)
        remove_703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), emptyset_702, 'remove')
        # Calling remove(args, kwargs) (line 212)
        remove_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), remove_703, *[pos_705], **kwargs_706)
        
        # SSA branch for the else part of an if statement (line 210)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 213)
        self_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'self')
        # Obtaining the member 'lastmove' of a type (line 213)
        lastmove_709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 13), self_708, 'lastmove')
        # Getting the type of 'PASS' (line 213)
        PASS_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'PASS')
        # Applying the binary operator '==' (line 213)
        result_eq_711 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 13), '==', lastmove_709, PASS_710)
        
        # Testing the type of an if condition (line 213)
        if_condition_712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 13), result_eq_711)
        # Assigning a type to the variable 'if_condition_712' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'if_condition_712', if_condition_712)
        # SSA begins for if statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 214):
        
        # Assigning a Name to a Attribute (line 214):
        # Getting the type of 'True' (line 214)
        True_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'True')
        # Getting the type of 'self' (line 214)
        self_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'self')
        # Setting the type of the member 'finished' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), self_714, 'finished', True_713)
        # SSA join for if statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 215)
        self_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'self')
        # Obtaining the member 'color' of a type (line 215)
        color_716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 11), self_715, 'color')
        # Getting the type of 'BLACK' (line 215)
        BLACK_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 25), 'BLACK')
        # Applying the binary operator '==' (line 215)
        result_eq_718 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), '==', color_716, BLACK_717)
        
        # Testing the type of an if condition (line 215)
        if_condition_719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_eq_718)
        # Assigning a type to the variable 'if_condition_719' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_719', if_condition_719)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 216):
        
        # Assigning a Name to a Attribute (line 216):
        # Getting the type of 'WHITE' (line 216)
        WHITE_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'WHITE')
        # Getting the type of 'self' (line 216)
        self_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'self')
        # Setting the type of the member 'color' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), self_721, 'color', WHITE_720)
        # SSA branch for the else part of an if statement (line 215)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 218):
        
        # Assigning a Name to a Attribute (line 218):
        # Getting the type of 'BLACK' (line 218)
        BLACK_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'BLACK')
        # Getting the type of 'self' (line 218)
        self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self')
        # Setting the type of the member 'color' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_723, 'color', BLACK_722)
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 219):
        
        # Assigning a Name to a Attribute (line 219):
        # Getting the type of 'pos' (line 219)
        pos_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'pos')
        # Getting the type of 'self' (line 219)
        self_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self')
        # Setting the type of the member 'lastmove' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_725, 'lastmove', pos_724)
        
        # Call to append(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'pos' (line 220)
        pos_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'pos', False)
        # Processing the call keyword arguments (line 220)
        kwargs_730 = {}
        # Getting the type of 'self' (line 220)
        self_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self', False)
        # Obtaining the member 'history' of a type (line 220)
        history_727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_726, 'history')
        # Obtaining the member 'append' of a type (line 220)
        append_728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), history_727, 'append')
        # Calling append(args, kwargs) (line 220)
        append_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), append_728, *[pos_729], **kwargs_730)
        
        
        # ################# End of 'move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move'
        return stypy_return_type_732


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
        kwargs_736 = {}
        # Getting the type of 'self' (line 223)
        self_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'self', False)
        # Obtaining the member 'emptyset' of a type (line 223)
        emptyset_734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), self_733, 'emptyset')
        # Obtaining the member 'random_choice' of a type (line 223)
        random_choice_735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), emptyset_734, 'random_choice')
        # Calling random_choice(args, kwargs) (line 223)
        random_choice_call_result_737 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), random_choice_735, *[], **kwargs_736)
        
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', random_choice_call_result_737)
        
        # ################# End of 'random_move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'random_move' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'random_move'
        return stypy_return_type_738


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
        square_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'square')
        # Obtaining the member 'used' of a type (line 226)
        used_740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 15), square_739, 'used')
        # Applying the 'not' unary operator (line 226)
        result_not__741 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 11), 'not', used_740)
        
        # Testing the type of an if condition (line 226)
        if_condition_742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), result_not__741)
        # Assigning a type to the variable 'if_condition_742' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_742', if_condition_742)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'square' (line 227)
        square_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'square')
        # Obtaining the member 'neighbours' of a type (line 227)
        neighbours_744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 29), square_743, 'neighbours')
        # Testing if the loop is going to be iterated (line 227)
        # Testing the type of a for loop iterable (line 227)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 227, 12), neighbours_744)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 227, 12), neighbours_744):
            # Getting the type of the for loop variable (line 227)
            for_loop_var_745 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 227, 12), neighbours_744)
            # Assigning a type to the variable 'neighbour' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'neighbour', for_loop_var_745)
            # SSA begins for a for statement (line 227)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'neighbour' (line 228)
            neighbour_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'neighbour')
            # Obtaining the member 'color' of a type (line 228)
            color_747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), neighbour_746, 'color')
            # Getting the type of 'EMPTY' (line 228)
            EMPTY_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'EMPTY')
            # Applying the binary operator '==' (line 228)
            result_eq_749 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 19), '==', color_747, EMPTY_748)
            
            # Testing the type of an if condition (line 228)
            if_condition_750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 16), result_eq_749)
            # Assigning a type to the variable 'if_condition_750' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'if_condition_750', if_condition_750)
            # SSA begins for if statement (line 228)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 229)
            True_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'stypy_return_type', True_751)
            # SSA join for if statement (line 228)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 230)
        False_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', False_752)
        
        # ################# End of 'useful_fast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'useful_fast' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'useful_fast'
        return stypy_return_type_753


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
        TIMESTAMP_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'TIMESTAMP')
        int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'int')
        # Applying the binary operator '+=' (line 234)
        result_iadd_756 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 8), '+=', TIMESTAMP_754, int_755)
        # Assigning a type to the variable 'TIMESTAMP' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'TIMESTAMP', result_iadd_756)
        
        
        # Assigning a Subscript to a Name (line 235):
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 235)
        pos_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'pos')
        # Getting the type of 'self' (line 235)
        self_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'self')
        # Obtaining the member 'squares' of a type (line 235)
        squares_759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), self_758, 'squares')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), squares_759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_761 = invoke(stypy.reporting.localization.Localization(__file__, 235, 17), getitem___760, pos_757)
        
        # Assigning a type to the variable 'square' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'square', subscript_call_result_761)
        
        
        # Call to useful_fast(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'square' (line 236)
        square_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'square', False)
        # Processing the call keyword arguments (line 236)
        kwargs_765 = {}
        # Getting the type of 'self' (line 236)
        self_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'self', False)
        # Obtaining the member 'useful_fast' of a type (line 236)
        useful_fast_763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 11), self_762, 'useful_fast')
        # Calling useful_fast(args, kwargs) (line 236)
        useful_fast_call_result_766 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), useful_fast_763, *[square_764], **kwargs_765)
        
        # Testing the type of an if condition (line 236)
        if_condition_767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), useful_fast_call_result_766)
        # Assigning a type to the variable 'if_condition_767' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_767', if_condition_767)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 237)
        True_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'stypy_return_type', True_768)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 238):
        
        # Assigning a Attribute to a Name (line 238):
        # Getting the type of 'self' (line 238)
        self_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 19), 'self')
        # Obtaining the member 'zobrist' of a type (line 238)
        zobrist_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 19), self_769, 'zobrist')
        # Obtaining the member 'hash' of a type (line 238)
        hash_771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 19), zobrist_770, 'hash')
        # Assigning a type to the variable 'old_hash' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'old_hash', hash_771)
        
        # Call to update(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'square' (line 239)
        square_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'square', False)
        # Getting the type of 'self' (line 239)
        self_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'self', False)
        # Obtaining the member 'color' of a type (line 239)
        color_777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 36), self_776, 'color')
        # Processing the call keyword arguments (line 239)
        kwargs_778 = {}
        # Getting the type of 'self' (line 239)
        self_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self', False)
        # Obtaining the member 'zobrist' of a type (line 239)
        zobrist_773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_772, 'zobrist')
        # Obtaining the member 'update' of a type (line 239)
        update_774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), zobrist_773, 'update')
        # Calling update(args, kwargs) (line 239)
        update_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), update_774, *[square_775, color_777], **kwargs_778)
        
        
        # Multiple assignment of 5 elements.
        
        # Assigning a Num to a Name (line 240):
        int_780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 74), 'int')
        # Assigning a type to the variable 'weak_neighs' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 60), 'weak_neighs', int_780)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'weak_neighs' (line 240)
        weak_neighs_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 60), 'weak_neighs')
        # Assigning a type to the variable 'strong_neighs' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'strong_neighs', weak_neighs_781)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'strong_neighs' (line 240)
        strong_neighs_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'strong_neighs')
        # Assigning a type to the variable 'weak_opps' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'weak_opps', strong_neighs_782)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'weak_opps' (line 240)
        weak_opps_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'weak_opps')
        # Assigning a type to the variable 'strong_opps' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'strong_opps', weak_opps_783)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'strong_opps' (line 240)
        strong_opps_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'strong_opps')
        # Assigning a type to the variable 'empties' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'empties', strong_opps_784)
        
        # Getting the type of 'square' (line 241)
        square_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'square')
        # Obtaining the member 'neighbours' of a type (line 241)
        neighbours_786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 25), square_785, 'neighbours')
        # Testing if the loop is going to be iterated (line 241)
        # Testing the type of a for loop iterable (line 241)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 8), neighbours_786)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 241, 8), neighbours_786):
            # Getting the type of the for loop variable (line 241)
            for_loop_var_787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 8), neighbours_786)
            # Assigning a type to the variable 'neighbour' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'neighbour', for_loop_var_787)
            # SSA begins for a for statement (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'neighbour' (line 242)
            neighbour_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'neighbour')
            # Obtaining the member 'color' of a type (line 242)
            color_789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 15), neighbour_788, 'color')
            # Getting the type of 'EMPTY' (line 242)
            EMPTY_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'EMPTY')
            # Applying the binary operator '==' (line 242)
            result_eq_791 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 15), '==', color_789, EMPTY_790)
            
            # Testing the type of an if condition (line 242)
            if_condition_792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 12), result_eq_791)
            # Assigning a type to the variable 'if_condition_792' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'if_condition_792', if_condition_792)
            # SSA begins for if statement (line 242)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'empties' (line 243)
            empties_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'empties')
            int_794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 27), 'int')
            # Applying the binary operator '+=' (line 243)
            result_iadd_795 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 16), '+=', empties_793, int_794)
            # Assigning a type to the variable 'empties' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'empties', result_iadd_795)
            
            # SSA branch for the else part of an if statement (line 242)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 245):
            
            # Assigning a Call to a Name (line 245):
            
            # Call to find(...): (line 245)
            # Processing the call keyword arguments (line 245)
            kwargs_798 = {}
            # Getting the type of 'neighbour' (line 245)
            neighbour_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 32), 'neighbour', False)
            # Obtaining the member 'find' of a type (line 245)
            find_797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 32), neighbour_796, 'find')
            # Calling find(args, kwargs) (line 245)
            find_call_result_799 = invoke(stypy.reporting.localization.Localization(__file__, 245, 32), find_797, *[], **kwargs_798)
            
            # Assigning a type to the variable 'neighbour_ref' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'neighbour_ref', find_call_result_799)
            
            
            # Getting the type of 'neighbour_ref' (line 246)
            neighbour_ref_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'neighbour_ref')
            # Obtaining the member 'timestamp' of a type (line 246)
            timestamp_801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), neighbour_ref_800, 'timestamp')
            # Getting the type of 'TIMESTAMP' (line 246)
            TIMESTAMP_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 46), 'TIMESTAMP')
            # Applying the binary operator '!=' (line 246)
            result_ne_803 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 19), '!=', timestamp_801, TIMESTAMP_802)
            
            # Testing the type of an if condition (line 246)
            if_condition_804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 16), result_ne_803)
            # Assigning a type to the variable 'if_condition_804' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'if_condition_804', if_condition_804)
            # SSA begins for if statement (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 247):
            
            # Assigning a Name to a Attribute (line 247):
            # Getting the type of 'TIMESTAMP' (line 247)
            TIMESTAMP_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 46), 'TIMESTAMP')
            # Getting the type of 'neighbour_ref' (line 247)
            neighbour_ref_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'neighbour_ref')
            # Setting the type of the member 'timestamp' of a type (line 247)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), neighbour_ref_806, 'timestamp', TIMESTAMP_805)
            
            # Assigning a Compare to a Name (line 248):
            
            # Assigning a Compare to a Name (line 248):
            
            # Getting the type of 'neighbour_ref' (line 248)
            neighbour_ref_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'neighbour_ref')
            # Obtaining the member 'liberties' of a type (line 248)
            liberties_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 28), neighbour_ref_807, 'liberties')
            int_809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 55), 'int')
            # Applying the binary operator '==' (line 248)
            result_eq_810 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 28), '==', liberties_808, int_809)
            
            # Assigning a type to the variable 'weak' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'weak', result_eq_810)
            
            
            # Getting the type of 'neighbour' (line 249)
            neighbour_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 23), 'neighbour')
            # Obtaining the member 'color' of a type (line 249)
            color_812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 23), neighbour_811, 'color')
            # Getting the type of 'self' (line 249)
            self_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'self')
            # Obtaining the member 'color' of a type (line 249)
            color_814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), self_813, 'color')
            # Applying the binary operator '==' (line 249)
            result_eq_815 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 23), '==', color_812, color_814)
            
            # Testing the type of an if condition (line 249)
            if_condition_816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 20), result_eq_815)
            # Assigning a type to the variable 'if_condition_816' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'if_condition_816', if_condition_816)
            # SSA begins for if statement (line 249)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'weak' (line 250)
            weak_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'weak')
            # Testing the type of an if condition (line 250)
            if_condition_818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 24), weak_817)
            # Assigning a type to the variable 'if_condition_818' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'if_condition_818', if_condition_818)
            # SSA begins for if statement (line 250)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'weak_neighs' (line 251)
            weak_neighs_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'weak_neighs')
            int_820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 43), 'int')
            # Applying the binary operator '+=' (line 251)
            result_iadd_821 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 28), '+=', weak_neighs_819, int_820)
            # Assigning a type to the variable 'weak_neighs' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'weak_neighs', result_iadd_821)
            
            # SSA branch for the else part of an if statement (line 250)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'strong_neighs' (line 253)
            strong_neighs_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs')
            int_823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'int')
            # Applying the binary operator '+=' (line 253)
            result_iadd_824 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 28), '+=', strong_neighs_822, int_823)
            # Assigning a type to the variable 'strong_neighs' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'strong_neighs', result_iadd_824)
            
            # SSA join for if statement (line 250)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 249)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'weak' (line 255)
            weak_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'weak')
            # Testing the type of an if condition (line 255)
            if_condition_826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 24), weak_825)
            # Assigning a type to the variable 'if_condition_826' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'if_condition_826', if_condition_826)
            # SSA begins for if statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'weak_opps' (line 256)
            weak_opps_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps')
            int_828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 41), 'int')
            # Applying the binary operator '+=' (line 256)
            result_iadd_829 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 28), '+=', weak_opps_827, int_828)
            # Assigning a type to the variable 'weak_opps' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'weak_opps', result_iadd_829)
            
            
            # Call to remove(...): (line 257)
            # Processing the call arguments (line 257)
            # Getting the type of 'neighbour_ref' (line 257)
            neighbour_ref_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 49), 'neighbour_ref', False)
            # Processing the call keyword arguments (line 257)
            # Getting the type of 'False' (line 257)
            False_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 71), 'False', False)
            keyword_834 = False_833
            kwargs_835 = {'update': keyword_834}
            # Getting the type of 'neighbour_ref' (line 257)
            neighbour_ref_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'neighbour_ref', False)
            # Obtaining the member 'remove' of a type (line 257)
            remove_831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 28), neighbour_ref_830, 'remove')
            # Calling remove(args, kwargs) (line 257)
            remove_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 257, 28), remove_831, *[neighbour_ref_832], **kwargs_835)
            
            # SSA branch for the else part of an if statement (line 255)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'strong_opps' (line 259)
            strong_opps_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps')
            int_838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
            # Applying the binary operator '+=' (line 259)
            result_iadd_839 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+=', strong_opps_837, int_838)
            # Assigning a type to the variable 'strong_opps' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'strong_opps', result_iadd_839)
            
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
        kwargs_843 = {}
        # Getting the type of 'self' (line 260)
        self_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self', False)
        # Obtaining the member 'zobrist' of a type (line 260)
        zobrist_841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_840, 'zobrist')
        # Obtaining the member 'dupe' of a type (line 260)
        dupe_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), zobrist_841, 'dupe')
        # Calling dupe(args, kwargs) (line 260)
        dupe_call_result_844 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), dupe_842, *[], **kwargs_843)
        
        # Assigning a type to the variable 'dupe' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'dupe', dupe_call_result_844)
        
        # Assigning a Name to a Attribute (line 261):
        
        # Assigning a Name to a Attribute (line 261):
        # Getting the type of 'old_hash' (line 261)
        old_hash_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 28), 'old_hash')
        # Getting the type of 'self' (line 261)
        self_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
        # Obtaining the member 'zobrist' of a type (line 261)
        zobrist_847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_846, 'zobrist')
        # Setting the type of the member 'hash' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), zobrist_847, 'hash', old_hash_845)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'dupe' (line 262)
        dupe_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'dupe')
        # Applying the 'not' unary operator (line 262)
        result_not__849 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), 'not', dupe_848)
        
        
        # Call to bool(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Evaluating a boolean operation
        # Getting the type of 'empties' (line 263)
        empties_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'empties', False)
        # Getting the type of 'weak_opps' (line 263)
        weak_opps_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 31), 'weak_opps', False)
        # Applying the binary operator 'or' (line 263)
        result_or_keyword_853 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 20), 'or', empties_851, weak_opps_852)
        
        # Evaluating a boolean operation
        # Getting the type of 'strong_neighs' (line 263)
        strong_neighs_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'strong_neighs', False)
        
        # Evaluating a boolean operation
        # Getting the type of 'strong_opps' (line 263)
        strong_opps_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 64), 'strong_opps', False)
        # Getting the type of 'weak_neighs' (line 263)
        weak_neighs_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 79), 'weak_neighs', False)
        # Applying the binary operator 'or' (line 263)
        result_or_keyword_857 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 64), 'or', strong_opps_855, weak_neighs_856)
        
        # Applying the binary operator 'and' (line 263)
        result_and_keyword_858 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 45), 'and', strong_neighs_854, result_or_keyword_857)
        
        # Applying the binary operator 'or' (line 263)
        result_or_keyword_859 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 20), 'or', result_or_keyword_853, result_and_keyword_858)
        
        # Processing the call keyword arguments (line 263)
        kwargs_860 = {}
        # Getting the type of 'bool' (line 263)
        bool_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'bool', False)
        # Calling bool(args, kwargs) (line 263)
        bool_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), bool_850, *[result_or_keyword_859], **kwargs_860)
        
        # Applying the binary operator 'and' (line 262)
        result_and_keyword_862 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), 'and', result_not__849, bool_call_result_861)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type', result_and_keyword_862)
        
        # ################# End of 'useful(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'useful' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_863)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'useful'
        return stypy_return_type_863


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
        self_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'self')
        # Obtaining the member 'emptyset' of a type (line 266)
        emptyset_871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 31), self_870, 'emptyset')
        # Obtaining the member 'empties' of a type (line 266)
        empties_872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 31), emptyset_871, 'empties')
        comprehension_873 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 16), empties_872)
        # Assigning a type to the variable 'pos' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'pos', comprehension_873)
        
        # Call to useful(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'pos' (line 266)
        pos_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 68), 'pos', False)
        # Processing the call keyword arguments (line 266)
        kwargs_868 = {}
        # Getting the type of 'self' (line 266)
        self_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 56), 'self', False)
        # Obtaining the member 'useful' of a type (line 266)
        useful_866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 56), self_865, 'useful')
        # Calling useful(args, kwargs) (line 266)
        useful_call_result_869 = invoke(stypy.reporting.localization.Localization(__file__, 266, 56), useful_866, *[pos_867], **kwargs_868)
        
        # Getting the type of 'pos' (line 266)
        pos_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'pos')
        list_874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 16), list_874, pos_864)
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type', list_874)
        
        # ################# End of 'useful_moves(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'useful_moves' in the type store
        # Getting the type of 'stypy_return_type' (line 265)
        stypy_return_type_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'useful_moves'
        return stypy_return_type_875


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
        history_876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'history')
        # Testing if the loop is going to be iterated (line 269)
        # Testing the type of a for loop iterable (line 269)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 8), history_876)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 269, 8), history_876):
            # Getting the type of the for loop variable (line 269)
            for_loop_var_877 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 8), history_876)
            # Assigning a type to the variable 'pos' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'pos', for_loop_var_877)
            # SSA begins for a for statement (line 269)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to move(...): (line 270)
            # Processing the call arguments (line 270)
            # Getting the type of 'pos' (line 270)
            pos_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'pos', False)
            # Processing the call keyword arguments (line 270)
            kwargs_881 = {}
            # Getting the type of 'self' (line 270)
            self_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'self', False)
            # Obtaining the member 'move' of a type (line 270)
            move_879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), self_878, 'move')
            # Calling move(args, kwargs) (line 270)
            move_call_result_882 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), move_879, *[pos_880], **kwargs_881)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'replay(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'replay' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'replay'
        return stypy_return_type_883


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
        color_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'color')
        # Getting the type of 'WHITE' (line 273)
        WHITE_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'WHITE')
        # Applying the binary operator '==' (line 273)
        result_eq_886 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), '==', color_884, WHITE_885)
        
        # Testing the type of an if condition (line 273)
        if_condition_887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_886)
        # Assigning a type to the variable 'if_condition_887' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_887', if_condition_887)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 274):
        
        # Assigning a BinOp to a Name (line 274):
        # Getting the type of 'KOMI' (line 274)
        KOMI_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'KOMI')
        # Getting the type of 'self' (line 274)
        self_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 27), 'self')
        # Obtaining the member 'black_dead' of a type (line 274)
        black_dead_890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 27), self_889, 'black_dead')
        # Applying the binary operator '+' (line 274)
        result_add_891 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 20), '+', KOMI_888, black_dead_890)
        
        # Assigning a type to the variable 'count' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'count', result_add_891)
        # SSA branch for the else part of an if statement (line 273)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 276):
        
        # Assigning a Attribute to a Name (line 276):
        # Getting the type of 'self' (line 276)
        self_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'self')
        # Obtaining the member 'white_dead' of a type (line 276)
        white_dead_893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 20), self_892, 'white_dead')
        # Assigning a type to the variable 'count' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'count', white_dead_893)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 277)
        self_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'self')
        # Obtaining the member 'squares' of a type (line 277)
        squares_895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 22), self_894, 'squares')
        # Testing if the loop is going to be iterated (line 277)
        # Testing the type of a for loop iterable (line 277)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 8), squares_895)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 277, 8), squares_895):
            # Getting the type of the for loop variable (line 277)
            for_loop_var_896 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 8), squares_895)
            # Assigning a type to the variable 'square' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'square', for_loop_var_896)
            # SSA begins for a for statement (line 277)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Attribute to a Name (line 278):
            
            # Assigning a Attribute to a Name (line 278):
            # Getting the type of 'square' (line 278)
            square_897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'square')
            # Obtaining the member 'color' of a type (line 278)
            color_898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 26), square_897, 'color')
            # Assigning a type to the variable 'squarecolor' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'squarecolor', color_898)
            
            
            # Getting the type of 'squarecolor' (line 279)
            squarecolor_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'squarecolor')
            # Getting the type of 'color' (line 279)
            color_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'color')
            # Applying the binary operator '==' (line 279)
            result_eq_901 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 15), '==', squarecolor_899, color_900)
            
            # Testing the type of an if condition (line 279)
            if_condition_902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 12), result_eq_901)
            # Assigning a type to the variable 'if_condition_902' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'if_condition_902', if_condition_902)
            # SSA begins for if statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'count' (line 280)
            count_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'count')
            int_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 25), 'int')
            # Applying the binary operator '+=' (line 280)
            result_iadd_905 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '+=', count_903, int_904)
            # Assigning a type to the variable 'count' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'count', result_iadd_905)
            
            # SSA branch for the else part of an if statement (line 279)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'squarecolor' (line 281)
            squarecolor_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'squarecolor')
            # Getting the type of 'EMPTY' (line 281)
            EMPTY_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'EMPTY')
            # Applying the binary operator '==' (line 281)
            result_eq_908 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 17), '==', squarecolor_906, EMPTY_907)
            
            # Testing the type of an if condition (line 281)
            if_condition_909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 17), result_eq_908)
            # Assigning a type to the variable 'if_condition_909' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'if_condition_909', if_condition_909)
            # SSA begins for if statement (line 281)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 282):
            
            # Assigning a Num to a Name (line 282):
            int_910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 27), 'int')
            # Assigning a type to the variable 'surround' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'surround', int_910)
            
            # Getting the type of 'square' (line 283)
            square_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 33), 'square')
            # Obtaining the member 'neighbours' of a type (line 283)
            neighbours_912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 33), square_911, 'neighbours')
            # Testing if the loop is going to be iterated (line 283)
            # Testing the type of a for loop iterable (line 283)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_912)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_912):
                # Getting the type of the for loop variable (line 283)
                for_loop_var_913 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 16), neighbours_912)
                # Assigning a type to the variable 'neighbour' (line 283)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'neighbour', for_loop_var_913)
                # SSA begins for a for statement (line 283)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Getting the type of 'neighbour' (line 284)
                neighbour_914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'neighbour')
                # Obtaining the member 'color' of a type (line 284)
                color_915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), neighbour_914, 'color')
                # Getting the type of 'color' (line 284)
                color_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 42), 'color')
                # Applying the binary operator '==' (line 284)
                result_eq_917 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 23), '==', color_915, color_916)
                
                # Testing the type of an if condition (line 284)
                if_condition_918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 20), result_eq_917)
                # Assigning a type to the variable 'if_condition_918' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'if_condition_918', if_condition_918)
                # SSA begins for if statement (line 284)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'surround' (line 285)
                surround_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'surround')
                int_920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 36), 'int')
                # Applying the binary operator '+=' (line 285)
                result_iadd_921 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 24), '+=', surround_919, int_920)
                # Assigning a type to the variable 'surround' (line 285)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'surround', result_iadd_921)
                
                # SSA join for if statement (line 284)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            
            # Getting the type of 'surround' (line 286)
            surround_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'surround')
            
            # Call to len(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 'square' (line 286)
            square_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), 'square', False)
            # Obtaining the member 'neighbours' of a type (line 286)
            neighbours_925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 35), square_924, 'neighbours')
            # Processing the call keyword arguments (line 286)
            kwargs_926 = {}
            # Getting the type of 'len' (line 286)
            len_923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'len', False)
            # Calling len(args, kwargs) (line 286)
            len_call_result_927 = invoke(stypy.reporting.localization.Localization(__file__, 286, 31), len_923, *[neighbours_925], **kwargs_926)
            
            # Applying the binary operator '==' (line 286)
            result_eq_928 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 19), '==', surround_922, len_call_result_927)
            
            # Testing the type of an if condition (line 286)
            if_condition_929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 16), result_eq_928)
            # Assigning a type to the variable 'if_condition_929' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'if_condition_929', if_condition_929)
            # SSA begins for if statement (line 286)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'count' (line 287)
            count_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'count')
            int_931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 29), 'int')
            # Applying the binary operator '+=' (line 287)
            result_iadd_932 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 20), '+=', count_930, int_931)
            # Assigning a type to the variable 'count' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'count', result_iadd_932)
            
            # SSA join for if statement (line 286)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 281)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 279)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'count' (line 288)
        count_933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'count')
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', count_933)
        
        # ################# End of 'score(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score'
        return stypy_return_type_934


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
        self_935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 22), 'self')
        # Obtaining the member 'squares' of a type (line 291)
        squares_936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 22), self_935, 'squares')
        # Testing if the loop is going to be iterated (line 291)
        # Testing the type of a for loop iterable (line 291)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 291, 8), squares_936)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 291, 8), squares_936):
            # Getting the type of the for loop variable (line 291)
            for_loop_var_937 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 291, 8), squares_936)
            # Assigning a type to the variable 'square' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'square', for_loop_var_937)
            # SSA begins for a for statement (line 291)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'square' (line 292)
            square_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'square')
            # Obtaining the member 'color' of a type (line 292)
            color_939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), square_938, 'color')
            # Getting the type of 'EMPTY' (line 292)
            EMPTY_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'EMPTY')
            # Applying the binary operator '==' (line 292)
            result_eq_941 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), '==', color_939, EMPTY_940)
            
            # Testing the type of an if condition (line 292)
            if_condition_942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 12), result_eq_941)
            # Assigning a type to the variable 'if_condition_942' (line 292)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'if_condition_942', if_condition_942)
            # SSA begins for if statement (line 292)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 292)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 295):
            
            # Assigning a Call to a Name (line 295):
            
            # Call to set(...): (line 295)
            # Processing the call arguments (line 295)
            
            # Obtaining an instance of the builtin type 'list' (line 295)
            list_944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 295)
            # Adding element type (line 295)
            # Getting the type of 'square' (line 295)
            square_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 28), 'square', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 27), list_944, square_945)
            
            # Processing the call keyword arguments (line 295)
            kwargs_946 = {}
            # Getting the type of 'set' (line 295)
            set_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'set', False)
            # Calling set(args, kwargs) (line 295)
            set_call_result_947 = invoke(stypy.reporting.localization.Localization(__file__, 295, 23), set_943, *[list_944], **kwargs_946)
            
            # Assigning a type to the variable 'members1' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'members1', set_call_result_947)
            
            # Assigning a Name to a Name (line 296):
            
            # Assigning a Name to a Name (line 296):
            # Getting the type of 'True' (line 296)
            True_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'True')
            # Assigning a type to the variable 'changed' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'changed', True_948)
            
            # Getting the type of 'changed' (line 297)
            changed_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 18), 'changed')
            # Testing the type of an if condition (line 297)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 12), changed_949)
            # SSA begins for while statement (line 297)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Name to a Name (line 298):
            
            # Assigning a Name to a Name (line 298):
            # Getting the type of 'False' (line 298)
            False_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 26), 'False')
            # Assigning a type to the variable 'changed' (line 298)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'changed', False_950)
            
            
            # Call to copy(...): (line 299)
            # Processing the call keyword arguments (line 299)
            kwargs_953 = {}
            # Getting the type of 'members1' (line 299)
            members1_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 30), 'members1', False)
            # Obtaining the member 'copy' of a type (line 299)
            copy_952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 30), members1_951, 'copy')
            # Calling copy(args, kwargs) (line 299)
            copy_call_result_954 = invoke(stypy.reporting.localization.Localization(__file__, 299, 30), copy_952, *[], **kwargs_953)
            
            # Testing if the loop is going to be iterated (line 299)
            # Testing the type of a for loop iterable (line 299)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 16), copy_call_result_954)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 299, 16), copy_call_result_954):
                # Getting the type of the for loop variable (line 299)
                for_loop_var_955 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 16), copy_call_result_954)
                # Assigning a type to the variable 'member' (line 299)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'member', for_loop_var_955)
                # SSA begins for a for statement (line 299)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'member' (line 300)
                member_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 37), 'member')
                # Obtaining the member 'neighbours' of a type (line 300)
                neighbours_957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 37), member_956, 'neighbours')
                # Testing if the loop is going to be iterated (line 300)
                # Testing the type of a for loop iterable (line 300)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 20), neighbours_957)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 300, 20), neighbours_957):
                    # Getting the type of the for loop variable (line 300)
                    for_loop_var_958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 20), neighbours_957)
                    # Assigning a type to the variable 'neighbour' (line 300)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'neighbour', for_loop_var_958)
                    # SSA begins for a for statement (line 300)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'neighbour' (line 301)
                    neighbour_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 27), 'neighbour')
                    # Obtaining the member 'color' of a type (line 301)
                    color_960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 27), neighbour_959, 'color')
                    # Getting the type of 'square' (line 301)
                    square_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 46), 'square')
                    # Obtaining the member 'color' of a type (line 301)
                    color_962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 46), square_961, 'color')
                    # Applying the binary operator '==' (line 301)
                    result_eq_963 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 27), '==', color_960, color_962)
                    
                    
                    # Getting the type of 'neighbour' (line 301)
                    neighbour_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 63), 'neighbour')
                    # Getting the type of 'members1' (line 301)
                    members1_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 80), 'members1')
                    # Applying the binary operator 'notin' (line 301)
                    result_contains_966 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 63), 'notin', neighbour_964, members1_965)
                    
                    # Applying the binary operator 'and' (line 301)
                    result_and_keyword_967 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 27), 'and', result_eq_963, result_contains_966)
                    
                    # Testing the type of an if condition (line 301)
                    if_condition_968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 24), result_and_keyword_967)
                    # Assigning a type to the variable 'if_condition_968' (line 301)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'if_condition_968', if_condition_968)
                    # SSA begins for if statement (line 301)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 302):
                    
                    # Assigning a Name to a Name (line 302):
                    # Getting the type of 'True' (line 302)
                    True_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'True')
                    # Assigning a type to the variable 'changed' (line 302)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'changed', True_969)
                    
                    # Call to add(...): (line 303)
                    # Processing the call arguments (line 303)
                    # Getting the type of 'neighbour' (line 303)
                    neighbour_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 41), 'neighbour', False)
                    # Processing the call keyword arguments (line 303)
                    kwargs_973 = {}
                    # Getting the type of 'members1' (line 303)
                    members1_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 28), 'members1', False)
                    # Obtaining the member 'add' of a type (line 303)
                    add_971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 28), members1_970, 'add')
                    # Calling add(args, kwargs) (line 303)
                    add_call_result_974 = invoke(stypy.reporting.localization.Localization(__file__, 303, 28), add_971, *[neighbour_972], **kwargs_973)
                    
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
            kwargs_976 = {}
            # Getting the type of 'set' (line 304)
            set_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'set', False)
            # Calling set(args, kwargs) (line 304)
            set_call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 304, 25), set_975, *[], **kwargs_976)
            
            # Assigning a type to the variable 'liberties1' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'liberties1', set_call_result_977)
            
            # Getting the type of 'members1' (line 305)
            members1_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 26), 'members1')
            # Testing if the loop is going to be iterated (line 305)
            # Testing the type of a for loop iterable (line 305)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 305, 12), members1_978)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 305, 12), members1_978):
                # Getting the type of the for loop variable (line 305)
                for_loop_var_979 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 305, 12), members1_978)
                # Assigning a type to the variable 'member' (line 305)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'member', for_loop_var_979)
                # SSA begins for a for statement (line 305)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'member' (line 306)
                member_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 'member')
                # Obtaining the member 'neighbours' of a type (line 306)
                neighbours_981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 33), member_980, 'neighbours')
                # Testing if the loop is going to be iterated (line 306)
                # Testing the type of a for loop iterable (line 306)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 306, 16), neighbours_981)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 306, 16), neighbours_981):
                    # Getting the type of the for loop variable (line 306)
                    for_loop_var_982 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 306, 16), neighbours_981)
                    # Assigning a type to the variable 'neighbour' (line 306)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'neighbour', for_loop_var_982)
                    # SSA begins for a for statement (line 306)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Getting the type of 'neighbour' (line 307)
                    neighbour_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'neighbour')
                    # Obtaining the member 'color' of a type (line 307)
                    color_984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 23), neighbour_983, 'color')
                    # Getting the type of 'EMPTY' (line 307)
                    EMPTY_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 42), 'EMPTY')
                    # Applying the binary operator '==' (line 307)
                    result_eq_986 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 23), '==', color_984, EMPTY_985)
                    
                    # Testing the type of an if condition (line 307)
                    if_condition_987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 20), result_eq_986)
                    # Assigning a type to the variable 'if_condition_987' (line 307)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'if_condition_987', if_condition_987)
                    # SSA begins for if statement (line 307)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to add(...): (line 308)
                    # Processing the call arguments (line 308)
                    # Getting the type of 'neighbour' (line 308)
                    neighbour_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'neighbour', False)
                    # Obtaining the member 'pos' of a type (line 308)
                    pos_991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 39), neighbour_990, 'pos')
                    # Processing the call keyword arguments (line 308)
                    kwargs_992 = {}
                    # Getting the type of 'liberties1' (line 308)
                    liberties1_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'liberties1', False)
                    # Obtaining the member 'add' of a type (line 308)
                    add_989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 24), liberties1_988, 'add')
                    # Calling add(args, kwargs) (line 308)
                    add_call_result_993 = invoke(stypy.reporting.localization.Localization(__file__, 308, 24), add_989, *[pos_991], **kwargs_992)
                    
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
            kwargs_996 = {}
            # Getting the type of 'square' (line 310)
            square_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'square', False)
            # Obtaining the member 'find' of a type (line 310)
            find_995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), square_994, 'find')
            # Calling find(args, kwargs) (line 310)
            find_call_result_997 = invoke(stypy.reporting.localization.Localization(__file__, 310, 19), find_995, *[], **kwargs_996)
            
            # Assigning a type to the variable 'root' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'root', find_call_result_997)
            
            # Assigning a Call to a Name (line 315):
            
            # Assigning a Call to a Name (line 315):
            
            # Call to set(...): (line 315)
            # Processing the call keyword arguments (line 315)
            kwargs_999 = {}
            # Getting the type of 'set' (line 315)
            set_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'set', False)
            # Calling set(args, kwargs) (line 315)
            set_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 315, 23), set_998, *[], **kwargs_999)
            
            # Assigning a type to the variable 'members2' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'members2', set_call_result_1000)
            
            # Getting the type of 'self' (line 316)
            self_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'self')
            # Obtaining the member 'squares' of a type (line 316)
            squares_1002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 27), self_1001, 'squares')
            # Testing if the loop is going to be iterated (line 316)
            # Testing the type of a for loop iterable (line 316)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 316, 12), squares_1002)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 316, 12), squares_1002):
                # Getting the type of the for loop variable (line 316)
                for_loop_var_1003 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 316, 12), squares_1002)
                # Assigning a type to the variable 'square2' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'square2', for_loop_var_1003)
                # SSA begins for a for statement (line 316)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'square2' (line 317)
                square2_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'square2')
                # Obtaining the member 'color' of a type (line 317)
                color_1005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 19), square2_1004, 'color')
                # Getting the type of 'EMPTY' (line 317)
                EMPTY_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 36), 'EMPTY')
                # Applying the binary operator '!=' (line 317)
                result_ne_1007 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 19), '!=', color_1005, EMPTY_1006)
                
                
                
                # Call to find(...): (line 317)
                # Processing the call keyword arguments (line 317)
                kwargs_1010 = {}
                # Getting the type of 'square2' (line 317)
                square2_1008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 46), 'square2', False)
                # Obtaining the member 'find' of a type (line 317)
                find_1009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 46), square2_1008, 'find')
                # Calling find(args, kwargs) (line 317)
                find_call_result_1011 = invoke(stypy.reporting.localization.Localization(__file__, 317, 46), find_1009, *[], **kwargs_1010)
                
                # Getting the type of 'root' (line 317)
                root_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 64), 'root')
                # Applying the binary operator '==' (line 317)
                result_eq_1013 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 46), '==', find_call_result_1011, root_1012)
                
                # Applying the binary operator 'and' (line 317)
                result_and_keyword_1014 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 19), 'and', result_ne_1007, result_eq_1013)
                
                # Testing the type of an if condition (line 317)
                if_condition_1015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 16), result_and_keyword_1014)
                # Assigning a type to the variable 'if_condition_1015' (line 317)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'if_condition_1015', if_condition_1015)
                # SSA begins for if statement (line 317)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to add(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 'square2' (line 318)
                square2_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 33), 'square2', False)
                # Processing the call keyword arguments (line 318)
                kwargs_1019 = {}
                # Getting the type of 'members2' (line 318)
                members2_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'members2', False)
                # Obtaining the member 'add' of a type (line 318)
                add_1017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 20), members2_1016, 'add')
                # Calling add(args, kwargs) (line 318)
                add_call_result_1020 = invoke(stypy.reporting.localization.Localization(__file__, 318, 20), add_1017, *[square2_1018], **kwargs_1019)
                
                # SSA join for if statement (line 317)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Attribute to a Name (line 320):
            
            # Assigning a Attribute to a Name (line 320):
            # Getting the type of 'root' (line 320)
            root_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), 'root')
            # Obtaining the member 'liberties' of a type (line 320)
            liberties_1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 25), root_1021, 'liberties')
            # Assigning a type to the variable 'liberties2' (line 320)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'liberties2', liberties_1022)
            # Evaluating assert statement condition
            
            # Getting the type of 'members1' (line 324)
            members1_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'members1')
            # Getting the type of 'members2' (line 324)
            members2_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 31), 'members2')
            # Applying the binary operator '==' (line 324)
            result_eq_1025 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 19), '==', members1_1023, members2_1024)
            
            # Evaluating assert statement condition
            
            
            # Call to len(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'liberties1' (line 325)
            liberties1_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), 'liberties1', False)
            # Processing the call keyword arguments (line 325)
            kwargs_1028 = {}
            # Getting the type of 'len' (line 325)
            len_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'len', False)
            # Calling len(args, kwargs) (line 325)
            len_call_result_1029 = invoke(stypy.reporting.localization.Localization(__file__, 325, 19), len_1026, *[liberties1_1027], **kwargs_1028)
            
            # Getting the type of 'liberties2' (line 325)
            liberties2_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'liberties2')
            # Applying the binary operator '==' (line 325)
            result_eq_1031 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 19), '==', len_call_result_1029, liberties2_1030)
            
            
            # Assigning a Call to a Name (line 328):
            
            # Assigning a Call to a Name (line 328):
            
            # Call to set(...): (line 328)
            # Processing the call arguments (line 328)
            # Getting the type of 'self' (line 328)
            self_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'self', False)
            # Obtaining the member 'emptyset' of a type (line 328)
            emptyset_1034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 27), self_1033, 'emptyset')
            # Obtaining the member 'empties' of a type (line 328)
            empties_1035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 27), emptyset_1034, 'empties')
            # Processing the call keyword arguments (line 328)
            kwargs_1036 = {}
            # Getting the type of 'set' (line 328)
            set_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), 'set', False)
            # Calling set(args, kwargs) (line 328)
            set_call_result_1037 = invoke(stypy.reporting.localization.Localization(__file__, 328, 23), set_1032, *[empties_1035], **kwargs_1036)
            
            # Assigning a type to the variable 'empties1' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'empties1', set_call_result_1037)
            
            # Assigning a Call to a Name (line 330):
            
            # Assigning a Call to a Name (line 330):
            
            # Call to set(...): (line 330)
            # Processing the call keyword arguments (line 330)
            kwargs_1039 = {}
            # Getting the type of 'set' (line 330)
            set_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 23), 'set', False)
            # Calling set(args, kwargs) (line 330)
            set_call_result_1040 = invoke(stypy.reporting.localization.Localization(__file__, 330, 23), set_1038, *[], **kwargs_1039)
            
            # Assigning a type to the variable 'empties2' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'empties2', set_call_result_1040)
            
            # Getting the type of 'self' (line 331)
            self_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'self')
            # Obtaining the member 'squares' of a type (line 331)
            squares_1042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 26), self_1041, 'squares')
            # Testing if the loop is going to be iterated (line 331)
            # Testing the type of a for loop iterable (line 331)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 331, 12), squares_1042)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 331, 12), squares_1042):
                # Getting the type of the for loop variable (line 331)
                for_loop_var_1043 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 331, 12), squares_1042)
                # Assigning a type to the variable 'square' (line 331)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'square', for_loop_var_1043)
                # SSA begins for a for statement (line 331)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Getting the type of 'square' (line 332)
                square_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'square')
                # Obtaining the member 'color' of a type (line 332)
                color_1045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), square_1044, 'color')
                # Getting the type of 'EMPTY' (line 332)
                EMPTY_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'EMPTY')
                # Applying the binary operator '==' (line 332)
                result_eq_1047 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 19), '==', color_1045, EMPTY_1046)
                
                # Testing the type of an if condition (line 332)
                if_condition_1048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 16), result_eq_1047)
                # Assigning a type to the variable 'if_condition_1048' (line 332)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'if_condition_1048', if_condition_1048)
                # SSA begins for if statement (line 332)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to add(...): (line 333)
                # Processing the call arguments (line 333)
                # Getting the type of 'square' (line 333)
                square_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 33), 'square', False)
                # Obtaining the member 'pos' of a type (line 333)
                pos_1052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 33), square_1051, 'pos')
                # Processing the call keyword arguments (line 333)
                kwargs_1053 = {}
                # Getting the type of 'empties2' (line 333)
                empties2_1049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'empties2', False)
                # Obtaining the member 'add' of a type (line 333)
                add_1050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 20), empties2_1049, 'add')
                # Calling add(args, kwargs) (line 333)
                add_call_result_1054 = invoke(stypy.reporting.localization.Localization(__file__, 333, 20), add_1050, *[pos_1052], **kwargs_1053)
                
                # SSA join for if statement (line 332)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Evaluating assert statement condition
            
            # Getting the type of 'empties1' (line 335)
            empties1_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'empties1')
            # Getting the type of 'empties2' (line 335)
            empties2_1056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'empties2')
            # Applying the binary operator '==' (line 335)
            result_eq_1057 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 19), '==', empties1_1055, empties2_1056)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1058)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_1058


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
        list_1059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        
        # Assigning a type to the variable 'result' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'result', list_1059)
        
        
        # Call to range(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'SIZE' (line 339)
        SIZE_1061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'SIZE', False)
        # Processing the call keyword arguments (line 339)
        kwargs_1062 = {}
        # Getting the type of 'range' (line 339)
        range_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'range', False)
        # Calling range(args, kwargs) (line 339)
        range_call_result_1063 = invoke(stypy.reporting.localization.Localization(__file__, 339, 17), range_1060, *[SIZE_1061], **kwargs_1062)
        
        # Testing if the loop is going to be iterated (line 339)
        # Testing the type of a for loop iterable (line 339)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_1063)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_1063):
            # Getting the type of the for loop variable (line 339)
            for_loop_var_1064 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 339, 8), range_call_result_1063)
            # Assigning a type to the variable 'y' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'y', for_loop_var_1064)
            # SSA begins for a for statement (line 339)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 340):
            
            # Assigning a Call to a Name (line 340):
            
            # Call to to_pos(...): (line 340)
            # Processing the call arguments (line 340)
            int_1066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 27), 'int')
            # Getting the type of 'y' (line 340)
            y_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 30), 'y', False)
            # Processing the call keyword arguments (line 340)
            kwargs_1068 = {}
            # Getting the type of 'to_pos' (line 340)
            to_pos_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'to_pos', False)
            # Calling to_pos(args, kwargs) (line 340)
            to_pos_call_result_1069 = invoke(stypy.reporting.localization.Localization(__file__, 340, 20), to_pos_1065, *[int_1066, y_1067], **kwargs_1068)
            
            # Assigning a type to the variable 'start' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'start', to_pos_call_result_1069)
            
            # Call to append(...): (line 341)
            # Processing the call arguments (line 341)
            
            # Call to join(...): (line 341)
            # Processing the call arguments (line 341)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Obtaining the type of the subscript
            # Getting the type of 'start' (line 341)
            start_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 87), 'start', False)
            # Getting the type of 'start' (line 341)
            start_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 93), 'start', False)
            # Getting the type of 'SIZE' (line 341)
            SIZE_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 101), 'SIZE', False)
            # Applying the binary operator '+' (line 341)
            result_add_1084 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 93), '+', start_1082, SIZE_1083)
            
            slice_1085 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 74), start_1081, result_add_1084, None)
            # Getting the type of 'self' (line 341)
            self_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 74), 'self', False)
            # Obtaining the member 'squares' of a type (line 341)
            squares_1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 74), self_1086, 'squares')
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___1088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 74), squares_1087, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_1089 = invoke(stypy.reporting.localization.Localization(__file__, 341, 74), getitem___1088, slice_1085)
            
            comprehension_1090 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 35), subscript_call_result_1089)
            # Assigning a type to the variable 'square' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'square', comprehension_1090)
            
            # Obtaining the type of the subscript
            # Getting the type of 'square' (line 341)
            square_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 40), 'square', False)
            # Obtaining the member 'color' of a type (line 341)
            color_1075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 40), square_1074, 'color')
            # Getting the type of 'SHOW' (line 341)
            SHOW_1076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'SHOW', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___1077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 35), SHOW_1076, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 341, 35), getitem___1077, color_1075)
            
            str_1079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 56), 'str', ' ')
            # Applying the binary operator '+' (line 341)
            result_add_1080 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 35), '+', subscript_call_result_1078, str_1079)
            
            list_1091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 35), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 35), list_1091, result_add_1080)
            # Processing the call keyword arguments (line 341)
            kwargs_1092 = {}
            str_1072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 26), 'str', '')
            # Obtaining the member 'join' of a type (line 341)
            join_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 26), str_1072, 'join')
            # Calling join(args, kwargs) (line 341)
            join_call_result_1093 = invoke(stypy.reporting.localization.Localization(__file__, 341, 26), join_1073, *[list_1091], **kwargs_1092)
            
            # Processing the call keyword arguments (line 341)
            kwargs_1094 = {}
            # Getting the type of 'result' (line 341)
            result_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'result', False)
            # Obtaining the member 'append' of a type (line 341)
            append_1071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), result_1070, 'append')
            # Calling append(args, kwargs) (line 341)
            append_call_result_1095 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), append_1071, *[join_call_result_1093], **kwargs_1094)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to join(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'result' (line 342)
        result_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'result', False)
        # Processing the call keyword arguments (line 342)
        kwargs_1099 = {}
        str_1096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 342)
        join_1097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), str_1096, 'join')
        # Calling join(args, kwargs) (line 342)
        join_call_result_1100 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), join_1097, *[result_1098], **kwargs_1099)
        
        # Assigning a type to the variable 'stypy_return_type' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'stypy_return_type', join_call_result_1100)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1101)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_1101


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
        None_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 25), 'None')
        # Getting the type of 'self' (line 347)
        self_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self')
        # Setting the type of the member 'bestchild' of a type (line 347)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_1103, 'bestchild', None_1102)
        
        # Assigning a Num to a Attribute (line 348):
        
        # Assigning a Num to a Attribute (line 348):
        int_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 19), 'int')
        # Getting the type of 'self' (line 348)
        self_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 348)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_1105, 'pos', int_1104)
        
        # Assigning a Num to a Attribute (line 349):
        
        # Assigning a Num to a Attribute (line 349):
        int_1106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 20), 'int')
        # Getting the type of 'self' (line 349)
        self_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'self')
        # Setting the type of the member 'wins' of a type (line 349)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), self_1107, 'wins', int_1106)
        
        # Assigning a Num to a Attribute (line 350):
        
        # Assigning a Num to a Attribute (line 350):
        int_1108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 22), 'int')
        # Getting the type of 'self' (line 350)
        self_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self')
        # Setting the type of the member 'losses' of a type (line 350)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_1109, 'losses', int_1108)
        
        # Assigning a ListComp to a Attribute (line 351):
        
        # Assigning a ListComp to a Attribute (line 351):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'SIZE' (line 351)
        SIZE_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 46), 'SIZE', False)
        # Getting the type of 'SIZE' (line 351)
        SIZE_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 53), 'SIZE', False)
        # Applying the binary operator '*' (line 351)
        result_mul_1114 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 46), '*', SIZE_1112, SIZE_1113)
        
        # Processing the call keyword arguments (line 351)
        kwargs_1115 = {}
        # Getting the type of 'range' (line 351)
        range_1111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 40), 'range', False)
        # Calling range(args, kwargs) (line 351)
        range_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 351, 40), range_1111, *[result_mul_1114], **kwargs_1115)
        
        comprehension_1117 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 26), range_call_result_1116)
        # Assigning a type to the variable 'x' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'x', comprehension_1117)
        # Getting the type of 'None' (line 351)
        None_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'None')
        list_1118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 26), list_1118, None_1110)
        # Getting the type of 'self' (line 351)
        self_1119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self')
        # Setting the type of the member 'pos_child' of a type (line 351)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_1119, 'pos_child', list_1118)
        
        # Assigning a Num to a Attribute (line 352):
        
        # Assigning a Num to a Attribute (line 352):
        int_1120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 26), 'int')
        # Getting the type of 'self' (line 352)
        self_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self')
        # Setting the type of the member 'amafvisits' of a type (line 352)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), self_1121, 'amafvisits', int_1120)
        
        # Assigning a ListComp to a Attribute (line 353):
        
        # Assigning a ListComp to a Attribute (line 353):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'SIZE' (line 353)
        SIZE_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 47), 'SIZE', False)
        # Getting the type of 'SIZE' (line 353)
        SIZE_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 54), 'SIZE', False)
        # Applying the binary operator '*' (line 353)
        result_mul_1126 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 47), '*', SIZE_1124, SIZE_1125)
        
        # Processing the call keyword arguments (line 353)
        kwargs_1127 = {}
        # Getting the type of 'range' (line 353)
        range_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 41), 'range', False)
        # Calling range(args, kwargs) (line 353)
        range_call_result_1128 = invoke(stypy.reporting.localization.Localization(__file__, 353, 41), range_1123, *[result_mul_1126], **kwargs_1127)
        
        comprehension_1129 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 30), range_call_result_1128)
        # Assigning a type to the variable 'x' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'x', comprehension_1129)
        int_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 30), 'int')
        list_1130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 30), list_1130, int_1122)
        # Getting the type of 'self' (line 353)
        self_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self')
        # Setting the type of the member 'pos_amaf_wins' of a type (line 353)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), self_1131, 'pos_amaf_wins', list_1130)
        
        # Assigning a ListComp to a Attribute (line 354):
        
        # Assigning a ListComp to a Attribute (line 354):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'SIZE' (line 354)
        SIZE_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 49), 'SIZE', False)
        # Getting the type of 'SIZE' (line 354)
        SIZE_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 56), 'SIZE', False)
        # Applying the binary operator '*' (line 354)
        result_mul_1136 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 49), '*', SIZE_1134, SIZE_1135)
        
        # Processing the call keyword arguments (line 354)
        kwargs_1137 = {}
        # Getting the type of 'range' (line 354)
        range_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 43), 'range', False)
        # Calling range(args, kwargs) (line 354)
        range_call_result_1138 = invoke(stypy.reporting.localization.Localization(__file__, 354, 43), range_1133, *[result_mul_1136], **kwargs_1137)
        
        comprehension_1139 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 32), range_call_result_1138)
        # Assigning a type to the variable 'x' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 32), 'x', comprehension_1139)
        int_1132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 32), 'int')
        list_1140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 32), list_1140, int_1132)
        # Getting the type of 'self' (line 354)
        self_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'self')
        # Setting the type of the member 'pos_amaf_losses' of a type (line 354)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), self_1141, 'pos_amaf_losses', list_1140)
        
        # Assigning a Name to a Attribute (line 355):
        
        # Assigning a Name to a Attribute (line 355):
        # Getting the type of 'None' (line 355)
        None_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'None')
        # Getting the type of 'self' (line 355)
        self_1143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self')
        # Setting the type of the member 'parent' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_1143, 'parent', None_1142)
        
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

        str_1144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 8), 'str', ' uct tree search ')
        
        # Assigning a Attribute to a Name (line 359):
        
        # Assigning a Attribute to a Name (line 359):
        # Getting the type of 'board' (line 359)
        board_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'board')
        # Obtaining the member 'color' of a type (line 359)
        color_1146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), board_1145, 'color')
        # Assigning a type to the variable 'color' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'color', color_1146)
        
        # Assigning a Name to a Name (line 360):
        
        # Assigning a Name to a Name (line 360):
        # Getting the type of 'self' (line 360)
        self_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'self')
        # Assigning a type to the variable 'node' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'node', self_1147)
        
        # Assigning a List to a Name (line 361):
        
        # Assigning a List to a Name (line 361):
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_1148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        # Getting the type of 'node' (line 361)
        node_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'node')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 15), list_1148, node_1149)
        
        # Assigning a type to the variable 'path' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'path', list_1148)
        
        # Assigning a Call to a Name (line 362):
        
        # Assigning a Call to a Name (line 362):
        
        # Call to len(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'board' (line 362)
        board_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'board', False)
        # Obtaining the member 'history' of a type (line 362)
        history_1152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 22), board_1151, 'history')
        # Processing the call keyword arguments (line 362)
        kwargs_1153 = {}
        # Getting the type of 'len' (line 362)
        len_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 18), 'len', False)
        # Calling len(args, kwargs) (line 362)
        len_call_result_1154 = invoke(stypy.reporting.localization.Localization(__file__, 362, 18), len_1150, *[history_1152], **kwargs_1153)
        
        # Assigning a type to the variable 'histpos' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'histpos', len_call_result_1154)
        
        # Getting the type of 'True' (line 363)
        True_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 14), 'True')
        # Testing the type of an if condition (line 363)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 8), True_1155)
        # SSA begins for while statement (line 363)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 364):
        
        # Assigning a Call to a Name (line 364):
        
        # Call to select(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'board' (line 364)
        board_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'board', False)
        # Processing the call keyword arguments (line 364)
        kwargs_1159 = {}
        # Getting the type of 'node' (line 364)
        node_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 18), 'node', False)
        # Obtaining the member 'select' of a type (line 364)
        select_1157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 18), node_1156, 'select')
        # Calling select(args, kwargs) (line 364)
        select_call_result_1160 = invoke(stypy.reporting.localization.Localization(__file__, 364, 18), select_1157, *[board_1158], **kwargs_1159)
        
        # Assigning a type to the variable 'pos' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'pos', select_call_result_1160)
        
        
        # Getting the type of 'pos' (line 365)
        pos_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'pos')
        # Getting the type of 'PASS' (line 365)
        PASS_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'PASS')
        # Applying the binary operator '==' (line 365)
        result_eq_1163 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 15), '==', pos_1161, PASS_1162)
        
        # Testing the type of an if condition (line 365)
        if_condition_1164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 12), result_eq_1163)
        # Assigning a type to the variable 'if_condition_1164' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'if_condition_1164', if_condition_1164)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to move(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'pos' (line 367)
        pos_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'pos', False)
        # Processing the call keyword arguments (line 367)
        kwargs_1168 = {}
        # Getting the type of 'board' (line 367)
        board_1165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'board', False)
        # Obtaining the member 'move' of a type (line 367)
        move_1166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), board_1165, 'move')
        # Calling move(args, kwargs) (line 367)
        move_call_result_1169 = invoke(stypy.reporting.localization.Localization(__file__, 367, 12), move_1166, *[pos_1167], **kwargs_1168)
        
        
        # Assigning a Subscript to a Name (line 368):
        
        # Assigning a Subscript to a Name (line 368):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 368)
        pos_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'pos')
        # Getting the type of 'node' (line 368)
        node_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 20), 'node')
        # Obtaining the member 'pos_child' of a type (line 368)
        pos_child_1172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 20), node_1171, 'pos_child')
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___1173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 20), pos_child_1172, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_1174 = invoke(stypy.reporting.localization.Localization(__file__, 368, 20), getitem___1173, pos_1170)
        
        # Assigning a type to the variable 'child' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'child', subscript_call_result_1174)
        
        
        # Getting the type of 'child' (line 369)
        child_1175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'child')
        # Applying the 'not' unary operator (line 369)
        result_not__1176 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 15), 'not', child_1175)
        
        # Testing the type of an if condition (line 369)
        if_condition_1177 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 12), result_not__1176)
        # Assigning a type to the variable 'if_condition_1177' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'if_condition_1177', if_condition_1177)
        # SSA begins for if statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Subscript (line 370):
        
        # Call to UCTNode(...): (line 370)
        # Processing the call keyword arguments (line 370)
        kwargs_1179 = {}
        # Getting the type of 'UCTNode' (line 370)
        UCTNode_1178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 46), 'UCTNode', False)
        # Calling UCTNode(args, kwargs) (line 370)
        UCTNode_call_result_1180 = invoke(stypy.reporting.localization.Localization(__file__, 370, 46), UCTNode_1178, *[], **kwargs_1179)
        
        # Getting the type of 'node' (line 370)
        node_1181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'node')
        # Obtaining the member 'pos_child' of a type (line 370)
        pos_child_1182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), node_1181, 'pos_child')
        # Getting the type of 'pos' (line 370)
        pos_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 39), 'pos')
        # Storing an element on a container (line 370)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 24), pos_child_1182, (pos_1183, UCTNode_call_result_1180))
        
        # Assigning a Subscript to a Name (line 370):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 370)
        pos_1184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 39), 'pos')
        # Getting the type of 'node' (line 370)
        node_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'node')
        # Obtaining the member 'pos_child' of a type (line 370)
        pos_child_1186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), node_1185, 'pos_child')
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___1187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), pos_child_1186, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_1188 = invoke(stypy.reporting.localization.Localization(__file__, 370, 24), getitem___1187, pos_1184)
        
        # Assigning a type to the variable 'child' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'child', subscript_call_result_1188)
        
        # Assigning a Call to a Attribute (line 371):
        
        # Assigning a Call to a Attribute (line 371):
        
        # Call to useful_moves(...): (line 371)
        # Processing the call keyword arguments (line 371)
        kwargs_1191 = {}
        # Getting the type of 'board' (line 371)
        board_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 35), 'board', False)
        # Obtaining the member 'useful_moves' of a type (line 371)
        useful_moves_1190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 35), board_1189, 'useful_moves')
        # Calling useful_moves(args, kwargs) (line 371)
        useful_moves_call_result_1192 = invoke(stypy.reporting.localization.Localization(__file__, 371, 35), useful_moves_1190, *[], **kwargs_1191)
        
        # Getting the type of 'child' (line 371)
        child_1193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'child')
        # Setting the type of the member 'unexplored' of a type (line 371)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 16), child_1193, 'unexplored', useful_moves_call_result_1192)
        
        # Assigning a Name to a Attribute (line 372):
        
        # Assigning a Name to a Attribute (line 372):
        # Getting the type of 'pos' (line 372)
        pos_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 28), 'pos')
        # Getting the type of 'child' (line 372)
        child_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'child')
        # Setting the type of the member 'pos' of a type (line 372)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), child_1195, 'pos', pos_1194)
        
        # Assigning a Name to a Attribute (line 373):
        
        # Assigning a Name to a Attribute (line 373):
        # Getting the type of 'node' (line 373)
        node_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'node')
        # Getting the type of 'child' (line 373)
        child_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'child')
        # Setting the type of the member 'parent' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), child_1197, 'parent', node_1196)
        
        # Call to append(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'child' (line 374)
        child_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'child', False)
        # Processing the call keyword arguments (line 374)
        kwargs_1201 = {}
        # Getting the type of 'path' (line 374)
        path_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'path', False)
        # Obtaining the member 'append' of a type (line 374)
        append_1199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 16), path_1198, 'append')
        # Calling append(args, kwargs) (line 374)
        append_call_result_1202 = invoke(stypy.reporting.localization.Localization(__file__, 374, 16), append_1199, *[child_1200], **kwargs_1201)
        
        # SSA join for if statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'child' (line 376)
        child_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 24), 'child', False)
        # Processing the call keyword arguments (line 376)
        kwargs_1206 = {}
        # Getting the type of 'path' (line 376)
        path_1203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'path', False)
        # Obtaining the member 'append' of a type (line 376)
        append_1204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), path_1203, 'append')
        # Calling append(args, kwargs) (line 376)
        append_call_result_1207 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), append_1204, *[child_1205], **kwargs_1206)
        
        
        # Assigning a Name to a Name (line 377):
        
        # Assigning a Name to a Name (line 377):
        # Getting the type of 'child' (line 377)
        child_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'child')
        # Assigning a type to the variable 'node' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'node', child_1208)
        # SSA join for while statement (line 363)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to random_playout(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'board' (line 378)
        board_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 28), 'board', False)
        # Processing the call keyword arguments (line 378)
        kwargs_1212 = {}
        # Getting the type of 'self' (line 378)
        self_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self', False)
        # Obtaining the member 'random_playout' of a type (line 378)
        random_playout_1210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_1209, 'random_playout')
        # Calling random_playout(args, kwargs) (line 378)
        random_playout_call_result_1213 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), random_playout_1210, *[board_1211], **kwargs_1212)
        
        
        # Call to update_path(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'board' (line 379)
        board_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'board', False)
        # Getting the type of 'histpos' (line 379)
        histpos_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 32), 'histpos', False)
        # Getting the type of 'color' (line 379)
        color_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 41), 'color', False)
        # Getting the type of 'path' (line 379)
        path_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 48), 'path', False)
        # Processing the call keyword arguments (line 379)
        kwargs_1220 = {}
        # Getting the type of 'self' (line 379)
        self_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'self', False)
        # Obtaining the member 'update_path' of a type (line 379)
        update_path_1215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), self_1214, 'update_path')
        # Calling update_path(args, kwargs) (line 379)
        update_path_call_result_1221 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), update_path_1215, *[board_1216, histpos_1217, color_1218, path_1219], **kwargs_1220)
        
        
        # ################# End of 'play(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'play' in the type store
        # Getting the type of 'stypy_return_type' (line 357)
        stypy_return_type_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1222)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'play'
        return stypy_return_type_1222


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

        str_1223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 8), 'str', ' select move; unexplored children first, then according to uct value ')
        
        # Getting the type of 'self' (line 383)
        self_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'self')
        # Obtaining the member 'unexplored' of a type (line 383)
        unexplored_1225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 11), self_1224, 'unexplored')
        # Testing the type of an if condition (line 383)
        if_condition_1226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), unexplored_1225)
        # Assigning a type to the variable 'if_condition_1226' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_1226', if_condition_1226)
        # SSA begins for if statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 384):
        
        # Assigning a Call to a Name (line 384):
        
        # Call to randrange(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Call to len(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'self' (line 384)
        self_1230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 37), 'self', False)
        # Obtaining the member 'unexplored' of a type (line 384)
        unexplored_1231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 37), self_1230, 'unexplored')
        # Processing the call keyword arguments (line 384)
        kwargs_1232 = {}
        # Getting the type of 'len' (line 384)
        len_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 33), 'len', False)
        # Calling len(args, kwargs) (line 384)
        len_call_result_1233 = invoke(stypy.reporting.localization.Localization(__file__, 384, 33), len_1229, *[unexplored_1231], **kwargs_1232)
        
        # Processing the call keyword arguments (line 384)
        kwargs_1234 = {}
        # Getting the type of 'random' (line 384)
        random_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'random', False)
        # Obtaining the member 'randrange' of a type (line 384)
        randrange_1228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 16), random_1227, 'randrange')
        # Calling randrange(args, kwargs) (line 384)
        randrange_call_result_1235 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), randrange_1228, *[len_call_result_1233], **kwargs_1234)
        
        # Assigning a type to the variable 'i' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'i', randrange_call_result_1235)
        
        # Assigning a Subscript to a Name (line 385):
        
        # Assigning a Subscript to a Name (line 385):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 385)
        i_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 34), 'i')
        # Getting the type of 'self' (line 385)
        self_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'self')
        # Obtaining the member 'unexplored' of a type (line 385)
        unexplored_1238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), self_1237, 'unexplored')
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___1239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), unexplored_1238, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_1240 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), getitem___1239, i_1236)
        
        # Assigning a type to the variable 'pos' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'pos', subscript_call_result_1240)
        
        # Assigning a Subscript to a Subscript (line 386):
        
        # Assigning a Subscript to a Subscript (line 386):
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'self' (line 386)
        self_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 53), 'self', False)
        # Obtaining the member 'unexplored' of a type (line 386)
        unexplored_1243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 53), self_1242, 'unexplored')
        # Processing the call keyword arguments (line 386)
        kwargs_1244 = {}
        # Getting the type of 'len' (line 386)
        len_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), 'len', False)
        # Calling len(args, kwargs) (line 386)
        len_call_result_1245 = invoke(stypy.reporting.localization.Localization(__file__, 386, 49), len_1241, *[unexplored_1243], **kwargs_1244)
        
        int_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 72), 'int')
        # Applying the binary operator '-' (line 386)
        result_sub_1247 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 49), '-', len_call_result_1245, int_1246)
        
        # Getting the type of 'self' (line 386)
        self_1248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 33), 'self')
        # Obtaining the member 'unexplored' of a type (line 386)
        unexplored_1249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 33), self_1248, 'unexplored')
        # Obtaining the member '__getitem__' of a type (line 386)
        getitem___1250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 33), unexplored_1249, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 386)
        subscript_call_result_1251 = invoke(stypy.reporting.localization.Localization(__file__, 386, 33), getitem___1250, result_sub_1247)
        
        # Getting the type of 'self' (line 386)
        self_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'self')
        # Obtaining the member 'unexplored' of a type (line 386)
        unexplored_1253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), self_1252, 'unexplored')
        # Getting the type of 'i' (line 386)
        i_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 28), 'i')
        # Storing an element on a container (line 386)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), unexplored_1253, (i_1254, subscript_call_result_1251))
        
        # Call to pop(...): (line 387)
        # Processing the call keyword arguments (line 387)
        kwargs_1258 = {}
        # Getting the type of 'self' (line 387)
        self_1255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'self', False)
        # Obtaining the member 'unexplored' of a type (line 387)
        unexplored_1256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), self_1255, 'unexplored')
        # Obtaining the member 'pop' of a type (line 387)
        pop_1257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), unexplored_1256, 'pop')
        # Calling pop(args, kwargs) (line 387)
        pop_call_result_1259 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), pop_1257, *[], **kwargs_1258)
        
        # Getting the type of 'pos' (line 388)
        pos_1260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'pos')
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'stypy_return_type', pos_1260)
        # SSA branch for the else part of an if statement (line 383)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 389)
        self_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'self')
        # Obtaining the member 'bestchild' of a type (line 389)
        bestchild_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 13), self_1261, 'bestchild')
        # Testing the type of an if condition (line 389)
        if_condition_1263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 13), bestchild_1262)
        # Assigning a type to the variable 'if_condition_1263' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'if_condition_1263', if_condition_1263)
        # SSA begins for if statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 390)
        self_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'self')
        # Obtaining the member 'bestchild' of a type (line 390)
        bestchild_1265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), self_1264, 'bestchild')
        # Obtaining the member 'pos' of a type (line 390)
        pos_1266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), bestchild_1265, 'pos')
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'stypy_return_type', pos_1266)
        # SSA branch for the else part of an if statement (line 389)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'PASS' (line 392)
        PASS_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'PASS')
        # Assigning a type to the variable 'stypy_return_type' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'stypy_return_type', PASS_1267)
        # SSA join for if statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'select(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'select' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1268)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'select'
        return stypy_return_type_1268


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

        str_1269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'str', ' random play until both players pass ')
        
        
        # Call to range(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'MAXMOVES' (line 396)
        MAXMOVES_1271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'MAXMOVES', False)
        # Processing the call keyword arguments (line 396)
        kwargs_1272 = {}
        # Getting the type of 'range' (line 396)
        range_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'range', False)
        # Calling range(args, kwargs) (line 396)
        range_call_result_1273 = invoke(stypy.reporting.localization.Localization(__file__, 396, 17), range_1270, *[MAXMOVES_1271], **kwargs_1272)
        
        # Testing if the loop is going to be iterated (line 396)
        # Testing the type of a for loop iterable (line 396)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 396, 8), range_call_result_1273)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 396, 8), range_call_result_1273):
            # Getting the type of the for loop variable (line 396)
            for_loop_var_1274 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 396, 8), range_call_result_1273)
            # Assigning a type to the variable 'x' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'x', for_loop_var_1274)
            # SSA begins for a for statement (line 396)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'board' (line 397)
            board_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'board')
            # Obtaining the member 'finished' of a type (line 397)
            finished_1276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 15), board_1275, 'finished')
            # Testing the type of an if condition (line 397)
            if_condition_1277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 12), finished_1276)
            # Assigning a type to the variable 'if_condition_1277' (line 397)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'if_condition_1277', if_condition_1277)
            # SSA begins for if statement (line 397)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 397)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 399):
            
            # Assigning a Name to a Name (line 399):
            # Getting the type of 'PASS' (line 399)
            PASS_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 18), 'PASS')
            # Assigning a type to the variable 'pos' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'pos', PASS_1278)
            
            # Getting the type of 'board' (line 400)
            board_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'board')
            # Obtaining the member 'atari' of a type (line 400)
            atari_1280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 15), board_1279, 'atari')
            # Testing the type of an if condition (line 400)
            if_condition_1281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 12), atari_1280)
            # Assigning a type to the variable 'if_condition_1281' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'if_condition_1281', if_condition_1281)
            # SSA begins for if statement (line 400)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 401):
            
            # Assigning a Call to a Name (line 401):
            
            # Call to liberty(...): (line 401)
            # Processing the call keyword arguments (line 401)
            kwargs_1285 = {}
            # Getting the type of 'board' (line 401)
            board_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 26), 'board', False)
            # Obtaining the member 'atari' of a type (line 401)
            atari_1283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 26), board_1282, 'atari')
            # Obtaining the member 'liberty' of a type (line 401)
            liberty_1284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 26), atari_1283, 'liberty')
            # Calling liberty(args, kwargs) (line 401)
            liberty_call_result_1286 = invoke(stypy.reporting.localization.Localization(__file__, 401, 26), liberty_1284, *[], **kwargs_1285)
            
            # Assigning a type to the variable 'liberty' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'liberty', liberty_call_result_1286)
            
            
            # Call to useful(...): (line 402)
            # Processing the call arguments (line 402)
            # Getting the type of 'liberty' (line 402)
            liberty_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 32), 'liberty', False)
            # Obtaining the member 'pos' of a type (line 402)
            pos_1290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 32), liberty_1289, 'pos')
            # Processing the call keyword arguments (line 402)
            kwargs_1291 = {}
            # Getting the type of 'board' (line 402)
            board_1287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 19), 'board', False)
            # Obtaining the member 'useful' of a type (line 402)
            useful_1288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 19), board_1287, 'useful')
            # Calling useful(args, kwargs) (line 402)
            useful_call_result_1292 = invoke(stypy.reporting.localization.Localization(__file__, 402, 19), useful_1288, *[pos_1290], **kwargs_1291)
            
            # Testing the type of an if condition (line 402)
            if_condition_1293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 16), useful_call_result_1292)
            # Assigning a type to the variable 'if_condition_1293' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'if_condition_1293', if_condition_1293)
            # SSA begins for if statement (line 402)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 403):
            
            # Assigning a Attribute to a Name (line 403):
            # Getting the type of 'liberty' (line 403)
            liberty_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 26), 'liberty')
            # Obtaining the member 'pos' of a type (line 403)
            pos_1295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 26), liberty_1294, 'pos')
            # Assigning a type to the variable 'pos' (line 403)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'pos', pos_1295)
            # SSA join for if statement (line 402)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 400)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'pos' (line 404)
            pos_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'pos')
            # Getting the type of 'PASS' (line 404)
            PASS_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 22), 'PASS')
            # Applying the binary operator '==' (line 404)
            result_eq_1298 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 15), '==', pos_1296, PASS_1297)
            
            # Testing the type of an if condition (line 404)
            if_condition_1299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 12), result_eq_1298)
            # Assigning a type to the variable 'if_condition_1299' (line 404)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'if_condition_1299', if_condition_1299)
            # SSA begins for if statement (line 404)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 405):
            
            # Assigning a Call to a Name (line 405):
            
            # Call to random_move(...): (line 405)
            # Processing the call keyword arguments (line 405)
            kwargs_1302 = {}
            # Getting the type of 'board' (line 405)
            board_1300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 22), 'board', False)
            # Obtaining the member 'random_move' of a type (line 405)
            random_move_1301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 22), board_1300, 'random_move')
            # Calling random_move(args, kwargs) (line 405)
            random_move_call_result_1303 = invoke(stypy.reporting.localization.Localization(__file__, 405, 22), random_move_1301, *[], **kwargs_1302)
            
            # Assigning a type to the variable 'pos' (line 405)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'pos', random_move_call_result_1303)
            # SSA join for if statement (line 404)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to move(...): (line 407)
            # Processing the call arguments (line 407)
            # Getting the type of 'pos' (line 407)
            pos_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 23), 'pos', False)
            # Processing the call keyword arguments (line 407)
            kwargs_1307 = {}
            # Getting the type of 'board' (line 407)
            board_1304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'board', False)
            # Obtaining the member 'move' of a type (line 407)
            move_1305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), board_1304, 'move')
            # Calling move(args, kwargs) (line 407)
            move_call_result_1308 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), move_1305, *[pos_1306], **kwargs_1307)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'random_playout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'random_playout' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_1309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1309)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'random_playout'
        return stypy_return_type_1309


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

        str_1310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 8), 'str', ' update win/loss count along path ')
        
        # Assigning a Compare to a Name (line 416):
        
        # Assigning a Compare to a Name (line 416):
        
        
        # Call to score(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'BLACK' (line 416)
        BLACK_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 27), 'BLACK', False)
        # Processing the call keyword arguments (line 416)
        kwargs_1314 = {}
        # Getting the type of 'board' (line 416)
        board_1311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'board', False)
        # Obtaining the member 'score' of a type (line 416)
        score_1312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), board_1311, 'score')
        # Calling score(args, kwargs) (line 416)
        score_call_result_1315 = invoke(stypy.reporting.localization.Localization(__file__, 416, 15), score_1312, *[BLACK_1313], **kwargs_1314)
        
        
        # Call to score(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'WHITE' (line 416)
        WHITE_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 49), 'WHITE', False)
        # Processing the call keyword arguments (line 416)
        kwargs_1319 = {}
        # Getting the type of 'board' (line 416)
        board_1316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 37), 'board', False)
        # Obtaining the member 'score' of a type (line 416)
        score_1317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 37), board_1316, 'score')
        # Calling score(args, kwargs) (line 416)
        score_call_result_1320 = invoke(stypy.reporting.localization.Localization(__file__, 416, 37), score_1317, *[WHITE_1318], **kwargs_1319)
        
        # Applying the binary operator '>=' (line 416)
        result_ge_1321 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 15), '>=', score_call_result_1315, score_call_result_1320)
        
        # Assigning a type to the variable 'wins' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'wins', result_ge_1321)
        
        # Getting the type of 'path' (line 417)
        path_1322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'path')
        # Testing if the loop is going to be iterated (line 417)
        # Testing the type of a for loop iterable (line 417)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 417, 8), path_1322)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 417, 8), path_1322):
            # Getting the type of the for loop variable (line 417)
            for_loop_var_1323 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 417, 8), path_1322)
            # Assigning a type to the variable 'node' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'node', for_loop_var_1323)
            # SSA begins for a for statement (line 417)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'color' (line 418)
            color_1324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'color')
            # Getting the type of 'BLACK' (line 418)
            BLACK_1325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'BLACK')
            # Applying the binary operator '==' (line 418)
            result_eq_1326 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 15), '==', color_1324, BLACK_1325)
            
            # Testing the type of an if condition (line 418)
            if_condition_1327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 12), result_eq_1326)
            # Assigning a type to the variable 'if_condition_1327' (line 418)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'if_condition_1327', if_condition_1327)
            # SSA begins for if statement (line 418)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 419):
            
            # Assigning a Name to a Name (line 419):
            # Getting the type of 'WHITE' (line 419)
            WHITE_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'WHITE')
            # Assigning a type to the variable 'color' (line 419)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'color', WHITE_1328)
            # SSA branch for the else part of an if statement (line 418)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 421):
            
            # Assigning a Name to a Name (line 421):
            # Getting the type of 'BLACK' (line 421)
            BLACK_1329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 24), 'BLACK')
            # Assigning a type to the variable 'color' (line 421)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'color', BLACK_1329)
            # SSA join for if statement (line 418)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'wins' (line 422)
            wins_1330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'wins')
            
            # Getting the type of 'color' (line 422)
            color_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'color')
            # Getting the type of 'BLACK' (line 422)
            BLACK_1332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 33), 'BLACK')
            # Applying the binary operator '==' (line 422)
            result_eq_1333 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 24), '==', color_1331, BLACK_1332)
            
            # Applying the binary operator '==' (line 422)
            result_eq_1334 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 15), '==', wins_1330, result_eq_1333)
            
            # Testing the type of an if condition (line 422)
            if_condition_1335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 12), result_eq_1334)
            # Assigning a type to the variable 'if_condition_1335' (line 422)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'if_condition_1335', if_condition_1335)
            # SSA begins for if statement (line 422)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'node' (line 423)
            node_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'node')
            # Obtaining the member 'wins' of a type (line 423)
            wins_1337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), node_1336, 'wins')
            int_1338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 29), 'int')
            # Applying the binary operator '+=' (line 423)
            result_iadd_1339 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 16), '+=', wins_1337, int_1338)
            # Getting the type of 'node' (line 423)
            node_1340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'node')
            # Setting the type of the member 'wins' of a type (line 423)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), node_1340, 'wins', result_iadd_1339)
            
            # SSA branch for the else part of an if statement (line 422)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'node' (line 425)
            node_1341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'node')
            # Obtaining the member 'losses' of a type (line 425)
            losses_1342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), node_1341, 'losses')
            int_1343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 31), 'int')
            # Applying the binary operator '+=' (line 425)
            result_iadd_1344 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 16), '+=', losses_1342, int_1343)
            # Getting the type of 'node' (line 425)
            node_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'node')
            # Setting the type of the member 'losses' of a type (line 425)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), node_1345, 'losses', result_iadd_1344)
            
            # SSA join for if statement (line 422)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'node' (line 426)
            node_1346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'node')
            # Obtaining the member 'parent' of a type (line 426)
            parent_1347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 15), node_1346, 'parent')
            # Testing the type of an if condition (line 426)
            if_condition_1348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 12), parent_1347)
            # Assigning a type to the variable 'if_condition_1348' (line 426)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'if_condition_1348', if_condition_1348)
            # SSA begins for if statement (line 426)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to range(...): (line 427)
            # Processing the call arguments (line 427)
            # Getting the type of 'histpos' (line 427)
            histpos_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'histpos', False)
            int_1351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 41), 'int')
            # Applying the binary operator '+' (line 427)
            result_add_1352 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 31), '+', histpos_1350, int_1351)
            
            
            # Call to len(...): (line 427)
            # Processing the call arguments (line 427)
            # Getting the type of 'board' (line 427)
            board_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 48), 'board', False)
            # Obtaining the member 'history' of a type (line 427)
            history_1355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 48), board_1354, 'history')
            # Processing the call keyword arguments (line 427)
            kwargs_1356 = {}
            # Getting the type of 'len' (line 427)
            len_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 44), 'len', False)
            # Calling len(args, kwargs) (line 427)
            len_call_result_1357 = invoke(stypy.reporting.localization.Localization(__file__, 427, 44), len_1353, *[history_1355], **kwargs_1356)
            
            int_1358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 64), 'int')
            # Processing the call keyword arguments (line 427)
            kwargs_1359 = {}
            # Getting the type of 'range' (line 427)
            range_1349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 25), 'range', False)
            # Calling range(args, kwargs) (line 427)
            range_call_result_1360 = invoke(stypy.reporting.localization.Localization(__file__, 427, 25), range_1349, *[result_add_1352, len_call_result_1357, int_1358], **kwargs_1359)
            
            # Testing if the loop is going to be iterated (line 427)
            # Testing the type of a for loop iterable (line 427)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 427, 16), range_call_result_1360)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 427, 16), range_call_result_1360):
                # Getting the type of the for loop variable (line 427)
                for_loop_var_1361 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 427, 16), range_call_result_1360)
                # Assigning a type to the variable 'i' (line 427)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'i', for_loop_var_1361)
                # SSA begins for a for statement (line 427)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 428):
                
                # Assigning a Subscript to a Name (line 428):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 428)
                i_1362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 40), 'i')
                # Getting the type of 'board' (line 428)
                board_1363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'board')
                # Obtaining the member 'history' of a type (line 428)
                history_1364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 26), board_1363, 'history')
                # Obtaining the member '__getitem__' of a type (line 428)
                getitem___1365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 26), history_1364, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 428)
                subscript_call_result_1366 = invoke(stypy.reporting.localization.Localization(__file__, 428, 26), getitem___1365, i_1362)
                
                # Assigning a type to the variable 'pos' (line 428)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'pos', subscript_call_result_1366)
                
                
                # Getting the type of 'pos' (line 429)
                pos_1367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'pos')
                # Getting the type of 'PASS' (line 429)
                PASS_1368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'PASS')
                # Applying the binary operator '==' (line 429)
                result_eq_1369 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 23), '==', pos_1367, PASS_1368)
                
                # Testing the type of an if condition (line 429)
                if_condition_1370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 429, 20), result_eq_1369)
                # Assigning a type to the variable 'if_condition_1370' (line 429)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'if_condition_1370', if_condition_1370)
                # SSA begins for if statement (line 429)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 429)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                # Getting the type of 'wins' (line 431)
                wins_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'wins')
                
                # Getting the type of 'color' (line 431)
                color_1372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 32), 'color')
                # Getting the type of 'BLACK' (line 431)
                BLACK_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 41), 'BLACK')
                # Applying the binary operator '==' (line 431)
                result_eq_1374 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 32), '==', color_1372, BLACK_1373)
                
                # Applying the binary operator '==' (line 431)
                result_eq_1375 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 23), '==', wins_1371, result_eq_1374)
                
                # Testing the type of an if condition (line 431)
                if_condition_1376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 20), result_eq_1375)
                # Assigning a type to the variable 'if_condition_1376' (line 431)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 20), 'if_condition_1376', if_condition_1376)
                # SSA begins for if statement (line 431)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'node' (line 432)
                node_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'node')
                # Obtaining the member 'parent' of a type (line 432)
                parent_1378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), node_1377, 'parent')
                # Obtaining the member 'pos_amaf_wins' of a type (line 432)
                pos_amaf_wins_1379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), parent_1378, 'pos_amaf_wins')
                
                # Obtaining the type of the subscript
                # Getting the type of 'pos' (line 432)
                pos_1380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'pos')
                # Getting the type of 'node' (line 432)
                node_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'node')
                # Obtaining the member 'parent' of a type (line 432)
                parent_1382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), node_1381, 'parent')
                # Obtaining the member 'pos_amaf_wins' of a type (line 432)
                pos_amaf_wins_1383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), parent_1382, 'pos_amaf_wins')
                # Obtaining the member '__getitem__' of a type (line 432)
                getitem___1384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), pos_amaf_wins_1383, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 432)
                subscript_call_result_1385 = invoke(stypy.reporting.localization.Localization(__file__, 432, 24), getitem___1384, pos_1380)
                
                int_1386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 58), 'int')
                # Applying the binary operator '+=' (line 432)
                result_iadd_1387 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 24), '+=', subscript_call_result_1385, int_1386)
                # Getting the type of 'node' (line 432)
                node_1388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'node')
                # Obtaining the member 'parent' of a type (line 432)
                parent_1389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), node_1388, 'parent')
                # Obtaining the member 'pos_amaf_wins' of a type (line 432)
                pos_amaf_wins_1390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), parent_1389, 'pos_amaf_wins')
                # Getting the type of 'pos' (line 432)
                pos_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'pos')
                # Storing an element on a container (line 432)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 24), pos_amaf_wins_1390, (pos_1391, result_iadd_1387))
                
                # SSA branch for the else part of an if statement (line 431)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'node' (line 434)
                node_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                # Obtaining the member 'parent' of a type (line 434)
                parent_1393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1392, 'parent')
                # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                pos_amaf_losses_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1393, 'pos_amaf_losses')
                
                # Obtaining the type of the subscript
                # Getting the type of 'pos' (line 434)
                pos_1395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'pos')
                # Getting the type of 'node' (line 434)
                node_1396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                # Obtaining the member 'parent' of a type (line 434)
                parent_1397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1396, 'parent')
                # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                pos_amaf_losses_1398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1397, 'pos_amaf_losses')
                # Obtaining the member '__getitem__' of a type (line 434)
                getitem___1399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), pos_amaf_losses_1398, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 434)
                subscript_call_result_1400 = invoke(stypy.reporting.localization.Localization(__file__, 434, 24), getitem___1399, pos_1395)
                
                int_1401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 60), 'int')
                # Applying the binary operator '+=' (line 434)
                result_iadd_1402 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 24), '+=', subscript_call_result_1400, int_1401)
                # Getting the type of 'node' (line 434)
                node_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'node')
                # Obtaining the member 'parent' of a type (line 434)
                parent_1404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), node_1403, 'parent')
                # Obtaining the member 'pos_amaf_losses' of a type (line 434)
                pos_amaf_losses_1405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), parent_1404, 'pos_amaf_losses')
                # Getting the type of 'pos' (line 434)
                pos_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'pos')
                # Storing an element on a container (line 434)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 24), pos_amaf_losses_1405, (pos_1406, result_iadd_1402))
                
                # SSA join for if statement (line 431)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Getting the type of 'node' (line 435)
                node_1407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'node')
                # Obtaining the member 'parent' of a type (line 435)
                parent_1408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), node_1407, 'parent')
                # Obtaining the member 'amafvisits' of a type (line 435)
                amafvisits_1409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), parent_1408, 'amafvisits')
                int_1410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 46), 'int')
                # Applying the binary operator '+=' (line 435)
                result_iadd_1411 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 20), '+=', amafvisits_1409, int_1410)
                # Getting the type of 'node' (line 435)
                node_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'node')
                # Obtaining the member 'parent' of a type (line 435)
                parent_1413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), node_1412, 'parent')
                # Setting the type of the member 'amafvisits' of a type (line 435)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), parent_1413, 'amafvisits', result_iadd_1411)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Attribute (line 436):
            
            # Assigning a Call to a Attribute (line 436):
            
            # Call to best_child(...): (line 436)
            # Processing the call keyword arguments (line 436)
            kwargs_1417 = {}
            # Getting the type of 'node' (line 436)
            node_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 40), 'node', False)
            # Obtaining the member 'parent' of a type (line 436)
            parent_1415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 40), node_1414, 'parent')
            # Obtaining the member 'best_child' of a type (line 436)
            best_child_1416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 40), parent_1415, 'best_child')
            # Calling best_child(args, kwargs) (line 436)
            best_child_call_result_1418 = invoke(stypy.reporting.localization.Localization(__file__, 436, 40), best_child_1416, *[], **kwargs_1417)
            
            # Getting the type of 'node' (line 436)
            node_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'node')
            # Obtaining the member 'parent' of a type (line 436)
            parent_1420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), node_1419, 'parent')
            # Setting the type of the member 'bestchild' of a type (line 436)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), parent_1420, 'bestchild', best_child_call_result_1418)
            # SSA join for if statement (line 426)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'update_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_path' in the type store
        # Getting the type of 'stypy_return_type' (line 414)
        stypy_return_type_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1421)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_path'
        return stypy_return_type_1421


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
        self_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'self')
        # Obtaining the member 'wins' of a type (line 439)
        wins_1423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 18), self_1422, 'wins')
        
        # Call to float(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'self' (line 439)
        self_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 36), 'self', False)
        # Obtaining the member 'wins' of a type (line 439)
        wins_1426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 36), self_1425, 'wins')
        # Getting the type of 'self' (line 439)
        self_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 48), 'self', False)
        # Obtaining the member 'losses' of a type (line 439)
        losses_1428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 48), self_1427, 'losses')
        # Applying the binary operator '+' (line 439)
        result_add_1429 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 36), '+', wins_1426, losses_1428)
        
        # Processing the call keyword arguments (line 439)
        kwargs_1430 = {}
        # Getting the type of 'float' (line 439)
        float_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 30), 'float', False)
        # Calling float(args, kwargs) (line 439)
        float_call_result_1431 = invoke(stypy.reporting.localization.Localization(__file__, 439, 30), float_1424, *[result_add_1429], **kwargs_1430)
        
        # Applying the binary operator 'div' (line 439)
        result_div_1432 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 18), 'div', wins_1423, float_call_result_1431)
        
        # Assigning a type to the variable 'winrate' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'winrate', result_div_1432)
        
        # Assigning a BinOp to a Name (line 440):
        
        # Assigning a BinOp to a Name (line 440):
        # Getting the type of 'self' (line 440)
        self_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'self')
        # Obtaining the member 'parent' of a type (line 440)
        parent_1434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), self_1433, 'parent')
        # Obtaining the member 'wins' of a type (line 440)
        wins_1435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), parent_1434, 'wins')
        # Getting the type of 'self' (line 440)
        self_1436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 42), 'self')
        # Obtaining the member 'parent' of a type (line 440)
        parent_1437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 42), self_1436, 'parent')
        # Obtaining the member 'losses' of a type (line 440)
        losses_1438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 42), parent_1437, 'losses')
        # Applying the binary operator '+' (line 440)
        result_add_1439 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 23), '+', wins_1435, losses_1438)
        
        # Assigning a type to the variable 'parentvisits' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'parentvisits', result_add_1439)
        
        
        # Getting the type of 'parentvisits' (line 441)
        parentvisits_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 15), 'parentvisits')
        # Applying the 'not' unary operator (line 441)
        result_not__1441 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 11), 'not', parentvisits_1440)
        
        # Testing the type of an if condition (line 441)
        if_condition_1442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 8), result_not__1441)
        # Assigning a type to the variable 'if_condition_1442' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'if_condition_1442', if_condition_1442)
        # SSA begins for if statement (line 441)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'winrate' (line 442)
        winrate_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 'winrate')
        # Assigning a type to the variable 'stypy_return_type' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'stypy_return_type', winrate_1443)
        # SSA join for if statement (line 441)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 443):
        
        # Assigning a BinOp to a Name (line 443):
        # Getting the type of 'self' (line 443)
        self_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 21), 'self')
        # Obtaining the member 'wins' of a type (line 443)
        wins_1445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 21), self_1444, 'wins')
        # Getting the type of 'self' (line 443)
        self_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 33), 'self')
        # Obtaining the member 'losses' of a type (line 443)
        losses_1447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 33), self_1446, 'losses')
        # Applying the binary operator '+' (line 443)
        result_add_1448 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 21), '+', wins_1445, losses_1447)
        
        # Assigning a type to the variable 'nodevisits' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'nodevisits', result_add_1448)
        
        # Assigning a BinOp to a Name (line 444):
        
        # Assigning a BinOp to a Name (line 444):
        # Getting the type of 'winrate' (line 444)
        winrate_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'winrate')
        
        # Call to sqrt(...): (line 444)
        # Processing the call arguments (line 444)
        
        # Call to log(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'parentvisits' (line 444)
        parentvisits_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 50), 'parentvisits', False)
        # Processing the call keyword arguments (line 444)
        kwargs_1455 = {}
        # Getting the type of 'math' (line 444)
        math_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 41), 'math', False)
        # Obtaining the member 'log' of a type (line 444)
        log_1453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 41), math_1452, 'log')
        # Calling log(args, kwargs) (line 444)
        log_call_result_1456 = invoke(stypy.reporting.localization.Localization(__file__, 444, 41), log_1453, *[parentvisits_1454], **kwargs_1455)
        
        int_1457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 68), 'int')
        # Getting the type of 'nodevisits' (line 444)
        nodevisits_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 72), 'nodevisits', False)
        # Applying the binary operator '*' (line 444)
        result_mul_1459 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 68), '*', int_1457, nodevisits_1458)
        
        # Applying the binary operator 'div' (line 444)
        result_div_1460 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 40), 'div', log_call_result_1456, result_mul_1459)
        
        # Processing the call keyword arguments (line 444)
        kwargs_1461 = {}
        # Getting the type of 'math' (line 444)
        math_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 30), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 444)
        sqrt_1451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 30), math_1450, 'sqrt')
        # Calling sqrt(args, kwargs) (line 444)
        sqrt_call_result_1462 = invoke(stypy.reporting.localization.Localization(__file__, 444, 30), sqrt_1451, *[result_div_1460], **kwargs_1461)
        
        # Applying the binary operator '+' (line 444)
        result_add_1463 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), '+', winrate_1449, sqrt_call_result_1462)
        
        # Assigning a type to the variable 'uct_score' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'uct_score', result_add_1463)
        
        # Assigning a BinOp to a Name (line 446):
        
        # Assigning a BinOp to a Name (line 446):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 446)
        self_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 47), 'self')
        # Obtaining the member 'pos' of a type (line 446)
        pos_1465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 47), self_1464, 'pos')
        # Getting the type of 'self' (line 446)
        self_1466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 21), 'self')
        # Obtaining the member 'parent' of a type (line 446)
        parent_1467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), self_1466, 'parent')
        # Obtaining the member 'pos_amaf_wins' of a type (line 446)
        pos_amaf_wins_1468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), parent_1467, 'pos_amaf_wins')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___1469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), pos_amaf_wins_1468, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_1470 = invoke(stypy.reporting.localization.Localization(__file__, 446, 21), getitem___1469, pos_1465)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 446)
        self_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 87), 'self')
        # Obtaining the member 'pos' of a type (line 446)
        pos_1472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 87), self_1471, 'pos')
        # Getting the type of 'self' (line 446)
        self_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 59), 'self')
        # Obtaining the member 'parent' of a type (line 446)
        parent_1474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 59), self_1473, 'parent')
        # Obtaining the member 'pos_amaf_losses' of a type (line 446)
        pos_amaf_losses_1475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 59), parent_1474, 'pos_amaf_losses')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___1476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 59), pos_amaf_losses_1475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_1477 = invoke(stypy.reporting.localization.Localization(__file__, 446, 59), getitem___1476, pos_1472)
        
        # Applying the binary operator '+' (line 446)
        result_add_1478 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 21), '+', subscript_call_result_1470, subscript_call_result_1477)
        
        # Assigning a type to the variable 'amafvisits' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'amafvisits', result_add_1478)
        
        
        # Getting the type of 'amafvisits' (line 447)
        amafvisits_1479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 15), 'amafvisits')
        # Applying the 'not' unary operator (line 447)
        result_not__1480 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 11), 'not', amafvisits_1479)
        
        # Testing the type of an if condition (line 447)
        if_condition_1481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 8), result_not__1480)
        # Assigning a type to the variable 'if_condition_1481' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'if_condition_1481', if_condition_1481)
        # SSA begins for if statement (line 447)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'uct_score' (line 448)
        uct_score_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 19), 'uct_score')
        # Assigning a type to the variable 'stypy_return_type' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'stypy_return_type', uct_score_1482)
        # SSA join for if statement (line 447)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 449):
        
        # Assigning a BinOp to a Name (line 449):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 449)
        self_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 48), 'self')
        # Obtaining the member 'pos' of a type (line 449)
        pos_1484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 48), self_1483, 'pos')
        # Getting the type of 'self' (line 449)
        self_1485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 22), 'self')
        # Obtaining the member 'parent' of a type (line 449)
        parent_1486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), self_1485, 'parent')
        # Obtaining the member 'pos_amaf_wins' of a type (line 449)
        pos_amaf_wins_1487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), parent_1486, 'pos_amaf_wins')
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___1488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), pos_amaf_wins_1487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_1489 = invoke(stypy.reporting.localization.Localization(__file__, 449, 22), getitem___1488, pos_1484)
        
        
        # Call to float(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'amafvisits' (line 449)
        amafvisits_1491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 66), 'amafvisits', False)
        # Processing the call keyword arguments (line 449)
        kwargs_1492 = {}
        # Getting the type of 'float' (line 449)
        float_1490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 60), 'float', False)
        # Calling float(args, kwargs) (line 449)
        float_call_result_1493 = invoke(stypy.reporting.localization.Localization(__file__, 449, 60), float_1490, *[amafvisits_1491], **kwargs_1492)
        
        # Applying the binary operator 'div' (line 449)
        result_div_1494 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 22), 'div', subscript_call_result_1489, float_call_result_1493)
        
        # Assigning a type to the variable 'amafwinrate' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'amafwinrate', result_div_1494)
        
        # Assigning a BinOp to a Name (line 450):
        
        # Assigning a BinOp to a Name (line 450):
        # Getting the type of 'amafwinrate' (line 450)
        amafwinrate_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'amafwinrate')
        
        # Call to sqrt(...): (line 450)
        # Processing the call arguments (line 450)
        
        # Call to log(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'self' (line 450)
        self_1500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 53), 'self', False)
        # Obtaining the member 'parent' of a type (line 450)
        parent_1501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 53), self_1500, 'parent')
        # Obtaining the member 'amafvisits' of a type (line 450)
        amafvisits_1502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 53), parent_1501, 'amafvisits')
        # Processing the call keyword arguments (line 450)
        kwargs_1503 = {}
        # Getting the type of 'math' (line 450)
        math_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 44), 'math', False)
        # Obtaining the member 'log' of a type (line 450)
        log_1499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 44), math_1498, 'log')
        # Calling log(args, kwargs) (line 450)
        log_call_result_1504 = invoke(stypy.reporting.localization.Localization(__file__, 450, 44), log_1499, *[amafvisits_1502], **kwargs_1503)
        
        int_1505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 81), 'int')
        # Getting the type of 'amafvisits' (line 450)
        amafvisits_1506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 85), 'amafvisits', False)
        # Applying the binary operator '*' (line 450)
        result_mul_1507 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 81), '*', int_1505, amafvisits_1506)
        
        # Applying the binary operator 'div' (line 450)
        result_div_1508 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 43), 'div', log_call_result_1504, result_mul_1507)
        
        # Processing the call keyword arguments (line 450)
        kwargs_1509 = {}
        # Getting the type of 'math' (line 450)
        math_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 33), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 450)
        sqrt_1497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 33), math_1496, 'sqrt')
        # Calling sqrt(args, kwargs) (line 450)
        sqrt_call_result_1510 = invoke(stypy.reporting.localization.Localization(__file__, 450, 33), sqrt_1497, *[result_div_1508], **kwargs_1509)
        
        # Applying the binary operator '+' (line 450)
        result_add_1511 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 19), '+', amafwinrate_1495, sqrt_call_result_1510)
        
        # Assigning a type to the variable 'uct_amaf' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'uct_amaf', result_add_1511)
        
        # Assigning a Call to a Name (line 452):
        
        # Assigning a Call to a Name (line 452):
        
        # Call to sqrt(...): (line 452)
        # Processing the call arguments (line 452)
        float_1514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 25), 'float')
        int_1515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 35), 'int')
        # Getting the type of 'parentvisits' (line 452)
        parentvisits_1516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 39), 'parentvisits', False)
        # Applying the binary operator '*' (line 452)
        result_mul_1517 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 35), '*', int_1515, parentvisits_1516)
        
        float_1518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 54), 'float')
        # Applying the binary operator '+' (line 452)
        result_add_1519 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 35), '+', result_mul_1517, float_1518)
        
        # Applying the binary operator 'div' (line 452)
        result_div_1520 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 25), 'div', float_1514, result_add_1519)
        
        # Processing the call keyword arguments (line 452)
        kwargs_1521 = {}
        # Getting the type of 'math' (line 452)
        math_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 452)
        sqrt_1513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 15), math_1512, 'sqrt')
        # Calling sqrt(args, kwargs) (line 452)
        sqrt_call_result_1522 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), sqrt_1513, *[result_div_1520], **kwargs_1521)
        
        # Assigning a type to the variable 'beta' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'beta', sqrt_call_result_1522)
        # Getting the type of 'beta' (line 453)
        beta_1523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'beta')
        # Getting the type of 'uct_amaf' (line 453)
        uct_amaf_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 22), 'uct_amaf')
        # Applying the binary operator '*' (line 453)
        result_mul_1525 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 15), '*', beta_1523, uct_amaf_1524)
        
        int_1526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 34), 'int')
        # Getting the type of 'beta' (line 453)
        beta_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 38), 'beta')
        # Applying the binary operator '-' (line 453)
        result_sub_1528 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 34), '-', int_1526, beta_1527)
        
        # Getting the type of 'uct_score' (line 453)
        uct_score_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 46), 'uct_score')
        # Applying the binary operator '*' (line 453)
        result_mul_1530 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 33), '*', result_sub_1528, uct_score_1529)
        
        # Applying the binary operator '+' (line 453)
        result_add_1531 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 15), '+', result_mul_1525, result_mul_1530)
        
        # Assigning a type to the variable 'stypy_return_type' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'stypy_return_type', result_add_1531)
        
        # ################# End of 'score(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score' in the type store
        # Getting the type of 'stypy_return_type' (line 438)
        stypy_return_type_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score'
        return stypy_return_type_1532


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
        int_1533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 19), 'int')
        # Assigning a type to the variable 'maxscore' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'maxscore', int_1533)
        
        # Assigning a Name to a Name (line 457):
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'None' (line 457)
        None_1534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'None')
        # Assigning a type to the variable 'maxchild' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'maxchild', None_1534)
        
        # Getting the type of 'self' (line 458)
        self_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 21), 'self')
        # Obtaining the member 'pos_child' of a type (line 458)
        pos_child_1536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 21), self_1535, 'pos_child')
        # Testing if the loop is going to be iterated (line 458)
        # Testing the type of a for loop iterable (line 458)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 458, 8), pos_child_1536)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 458, 8), pos_child_1536):
            # Getting the type of the for loop variable (line 458)
            for_loop_var_1537 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 458, 8), pos_child_1536)
            # Assigning a type to the variable 'child' (line 458)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'child', for_loop_var_1537)
            # SSA begins for a for statement (line 458)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Evaluating a boolean operation
            # Getting the type of 'child' (line 459)
            child_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'child')
            
            
            # Call to score(...): (line 459)
            # Processing the call keyword arguments (line 459)
            kwargs_1541 = {}
            # Getting the type of 'child' (line 459)
            child_1539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'child', False)
            # Obtaining the member 'score' of a type (line 459)
            score_1540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 25), child_1539, 'score')
            # Calling score(args, kwargs) (line 459)
            score_call_result_1542 = invoke(stypy.reporting.localization.Localization(__file__, 459, 25), score_1540, *[], **kwargs_1541)
            
            # Getting the type of 'maxscore' (line 459)
            maxscore_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 41), 'maxscore')
            # Applying the binary operator '>' (line 459)
            result_gt_1544 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 25), '>', score_call_result_1542, maxscore_1543)
            
            # Applying the binary operator 'and' (line 459)
            result_and_keyword_1545 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 15), 'and', child_1538, result_gt_1544)
            
            # Testing the type of an if condition (line 459)
            if_condition_1546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 12), result_and_keyword_1545)
            # Assigning a type to the variable 'if_condition_1546' (line 459)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'if_condition_1546', if_condition_1546)
            # SSA begins for if statement (line 459)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 460):
            
            # Assigning a Name to a Name (line 460):
            # Getting the type of 'child' (line 460)
            child_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 27), 'child')
            # Assigning a type to the variable 'maxchild' (line 460)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'maxchild', child_1547)
            
            # Assigning a Call to a Name (line 461):
            
            # Assigning a Call to a Name (line 461):
            
            # Call to score(...): (line 461)
            # Processing the call keyword arguments (line 461)
            kwargs_1550 = {}
            # Getting the type of 'child' (line 461)
            child_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 27), 'child', False)
            # Obtaining the member 'score' of a type (line 461)
            score_1549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 27), child_1548, 'score')
            # Calling score(args, kwargs) (line 461)
            score_call_result_1551 = invoke(stypy.reporting.localization.Localization(__file__, 461, 27), score_1549, *[], **kwargs_1550)
            
            # Assigning a type to the variable 'maxscore' (line 461)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'maxscore', score_call_result_1551)
            # SSA join for if statement (line 459)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'maxchild' (line 462)
        maxchild_1552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'maxchild')
        # Assigning a type to the variable 'stypy_return_type' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'stypy_return_type', maxchild_1552)
        
        # ################# End of 'best_child(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'best_child' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1553)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'best_child'
        return stypy_return_type_1553


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
        int_1554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 20), 'int')
        # Assigning a type to the variable 'maxvisits' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'maxvisits', int_1554)
        
        # Assigning a Name to a Name (line 466):
        
        # Assigning a Name to a Name (line 466):
        # Getting the type of 'None' (line 466)
        None_1555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 19), 'None')
        # Assigning a type to the variable 'maxchild' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'maxchild', None_1555)
        
        # Getting the type of 'self' (line 467)
        self_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'self')
        # Obtaining the member 'pos_child' of a type (line 467)
        pos_child_1557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 21), self_1556, 'pos_child')
        # Testing if the loop is going to be iterated (line 467)
        # Testing the type of a for loop iterable (line 467)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 467, 8), pos_child_1557)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 467, 8), pos_child_1557):
            # Getting the type of the for loop variable (line 467)
            for_loop_var_1558 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 467, 8), pos_child_1557)
            # Assigning a type to the variable 'child' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'child', for_loop_var_1558)
            # SSA begins for a for statement (line 467)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Evaluating a boolean operation
            # Getting the type of 'child' (line 470)
            child_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'child')
            
            # Getting the type of 'child' (line 470)
            child_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'child')
            # Obtaining the member 'wins' of a type (line 470)
            wins_1561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 26), child_1560, 'wins')
            # Getting the type of 'child' (line 470)
            child_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 39), 'child')
            # Obtaining the member 'losses' of a type (line 470)
            losses_1563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 39), child_1562, 'losses')
            # Applying the binary operator '+' (line 470)
            result_add_1564 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 26), '+', wins_1561, losses_1563)
            
            # Getting the type of 'maxvisits' (line 470)
            maxvisits_1565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 55), 'maxvisits')
            # Applying the binary operator '>' (line 470)
            result_gt_1566 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 25), '>', result_add_1564, maxvisits_1565)
            
            # Applying the binary operator 'and' (line 470)
            result_and_keyword_1567 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 15), 'and', child_1559, result_gt_1566)
            
            # Testing the type of an if condition (line 470)
            if_condition_1568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 12), result_and_keyword_1567)
            # Assigning a type to the variable 'if_condition_1568' (line 470)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'if_condition_1568', if_condition_1568)
            # SSA begins for if statement (line 470)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 471):
            
            # Assigning a BinOp to a Name (line 471):
            # Getting the type of 'child' (line 471)
            child_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 39), 'child')
            # Obtaining the member 'wins' of a type (line 471)
            wins_1570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 39), child_1569, 'wins')
            # Getting the type of 'child' (line 471)
            child_1571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 52), 'child')
            # Obtaining the member 'losses' of a type (line 471)
            losses_1572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 52), child_1571, 'losses')
            # Applying the binary operator '+' (line 471)
            result_add_1573 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 39), '+', wins_1570, losses_1572)
            
            # Assigning a type to the variable 'tuple_assignment_11' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_11', result_add_1573)
            
            # Assigning a Name to a Name (line 471):
            # Getting the type of 'child' (line 471)
            child_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 67), 'child')
            # Assigning a type to the variable 'tuple_assignment_12' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_12', child_1574)
            
            # Assigning a Name to a Name (line 471):
            # Getting the type of 'tuple_assignment_11' (line 471)
            tuple_assignment_11_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_11')
            # Assigning a type to the variable 'maxvisits' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'maxvisits', tuple_assignment_11_1575)
            
            # Assigning a Name to a Name (line 471):
            # Getting the type of 'tuple_assignment_12' (line 471)
            tuple_assignment_12_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'tuple_assignment_12')
            # Assigning a type to the variable 'maxchild' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 27), 'maxchild', tuple_assignment_12_1576)
            # SSA join for if statement (line 470)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'maxchild' (line 472)
        maxchild_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'maxchild')
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'stypy_return_type', maxchild_1577)
        
        # ################# End of 'best_visited(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'best_visited' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1578)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'best_visited'
        return stypy_return_type_1578


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
    int_1580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 17), 'int')
    int_1581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 20), 'int')
    # Processing the call keyword arguments (line 476)
    kwargs_1582 = {}
    # Getting the type of 'to_pos' (line 476)
    to_pos_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 10), 'to_pos', False)
    # Calling to_pos(args, kwargs) (line 476)
    to_pos_call_result_1583 = invoke(stypy.reporting.localization.Localization(__file__, 476, 10), to_pos_1579, *[int_1580, int_1581], **kwargs_1582)
    
    # Assigning a type to the variable 'pos' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'pos', to_pos_call_result_1583)
    # Getting the type of 'pos' (line 477)
    pos_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'pos')
    # Assigning a type to the variable 'stypy_return_type' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type', pos_1584)
    
    # ################# End of 'user_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'user_move' in the type store
    # Getting the type of 'stypy_return_type' (line 475)
    stypy_return_type_1585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1585)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'user_move'
    return stypy_return_type_1585

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
    kwargs_1588 = {}
    # Getting the type of 'board' (line 498)
    board_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 10), 'board', False)
    # Obtaining the member 'random_move' of a type (line 498)
    random_move_1587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 10), board_1586, 'random_move')
    # Calling random_move(args, kwargs) (line 498)
    random_move_call_result_1589 = invoke(stypy.reporting.localization.Localization(__file__, 498, 10), random_move_1587, *[], **kwargs_1588)
    
    # Assigning a type to the variable 'pos' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'pos', random_move_call_result_1589)
    
    
    # Getting the type of 'pos' (line 499)
    pos_1590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 7), 'pos')
    # Getting the type of 'PASS' (line 499)
    PASS_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 14), 'PASS')
    # Applying the binary operator '==' (line 499)
    result_eq_1592 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 7), '==', pos_1590, PASS_1591)
    
    # Testing the type of an if condition (line 499)
    if_condition_1593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 4), result_eq_1592)
    # Assigning a type to the variable 'if_condition_1593' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'if_condition_1593', if_condition_1593)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'PASS' (line 500)
    PASS_1594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'PASS')
    # Assigning a type to the variable 'stypy_return_type' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'stypy_return_type', PASS_1594)
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 501):
    
    # Assigning a Call to a Name (line 501):
    
    # Call to UCTNode(...): (line 501)
    # Processing the call keyword arguments (line 501)
    kwargs_1596 = {}
    # Getting the type of 'UCTNode' (line 501)
    UCTNode_1595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 11), 'UCTNode', False)
    # Calling UCTNode(args, kwargs) (line 501)
    UCTNode_call_result_1597 = invoke(stypy.reporting.localization.Localization(__file__, 501, 11), UCTNode_1595, *[], **kwargs_1596)
    
    # Assigning a type to the variable 'tree' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'tree', UCTNode_call_result_1597)
    
    # Assigning a Call to a Attribute (line 502):
    
    # Assigning a Call to a Attribute (line 502):
    
    # Call to useful_moves(...): (line 502)
    # Processing the call keyword arguments (line 502)
    kwargs_1600 = {}
    # Getting the type of 'board' (line 502)
    board_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'board', False)
    # Obtaining the member 'useful_moves' of a type (line 502)
    useful_moves_1599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 22), board_1598, 'useful_moves')
    # Calling useful_moves(args, kwargs) (line 502)
    useful_moves_call_result_1601 = invoke(stypy.reporting.localization.Localization(__file__, 502, 22), useful_moves_1599, *[], **kwargs_1600)
    
    # Getting the type of 'tree' (line 502)
    tree_1602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'tree')
    # Setting the type of the member 'unexplored' of a type (line 502)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 4), tree_1602, 'unexplored', useful_moves_call_result_1601)
    
    # Assigning a Call to a Name (line 503):
    
    # Assigning a Call to a Name (line 503):
    
    # Call to Board(...): (line 503)
    # Processing the call keyword arguments (line 503)
    kwargs_1604 = {}
    # Getting the type of 'Board' (line 503)
    Board_1603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 13), 'Board', False)
    # Calling Board(args, kwargs) (line 503)
    Board_call_result_1605 = invoke(stypy.reporting.localization.Localization(__file__, 503, 13), Board_1603, *[], **kwargs_1604)
    
    # Assigning a type to the variable 'nboard' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'nboard', Board_call_result_1605)
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to min(...): (line 504)
    # Processing the call arguments (line 504)
    int_1607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 16), 'int')
    int_1608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 25), 'int')
    
    # Call to len(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'board' (line 504)
    board_1610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'board', False)
    # Obtaining the member 'history' of a type (line 504)
    history_1611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 36), board_1610, 'history')
    # Processing the call keyword arguments (line 504)
    kwargs_1612 = {}
    # Getting the type of 'len' (line 504)
    len_1609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'len', False)
    # Calling len(args, kwargs) (line 504)
    len_call_result_1613 = invoke(stypy.reporting.localization.Localization(__file__, 504, 32), len_1609, *[history_1611], **kwargs_1612)
    
    # Applying the binary operator '*' (line 504)
    result_mul_1614 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 25), '*', int_1608, len_call_result_1613)
    
    int_1615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 54), 'int')
    # Applying the binary operator 'div' (line 504)
    result_div_1616 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 24), 'div', result_mul_1614, int_1615)
    
    # Applying the binary operator '-' (line 504)
    result_sub_1617 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 16), '-', int_1607, result_div_1616)
    
    int_1618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 57), 'int')
    # Processing the call keyword arguments (line 504)
    kwargs_1619 = {}
    # Getting the type of 'min' (line 504)
    min_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'min', False)
    # Calling min(args, kwargs) (line 504)
    min_call_result_1620 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), min_1606, *[result_sub_1617, int_1618], **kwargs_1619)
    
    # Assigning a type to the variable 'GAMES' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'GAMES', min_call_result_1620)
    
    
    # Call to range(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'GAMES' (line 506)
    GAMES_1622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 22), 'GAMES', False)
    # Processing the call keyword arguments (line 506)
    kwargs_1623 = {}
    # Getting the type of 'range' (line 506)
    range_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'range', False)
    # Calling range(args, kwargs) (line 506)
    range_call_result_1624 = invoke(stypy.reporting.localization.Localization(__file__, 506, 16), range_1621, *[GAMES_1622], **kwargs_1623)
    
    # Testing if the loop is going to be iterated (line 506)
    # Testing the type of a for loop iterable (line 506)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 506, 4), range_call_result_1624)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 506, 4), range_call_result_1624):
        # Getting the type of the for loop variable (line 506)
        for_loop_var_1625 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 506, 4), range_call_result_1624)
        # Assigning a type to the variable 'game' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'game', for_loop_var_1625)
        # SSA begins for a for statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 507):
        
        # Assigning a Name to a Name (line 507):
        # Getting the type of 'tree' (line 507)
        tree_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 15), 'tree')
        # Assigning a type to the variable 'node' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'node', tree_1626)
        
        # Call to reset(...): (line 508)
        # Processing the call keyword arguments (line 508)
        kwargs_1629 = {}
        # Getting the type of 'nboard' (line 508)
        nboard_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'nboard', False)
        # Obtaining the member 'reset' of a type (line 508)
        reset_1628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), nboard_1627, 'reset')
        # Calling reset(args, kwargs) (line 508)
        reset_call_result_1630 = invoke(stypy.reporting.localization.Localization(__file__, 508, 8), reset_1628, *[], **kwargs_1629)
        
        
        # Call to replay(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'board' (line 509)
        board_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 22), 'board', False)
        # Obtaining the member 'history' of a type (line 509)
        history_1634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 22), board_1633, 'history')
        # Processing the call keyword arguments (line 509)
        kwargs_1635 = {}
        # Getting the type of 'nboard' (line 509)
        nboard_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'nboard', False)
        # Obtaining the member 'replay' of a type (line 509)
        replay_1632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), nboard_1631, 'replay')
        # Calling replay(args, kwargs) (line 509)
        replay_call_result_1636 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), replay_1632, *[history_1634], **kwargs_1635)
        
        
        # Call to play(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'nboard' (line 510)
        nboard_1639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 18), 'nboard', False)
        # Processing the call keyword arguments (line 510)
        kwargs_1640 = {}
        # Getting the type of 'node' (line 510)
        node_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'node', False)
        # Obtaining the member 'play' of a type (line 510)
        play_1638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), node_1637, 'play')
        # Calling play(args, kwargs) (line 510)
        play_call_result_1641 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), play_1638, *[nboard_1639], **kwargs_1640)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to best_visited(...): (line 514)
    # Processing the call keyword arguments (line 514)
    kwargs_1644 = {}
    # Getting the type of 'tree' (line 514)
    tree_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'tree', False)
    # Obtaining the member 'best_visited' of a type (line 514)
    best_visited_1643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 11), tree_1642, 'best_visited')
    # Calling best_visited(args, kwargs) (line 514)
    best_visited_call_result_1645 = invoke(stypy.reporting.localization.Localization(__file__, 514, 11), best_visited_1643, *[], **kwargs_1644)
    
    # Obtaining the member 'pos' of a type (line 514)
    pos_1646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 11), best_visited_call_result_1645, 'pos')
    # Assigning a type to the variable 'stypy_return_type' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type', pos_1646)
    
    # ################# End of 'computer_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'computer_move' in the type store
    # Getting the type of 'stypy_return_type' (line 496)
    stypy_return_type_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1647)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'computer_move'
    return stypy_return_type_1647

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
    kwargs_1649 = {}
    # Getting the type of 'Board' (line 518)
    Board_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'Board', False)
    # Calling Board(args, kwargs) (line 518)
    Board_call_result_1650 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), Board_1648, *[], **kwargs_1649)
    
    # Assigning a type to the variable 'board' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'board', Board_call_result_1650)
    
    # Assigning a Num to a Name (line 519):
    
    # Assigning a Num to a Name (line 519):
    int_1651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 15), 'int')
    # Assigning a type to the variable 'maxturns' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'maxturns', int_1651)
    
    # Assigning a Num to a Name (line 520):
    
    # Assigning a Num to a Name (line 520):
    int_1652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
    # Assigning a type to the variable 'turns' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'turns', int_1652)
    
    
    # Getting the type of 'turns' (line 521)
    turns_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 10), 'turns')
    # Getting the type of 'maxturns' (line 521)
    maxturns_1654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 18), 'maxturns')
    # Applying the binary operator '<' (line 521)
    result_lt_1655 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 10), '<', turns_1653, maxturns_1654)
    
    # Testing the type of an if condition (line 521)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 4), result_lt_1655)
    # SSA begins for while statement (line 521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'board' (line 522)
    board_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 11), 'board')
    # Obtaining the member 'lastmove' of a type (line 522)
    lastmove_1657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 11), board_1656, 'lastmove')
    # Getting the type of 'PASS' (line 522)
    PASS_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 29), 'PASS')
    # Applying the binary operator '!=' (line 522)
    result_ne_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 11), '!=', lastmove_1657, PASS_1658)
    
    # Testing the type of an if condition (line 522)
    if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 8), result_ne_1659)
    # Assigning a type to the variable 'if_condition_1660' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'if_condition_1660', if_condition_1660)
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
    board_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 28), 'board', False)
    # Processing the call keyword arguments (line 525)
    kwargs_1663 = {}
    # Getting the type of 'computer_move' (line 525)
    computer_move_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 14), 'computer_move', False)
    # Calling computer_move(args, kwargs) (line 525)
    computer_move_call_result_1664 = invoke(stypy.reporting.localization.Localization(__file__, 525, 14), computer_move_1661, *[board_1662], **kwargs_1663)
    
    # Assigning a type to the variable 'pos' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'pos', computer_move_call_result_1664)
    
    
    # Getting the type of 'pos' (line 526)
    pos_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 11), 'pos')
    # Getting the type of 'PASS' (line 526)
    PASS_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 18), 'PASS')
    # Applying the binary operator '==' (line 526)
    result_eq_1667 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 11), '==', pos_1665, PASS_1666)
    
    # Testing the type of an if condition (line 526)
    if_condition_1668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 526, 8), result_eq_1667)
    # Assigning a type to the variable 'if_condition_1668' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'if_condition_1668', if_condition_1668)
    # SSA begins for if statement (line 526)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 526)
    module_type_store.open_ssa_branch('else')
    
    # Call to to_xy(...): (line 530)
    # Processing the call arguments (line 530)
    # Getting the type of 'pos' (line 530)
    pos_1670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 18), 'pos', False)
    # Processing the call keyword arguments (line 530)
    kwargs_1671 = {}
    # Getting the type of 'to_xy' (line 530)
    to_xy_1669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'to_xy', False)
    # Calling to_xy(args, kwargs) (line 530)
    to_xy_call_result_1672 = invoke(stypy.reporting.localization.Localization(__file__, 530, 12), to_xy_1669, *[pos_1670], **kwargs_1671)
    
    # SSA join for if statement (line 526)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to move(...): (line 531)
    # Processing the call arguments (line 531)
    # Getting the type of 'pos' (line 531)
    pos_1675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 19), 'pos', False)
    # Processing the call keyword arguments (line 531)
    kwargs_1676 = {}
    # Getting the type of 'board' (line 531)
    board_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'board', False)
    # Obtaining the member 'move' of a type (line 531)
    move_1674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), board_1673, 'move')
    # Calling move(args, kwargs) (line 531)
    move_call_result_1677 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), move_1674, *[pos_1675], **kwargs_1676)
    
    
    # Getting the type of 'board' (line 534)
    board_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 11), 'board')
    # Obtaining the member 'finished' of a type (line 534)
    finished_1679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 11), board_1678, 'finished')
    # Testing the type of an if condition (line 534)
    if_condition_1680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 8), finished_1679)
    # Assigning a type to the variable 'if_condition_1680' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'if_condition_1680', if_condition_1680)
    # SSA begins for if statement (line 534)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 534)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'board' (line 536)
    board_1681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'board')
    # Obtaining the member 'lastmove' of a type (line 536)
    lastmove_1682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 11), board_1681, 'lastmove')
    # Getting the type of 'PASS' (line 536)
    PASS_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 29), 'PASS')
    # Applying the binary operator '!=' (line 536)
    result_ne_1684 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), '!=', lastmove_1682, PASS_1683)
    
    # Testing the type of an if condition (line 536)
    if_condition_1685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), result_ne_1684)
    # Assigning a type to the variable 'if_condition_1685' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_1685', if_condition_1685)
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
    board_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 24), 'board', False)
    # Processing the call keyword arguments (line 538)
    kwargs_1688 = {}
    # Getting the type of 'user_move' (line 538)
    user_move_1686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 14), 'user_move', False)
    # Calling user_move(args, kwargs) (line 538)
    user_move_call_result_1689 = invoke(stypy.reporting.localization.Localization(__file__, 538, 14), user_move_1686, *[board_1687], **kwargs_1688)
    
    # Assigning a type to the variable 'pos' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'pos', user_move_call_result_1689)
    
    # Call to move(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'pos' (line 539)
    pos_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 19), 'pos', False)
    # Processing the call keyword arguments (line 539)
    kwargs_1693 = {}
    # Getting the type of 'board' (line 539)
    board_1690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'board', False)
    # Obtaining the member 'move' of a type (line 539)
    move_1691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 8), board_1690, 'move')
    # Calling move(args, kwargs) (line 539)
    move_call_result_1694 = invoke(stypy.reporting.localization.Localization(__file__, 539, 8), move_1691, *[pos_1692], **kwargs_1693)
    
    
    # Getting the type of 'board' (line 541)
    board_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 11), 'board')
    # Obtaining the member 'finished' of a type (line 541)
    finished_1696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 11), board_1695, 'finished')
    # Testing the type of an if condition (line 541)
    if_condition_1697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 8), finished_1696)
    # Assigning a type to the variable 'if_condition_1697' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'if_condition_1697', if_condition_1697)
    # SSA begins for if statement (line 541)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 541)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'turns' (line 543)
    turns_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'turns')
    int_1699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 17), 'int')
    # Applying the binary operator '+=' (line 543)
    result_iadd_1700 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 8), '+=', turns_1698, int_1699)
    # Assigning a type to the variable 'turns' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'turns', result_iadd_1700)
    
    # SSA join for while statement (line 521)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to score(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'WHITE' (line 545)
    WHITE_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'WHITE', False)
    # Processing the call keyword arguments (line 545)
    kwargs_1704 = {}
    # Getting the type of 'board' (line 545)
    board_1701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'board', False)
    # Obtaining the member 'score' of a type (line 545)
    score_1702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 4), board_1701, 'score')
    # Calling score(args, kwargs) (line 545)
    score_call_result_1705 = invoke(stypy.reporting.localization.Localization(__file__, 545, 4), score_1702, *[WHITE_1703], **kwargs_1704)
    
    
    # Call to score(...): (line 547)
    # Processing the call arguments (line 547)
    # Getting the type of 'BLACK' (line 547)
    BLACK_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'BLACK', False)
    # Processing the call keyword arguments (line 547)
    kwargs_1709 = {}
    # Getting the type of 'board' (line 547)
    board_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'board', False)
    # Obtaining the member 'score' of a type (line 547)
    score_1707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 4), board_1706, 'score')
    # Calling score(args, kwargs) (line 547)
    score_call_result_1710 = invoke(stypy.reporting.localization.Localization(__file__, 547, 4), score_1707, *[BLACK_1708], **kwargs_1709)
    
    
    # ################# End of 'versus_cpu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'versus_cpu' in the type store
    # Getting the type of 'stypy_return_type' (line 517)
    stypy_return_type_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1711)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'versus_cpu'
    return stypy_return_type_1711

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
    int_1714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 16), 'int')
    # Processing the call keyword arguments (line 551)
    kwargs_1715 = {}
    # Getting the type of 'random' (line 551)
    random_1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'random', False)
    # Obtaining the member 'seed' of a type (line 551)
    seed_1713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 4), random_1712, 'seed')
    # Calling seed(args, kwargs) (line 551)
    seed_call_result_1716 = invoke(stypy.reporting.localization.Localization(__file__, 551, 4), seed_1713, *[int_1714], **kwargs_1715)
    
    
    
    # SSA begins for try-except statement (line 552)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to versus_cpu(...): (line 553)
    # Processing the call keyword arguments (line 553)
    kwargs_1718 = {}
    # Getting the type of 'versus_cpu' (line 553)
    versus_cpu_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'versus_cpu', False)
    # Calling versus_cpu(args, kwargs) (line 553)
    versus_cpu_call_result_1719 = invoke(stypy.reporting.localization.Localization(__file__, 553, 8), versus_cpu_1717, *[], **kwargs_1718)
    
    # SSA branch for the except part of a try statement (line 552)
    # SSA branch for the except 'EOFError' branch of a try statement (line 552)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 552)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 556)
    True_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'stypy_return_type', True_1720)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 550)
    stypy_return_type_1721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1721)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1721

# Assigning a type to the variable 'run' (line 550)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'run', run)

# Call to run(...): (line 559)
# Processing the call keyword arguments (line 559)
kwargs_1723 = {}
# Getting the type of 'run' (line 559)
run_1722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 0), 'run', False)
# Calling run(args, kwargs) (line 559)
run_call_result_1724 = invoke(stypy.reporting.localization.Localization(__file__, 559, 0), run_1722, *[], **kwargs_1723)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

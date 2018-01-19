
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: # This is an extremely simple chess like speed test program written in Python
3: # This program can be distributed under GNU General Public License Version 2.
4: # (C) Jyrki Alakuijala 2005
5: #
6: # Despite its looks, this program was written in Python, not converted to it.
7: # This program is incomplete, castlings, enpassant situation etc. are not properly implemented
8: # game ending is not recognized. The evaluator as simple as it ever could be. 
9: #
10: # The board is an 160-element array of ints, Nones and Booleans,
11: # The board contains the real board in squares indexed in variable 'squares'
12: # The oversized board is to allow for "0x88" chess programming trick for move generation.
13: # Other board data:
14: # 4x castling flags, indices [10-13], queen side white, king side white, queen side black, king side white
15: # turn, enpassant [26, 27]
16: 
17: from copy import copy
18: 
19: iNone = -999
20: iTrue = 1
21: iFalse = 0
22: 
23: setup = (4, 2, 3, 5, 6, 3, 2, 4, iNone, iNone) + (iTrue,)*4 + (iNone, iNone) +   (1,) * 8 + (iNone, iNone, iTrue, iNone, iNone, iNone, iNone, iNone,) +   ((0, ) * 8 + (iNone,) * 8) * 4 +   (-1,) * 8 + (iNone,) * 8 +   (-4, -2, -3, -5, -6, -3, -2, -4) + (iNone,) * 40
24: 
25: squares = tuple([i for i in range(128) if not i & 8])
26: knightMoves = (-33, -31, -18, -14, 14, 18, 31, 33)
27: bishopLines = (tuple(range(17, 120, 17)), tuple(range(-17, -120, -17)), tuple(range(15, 106, 15)), tuple(range(-15, -106, -15)))
28: rookLines = (tuple(range(1, 8)), tuple(range(-1, -8, -1)), tuple(range(16, 128, 16)), tuple(range(-16, -128, -16)))
29: queenLines = bishopLines + rookLines
30: kingMoves = (-17, -16, -15, -1, 1, 15, 16, 17)
31: 
32: linePieces = ((), (), (), bishopLines, rookLines, queenLines, (), (), queenLines, rookLines, bishopLines, (), ())
33: 
34: clearCastlingOpportunities = [None] * 0x80
35: for (i, v) in ((0x0, (10,)), (0x4, (10, 11)), (0x7, (11,)), (0x70, (12,)), (0x74, (12, 13)), (0x77, (13,))):
36:   clearCastlingOpportunities[i] = v
37: 
38: pieces = ".pnbrqkKQRBNP"
39: 
40: def evaluate(board):
41:   evals = (0, 100, 300, 330, 510, 950, 100000, -100000, -950, -510, -330, -300, -100)
42:   return sum([evals[board[i]] for i in squares])
43: 
44: def printBoard(board):
45:   for i in range(7,-1,-1):
46:     for j in range(8):
47:       ix = i * 16 + j
48:       #print pieces[board[ix]],
49:     #print
50: 
51: def move(board, mv):
52:   ix = (mv >> 8) & 0xff
53:   board[mv & 0xff] = board[ix]
54:   board[ix] = 0
55:   if clearCastlingOpportunities[ix]:
56:     for i in clearCastlingOpportunities[ix]:
57:       board[i] = iFalse
58: 
59:   board[26] = int(not board[26]) # Turn
60:   if (mv & 0x7fff0000) == 0:
61:     return
62:   if (mv & 0x01000000): # double step
63:     board[27] = mv & 7
64:   else:
65:     board[27] = iNone # no enpassant
66:   if (mv & 0x04000000): # castling
67:     toix = mv & 0xff
68:     if toix == 0x02:
69:       board[0x00] = 0
70:       board[0x03] = 4
71:     elif toix == 0x06:
72:       board[0x07] = 0
73:       board[0x05] = 4
74:     elif toix == 0x72:
75:       board[0x70] = 0
76:       board[0x73] = -4
77:     elif toix == 0x76:
78:       board[0x77] = 0
79:       board[0x75] = -4
80:     else:
81:       raise "faulty castling"
82:   if mv & 0x08000000: # enpassant capture
83:     if board[26]: # turn after this move
84:       board[mv & 0x07 + 64] = 0
85:     else:
86:       board[mv & 0x07 + 48] = 0
87:   if mv & 0x10000000: # promotion
88:     a = (mv & 0xff0000) >> 16
89:     if (a >= 0x80):
90:       a = a - 0x100 
91:     board[mv & 0xff] = a
92: 
93: def toString(move):
94:   fr = (move >> 8) & 0xff
95:   to = move & 0xff
96:   letters = "abcdefgh"
97:   numbers = "12345678"
98:   mid = "-"
99:   if (move & 0x04000000):
100:     if (move & 0x7) == 0x02:
101:       return "O-O-O"
102:     else:
103:       return "O-O"
104:   if move & 0x02000000:
105:     mid = "x"
106:   retval = letters[fr & 7] + numbers[fr >> 4] + mid + letters[to & 7] + numbers[to >> 4]
107:   return retval
108: 
109: def moveStr(board, strMove):
110:   moves = pseudoLegalMoves(board)
111:   for m in moves:
112:     if strMove == toString(m):
113:       move(board, m)
114:       return
115:   for m in moves:
116:     pass#print toString(m)
117:     toString(m)
118:   raise "no move found" #, strMove
119: 
120: def rowAttack(board, attackers, ix, dir):
121:   own = attackers[0]
122:   for k in [i + ix for i in dir]:
123:     if k & 0x88:
124:       return False
125:     if board[k]:
126:       return (board[k] * own < 0) and board[k] in attackers
127: 
128: def nonpawnAttacks(board, ix, color):
129:   return (max([board[ix + i] == color * 2 for i in knightMoves]) or 
130:           max([rowAttack(board, (color * 3, color * 5), ix, bishopLine) for bishopLine in bishopLines]) or
131:           max([rowAttack(board, (color * 4, color * 5), ix, rookLine) for rookLine in rookLines]))
132: 
133: nonpawnBlackAttacks = lambda board, ix: nonpawnAttacks(board, ix, -1)
134: nonpawnWhiteAttacks = lambda board, ix: nonpawnAttacks(board, ix, 1)
135: 
136: def pseudoLegalMovesWhite(board):
137:   retval = pseudoLegalCapturesWhite(board)
138:   for sq in squares:
139:     b = board[sq]
140:     if b >= 1:
141:       if b == 1 and not (sq + 16 & 0x88) and board[sq + 16] == 0:
142:         if sq >= 16 and sq < 32 and board[sq + 32] == 0:
143:           retval.append(sq * 0x101 + 32)
144:         retval.append(sq * 0x101 + 16)
145:       elif b == 2:
146:         for k in knightMoves:
147:           if board[k + sq] == 0:
148:             retval.append(sq * 0x101 + k)
149:       elif b == 3 or b == 5:
150:         for line in bishopLines:
151:           for k in line:
152:             if (k + sq & 0x88) or board[k + sq] != 0:
153:               break
154:             retval.append(sq * 0x101 + k)
155:       if b == 4 or b == 5:
156:         for line in rookLines:
157:           for k in line:
158:             if (k + sq & 0x88) or board[k + sq] != 0:
159:               break
160:             retval.append(sq * 0x101 + k)
161:       elif b == 6:
162:         for k in kingMoves:
163:           if not (k + sq & 0x88) and board[k + sq] == 0:
164:             retval.append(sq * 0x101 + k)
165:   if (board[10] and board[1] == 0 and board[2] == 0 and board[3] == 0 and
166:       not -1 in board[17:22] and
167:       not nonpawnBlackAttacks(board, 2) and not nonpawnBlackAttacks(board, 3) and not nonpawnBlackAttacks(board, 4)):
168:     retval.append(0x04000000 + 4 * 0x101 - 2)
169:   if (board[11] and board[5] == 0 and board[6] == 0 and
170:       not -1 in board[19:24] and
171:       not nonpawnBlackAttacks(board, 4) and not nonpawnBlackAttacks(board, 5) and not nonpawnBlackAttacks(board, 6)):
172:     retval.append(0x04000000 + 4 * 0x101 + 2)
173:   return retval
174: 
175: def pseudoLegalMovesBlack(board):
176:   retval = pseudoLegalCapturesBlack(board)
177:   for sq in squares:
178:     b = board[sq]
179:     if b < 0:
180:       if b == -1 and not (sq + 16 & 0x88) and board[sq - 16] == 0:
181:         if sq >= 96 and sq < 112 and board[sq - 32] == 0:
182:           retval.append(sq * 0x101 - 32)
183:         retval.append(sq * 0x101 - 16)
184:       elif b == -2:
185:         for k in knightMoves:
186:           if board[k + sq] == 0:
187:             retval.append(sq * 0x101 + k)
188:       elif b == -3 or b == -5: 
189:         for line in bishopLines:
190:           for k in line:
191:             if (k + sq & 0x88) or board[k + sq] != 0:
192:               break
193:             retval.append(sq * 0x101 + k)
194: 
195:       if b == -4 or b == -5:
196:         for line in rookLines:
197:           for k in line:
198:             if (k + sq & 0x88) or board[k + sq] != 0:
199:               break
200:             retval.append(sq * 0x101 + k)
201:       elif b == -6: 
202:         for k in kingMoves:
203:           if not (k + sq & 0x88) and board[k + sq] == 0:
204:             retval.append(sq * 0x101 + k)
205:   if (board[12] and board[0x71] == 0 and board[0x72] == 0 and board[0x73] == 0 and
206:       not 1 in board[0x61:0x65] and
207:       not nonpawnWhiteAttacks(board, 0x72) and not nonpawnWhiteAttacks(board, 0x73) and not nonpawnWhiteAttacks(board, 0x74)):
208:     retval.append(0x04000000 + 0x74 * 0x101 - 2)
209:   if (board[11] and board[0x75] == 0 and board[0x76] == 0 and
210:       not -1 in board[0x63:0x68] and
211:       not nonpawnWhiteAttacks(board, 0x74) and not nonpawnWhiteAttacks(board, 0x75) and not nonpawnWhiteAttacks(board, 0x76)):
212:     retval.append(0x04000000 + 0x74 * 0x101 + 2)
213:   return retval
214: 
215: def pseudoLegalMoves(board):
216:   if board[26]:
217:     return pseudoLegalMovesWhite(board)
218:   else:
219:     return pseudoLegalMovesBlack(board)
220: 
221: def pseudoLegalCapturesWhite(board):
222:   retval = []
223:   for sq in squares:
224:     b = board[sq]
225:     if b >= 1:
226:       if b == 1: 
227:         if not (sq + 17 & 0x88) and board[sq + 17] < 0:
228:           retval.append(0x02000000 + sq * 0x101 + 17)
229:         if not (sq + 15 & 0x88) and board[sq + 15] < 0:
230:           retval.append(0x02000000 + sq * 0x101 + 15)
231:         if sq >= 64 and sq < 72 and abs((sq & 7) - board[27]) == 1: # enpassant
232:           retval.append(0x02000000 + sq * 0x100 + (sq & 0x70) + 16 + board[27])
233:       elif b == 2:
234:         for k in knightMoves:
235:           if not (sq + k & 0x88) and board[k + sq] < 0:
236:             retval.append(0x02000000 + sq * 0x101 + k)
237:       elif b == 6:
238:         for k in kingMoves:
239:           if not(k + sq & 0x88) and board[k + sq] < 0:
240:             retval.append(0x02000000 + sq * 0x101 + k)
241:       else:
242:         for line in linePieces[b]:
243:           for k in line:
244:             if (sq + k & 0x88) or board[k + sq] >= 1:
245:               break
246:             if board[k + sq] < 0:
247:               retval.append(0x02000000 + sq * 0x101 + k)
248:               break
249:   return retval
250: 
251: def pseudoLegalCapturesBlack(board):
252:   retval = []
253:   for sq in squares:
254:     b = board[sq]
255:     if b < 0:
256:       if b == -1: 
257:         if board[sq - 17] >= 1:
258:           retval.append(0x02000000 + sq * 0x101 - 17)
259:         if board[sq - 15] >= 1:
260:           retval.append(0x02000000 + sq * 0x101 - 15)
261:         if sq >= 48 and sq < 56 and abs((sq & 7) - board[27]) == 1: # enpassant
262:           retval.append(0x0a000000 + sq * 0x100 + (sq & 0x70) - 16 + board[27])
263:       elif b == -2:
264:         for k in knightMoves:
265:           if not (sq + k & 0x88) and board[k + sq] >= 1:
266:             retval.append(0x02000000 + sq * 0x101 + k)
267:       elif b == -3:
268:         for line in bishopLines:
269:           for k in line:
270:             if board[k + sq] < 0:
271:               break
272:             if board[k + sq] >= 1:
273:               retval.append(0x02000000 + sq * 0x101 + k)
274:               break
275:       elif b == -4:
276:         for line in rookLines:
277:           for k in line:
278:             if board[k + sq] < 0:
279:               break
280:             if board[k + sq] >= 1:
281:               retval.append(0x02000000 + sq * 0x101 + k)
282:               break
283:       elif b == -5:
284:         for line in queenLines:
285:           for k in line:
286:             if board[k + sq] < 0:
287:               break
288:             if board[k + sq] >= 1:
289:               retval.append(0x02000000 + sq * 0x101 + k)
290:               break
291:       elif b == -6:
292:         for k in kingMoves:
293:           if board[k + sq] >= 1:
294:             retval.append(0x02000000 + sq * 0x101 + k)
295:   return retval
296: 
297: def pseudoLegalCaptures(board):
298:   if board[26]:
299:     return pseudoLegalCapturesWhite(board)
300:   else:
301:     return pseudoLegalCapturesBlack(board)
302: 
303: def legalMoves(board):
304:   allMoves = pseudoLegalMoves(board)
305:   retval = []
306:   #from copy import copy
307:   kingVal = 6
308:   if board[26]:
309:     kingVal = -kingVal
310:   for mv in allMoves:
311:     board2 = copy(board)
312:     move(board2, mv)
313:     #print "trying to reduce move", toString(mv)
314:     if not [i for i in pseudoLegalCaptures(board2) if board2[i & 0xff] == kingVal]:
315:       retval.append(mv)
316:   return retval
317: 
318: def alphaBetaQui(board, alpha, beta, n):
319:   e = evaluate(board)
320:   if not board[26]:
321:     e = -e
322:   if e >= beta:
323:     return (beta, iNone) # XXX
324:   if (e > alpha): 
325:     alpha = e
326:   bestMove = iNone # XXX
327:   if n >= -4:
328:     #from copy import copy
329:     for mv in pseudoLegalCaptures(board):
330:       newboard = copy(board)
331:       move(newboard, mv)
332:       value = alphaBetaQui(newboard, -beta, -alpha, n - 1)
333:       value = (-value[0], value[1])
334:       if value[0] >= beta:
335:         return (beta, mv)
336:       if (value[0] > alpha):
337:         alpha = value[0]
338:         bestMove = mv
339:   return (alpha, bestMove)
340: 
341: def alphaBeta(board, alpha, beta, n):
342:   if n == 0:
343:     return alphaBetaQui(board, alpha, beta, n)
344: #  from copy import copy
345:   bestMove = iNone # XXX
346: 
347:   for mv in legalMoves(board):
348:     newboard = copy(board)
349:     move(newboard, mv)
350:     value = alphaBeta(newboard, -beta, -alpha, n - 1)
351:     value = (-value[0], value[1])
352:     if value[0] >= beta:
353:       return (beta, mv)
354:     if (value[0] > alpha):
355:       alpha = value[0]
356:       bestMove = mv
357:   return (alpha, bestMove)
358: 
359: def speedTest():
360:   board = list(setup)
361:   moveStr(board, "c2-c4")
362:   moveStr(board, "e7-e5")
363:   moveStr(board, "d2-d4")
364: 
365:   res = alphaBeta(board, -99999999, 99999999, 4)
366:   #print res
367:   moveStr(board, "d7-d6")
368:   res = alphaBeta(board, -99999999, 99999999, 4)
369:   #print res
370: 
371: def run():
372:   speedTest()
373:   return True
374: 
375: run()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from copy import copy' statement (line 17)
try:
    from copy import copy

except:
    copy = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'copy', None, module_type_store, ['copy'], [copy])


# Assigning a Num to a Name (line 19):
int_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'int')
# Assigning a type to the variable 'iNone' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'iNone', int_1)

# Assigning a Num to a Name (line 20):
int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'int')
# Assigning a type to the variable 'iTrue' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'iTrue', int_2)

# Assigning a Num to a Name (line 21):
int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'int')
# Assigning a type to the variable 'iFalse' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'iFalse', int_3)

# Assigning a BinOp to a Name (line 23):

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_5)
# Adding element type (line 23)
int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_6)
# Adding element type (line 23)
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_7)
# Adding element type (line 23)
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_8)
# Adding element type (line 23)
int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_9)
# Adding element type (line 23)
int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_10)
# Adding element type (line 23)
int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_11)
# Adding element type (line 23)
int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, int_12)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 33), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, iNone_13)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 40), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_4, iNone_14)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
# Getting the type of 'iTrue' (line 23)
iTrue_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 50), 'iTrue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 50), tuple_15, iTrue_16)

int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 58), 'int')
# Applying the binary operator '*' (line 23)
result_mul_18 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 49), '*', tuple_15, int_17)

# Applying the binary operator '+' (line 23)
result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 8), '+', tuple_4, result_mul_18)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 63), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 63), tuple_20, iNone_21)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 70), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 63), tuple_20, iNone_22)

# Applying the binary operator '+' (line 23)
result_add_23 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 60), '+', result_add_19, tuple_20)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 82), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 82), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 82), tuple_24, int_25)

int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 88), 'int')
# Applying the binary operator '*' (line 23)
result_mul_27 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 81), '*', tuple_24, int_26)

# Applying the binary operator '+' (line 23)
result_add_28 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 77), '+', result_add_23, result_mul_27)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 93), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 93), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iNone_30)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 100), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iNone_31)
# Adding element type (line 23)
# Getting the type of 'iTrue' (line 23)
iTrue_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 107), 'iTrue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iTrue_32)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 114), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iNone_33)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 121), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iNone_34)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 128), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iNone_35)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 135), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iNone_36)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 142), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 93), tuple_29, iNone_37)

# Applying the binary operator '+' (line 23)
result_add_38 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 90), '+', result_add_28, tuple_29)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 156), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 156), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 156), tuple_39, int_40)

int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 163), 'int')
# Applying the binary operator '*' (line 23)
result_mul_42 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 155), '*', tuple_39, int_41)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 168), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 168), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 168), tuple_43, iNone_44)

int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 178), 'int')
# Applying the binary operator '*' (line 23)
result_mul_46 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 167), '*', tuple_43, int_45)

# Applying the binary operator '+' (line 23)
result_add_47 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 155), '+', result_mul_42, result_mul_46)

int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 183), 'int')
# Applying the binary operator '*' (line 23)
result_mul_49 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 154), '*', result_add_47, int_48)

# Applying the binary operator '+' (line 23)
result_add_50 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 150), '+', result_add_38, result_mul_49)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 190), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 190), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 190), tuple_51, int_52)

int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 197), 'int')
# Applying the binary operator '*' (line 23)
result_mul_54 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 189), '*', tuple_51, int_53)

# Applying the binary operator '+' (line 23)
result_add_55 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 185), '+', result_add_50, result_mul_54)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 202), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 202), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 202), tuple_56, iNone_57)

int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 212), 'int')
# Applying the binary operator '*' (line 23)
result_mul_59 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 201), '*', tuple_56, int_58)

# Applying the binary operator '+' (line 23)
result_add_60 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 199), '+', result_add_55, result_mul_59)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 219), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 219), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_62)
# Adding element type (line 23)
int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 223), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_63)
# Adding element type (line 23)
int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 227), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_64)
# Adding element type (line 23)
int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 231), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_65)
# Adding element type (line 23)
int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 235), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_66)
# Adding element type (line 23)
int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 239), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_67)
# Adding element type (line 23)
int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 243), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_68)
# Adding element type (line 23)
int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 247), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 219), tuple_61, int_69)

# Applying the binary operator '+' (line 23)
result_add_70 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 214), '+', result_add_60, tuple_61)


# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_71 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 254), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
# Getting the type of 'iNone' (line 23)
iNone_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 254), 'iNone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 254), tuple_71, iNone_72)

int_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 264), 'int')
# Applying the binary operator '*' (line 23)
result_mul_74 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 253), '*', tuple_71, int_73)

# Applying the binary operator '+' (line 23)
result_add_75 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 251), '+', result_add_70, result_mul_74)

# Assigning a type to the variable 'setup' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'setup', result_add_75)

# Assigning a Call to a Name (line 25):

# Call to tuple(...): (line 25)
# Processing the call arguments (line 25)
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 25)
# Processing the call arguments (line 25)
int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 34), 'int')
# Processing the call keyword arguments (line 25)
kwargs_84 = {}
# Getting the type of 'range' (line 25)
range_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'range', False)
# Calling range(args, kwargs) (line 25)
range_call_result_85 = invoke(stypy.reporting.localization.Localization(__file__, 25, 28), range_82, *[int_83], **kwargs_84)

comprehension_86 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), range_call_result_85)
# Assigning a type to the variable 'i' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'i', comprehension_86)

# Getting the type of 'i' (line 25)
i_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 46), 'i', False)
int_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 50), 'int')
# Applying the binary operator '&' (line 25)
result_and__80 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 46), '&', i_78, int_79)

# Applying the 'not' unary operator (line 25)
result_not__81 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 42), 'not', result_and__80)

# Getting the type of 'i' (line 25)
i_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'i', False)
list_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), list_87, i_77)
# Processing the call keyword arguments (line 25)
kwargs_88 = {}
# Getting the type of 'tuple' (line 25)
tuple_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'tuple', False)
# Calling tuple(args, kwargs) (line 25)
tuple_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), tuple_76, *[list_87], **kwargs_88)

# Assigning a type to the variable 'squares' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'squares', tuple_call_result_89)

# Assigning a Tuple to a Name (line 26):

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)
# Adding element type (line 26)
int_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_91)
# Adding element type (line 26)
int_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_92)
# Adding element type (line 26)
int_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_93)
# Adding element type (line 26)
int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_94)
# Adding element type (line 26)
int_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_95)
# Adding element type (line 26)
int_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_96)
# Adding element type (line 26)
int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_97)
# Adding element type (line 26)
int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_90, int_98)

# Assigning a type to the variable 'knightMoves' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'knightMoves', tuple_90)

# Assigning a Tuple to a Name (line 27):

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)

# Call to tuple(...): (line 27)
# Processing the call arguments (line 27)

# Call to range(...): (line 27)
# Processing the call arguments (line 27)
int_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 27), 'int')
int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'int')
int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'int')
# Processing the call keyword arguments (line 27)
kwargs_105 = {}
# Getting the type of 'range' (line 27)
range_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'range', False)
# Calling range(args, kwargs) (line 27)
range_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 27, 21), range_101, *[int_102, int_103, int_104], **kwargs_105)

# Processing the call keyword arguments (line 27)
kwargs_107 = {}
# Getting the type of 'tuple' (line 27)
tuple_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'tuple', False)
# Calling tuple(args, kwargs) (line 27)
tuple_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_100, *[range_call_result_106], **kwargs_107)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_99, tuple_call_result_108)
# Adding element type (line 27)

# Call to tuple(...): (line 27)
# Processing the call arguments (line 27)

# Call to range(...): (line 27)
# Processing the call arguments (line 27)
int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 54), 'int')
int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 59), 'int')
int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 65), 'int')
# Processing the call keyword arguments (line 27)
kwargs_114 = {}
# Getting the type of 'range' (line 27)
range_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 48), 'range', False)
# Calling range(args, kwargs) (line 27)
range_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 27, 48), range_110, *[int_111, int_112, int_113], **kwargs_114)

# Processing the call keyword arguments (line 27)
kwargs_116 = {}
# Getting the type of 'tuple' (line 27)
tuple_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 42), 'tuple', False)
# Calling tuple(args, kwargs) (line 27)
tuple_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 27, 42), tuple_109, *[range_call_result_115], **kwargs_116)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_99, tuple_call_result_117)
# Adding element type (line 27)

# Call to tuple(...): (line 27)
# Processing the call arguments (line 27)

# Call to range(...): (line 27)
# Processing the call arguments (line 27)
int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 84), 'int')
int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 88), 'int')
int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 93), 'int')
# Processing the call keyword arguments (line 27)
kwargs_123 = {}
# Getting the type of 'range' (line 27)
range_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 78), 'range', False)
# Calling range(args, kwargs) (line 27)
range_call_result_124 = invoke(stypy.reporting.localization.Localization(__file__, 27, 78), range_119, *[int_120, int_121, int_122], **kwargs_123)

# Processing the call keyword arguments (line 27)
kwargs_125 = {}
# Getting the type of 'tuple' (line 27)
tuple_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 72), 'tuple', False)
# Calling tuple(args, kwargs) (line 27)
tuple_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 27, 72), tuple_118, *[range_call_result_124], **kwargs_125)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_99, tuple_call_result_126)
# Adding element type (line 27)

# Call to tuple(...): (line 27)
# Processing the call arguments (line 27)

# Call to range(...): (line 27)
# Processing the call arguments (line 27)
int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 111), 'int')
int_130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 116), 'int')
int_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 122), 'int')
# Processing the call keyword arguments (line 27)
kwargs_132 = {}
# Getting the type of 'range' (line 27)
range_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 105), 'range', False)
# Calling range(args, kwargs) (line 27)
range_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 27, 105), range_128, *[int_129, int_130, int_131], **kwargs_132)

# Processing the call keyword arguments (line 27)
kwargs_134 = {}
# Getting the type of 'tuple' (line 27)
tuple_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 99), 'tuple', False)
# Calling tuple(args, kwargs) (line 27)
tuple_call_result_135 = invoke(stypy.reporting.localization.Localization(__file__, 27, 99), tuple_127, *[range_call_result_133], **kwargs_134)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_99, tuple_call_result_135)

# Assigning a type to the variable 'bishopLines' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'bishopLines', tuple_99)

# Assigning a Tuple to a Name (line 28):

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)

# Call to tuple(...): (line 28)
# Processing the call arguments (line 28)

# Call to range(...): (line 28)
# Processing the call arguments (line 28)
int_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
int_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
# Processing the call keyword arguments (line 28)
kwargs_141 = {}
# Getting the type of 'range' (line 28)
range_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'range', False)
# Calling range(args, kwargs) (line 28)
range_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 28, 19), range_138, *[int_139, int_140], **kwargs_141)

# Processing the call keyword arguments (line 28)
kwargs_143 = {}
# Getting the type of 'tuple' (line 28)
tuple_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'tuple', False)
# Calling tuple(args, kwargs) (line 28)
tuple_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 28, 13), tuple_137, *[range_call_result_142], **kwargs_143)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 13), tuple_136, tuple_call_result_144)
# Adding element type (line 28)

# Call to tuple(...): (line 28)
# Processing the call arguments (line 28)

# Call to range(...): (line 28)
# Processing the call arguments (line 28)
int_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 45), 'int')
int_148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 49), 'int')
int_149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 53), 'int')
# Processing the call keyword arguments (line 28)
kwargs_150 = {}
# Getting the type of 'range' (line 28)
range_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 39), 'range', False)
# Calling range(args, kwargs) (line 28)
range_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 28, 39), range_146, *[int_147, int_148, int_149], **kwargs_150)

# Processing the call keyword arguments (line 28)
kwargs_152 = {}
# Getting the type of 'tuple' (line 28)
tuple_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 33), 'tuple', False)
# Calling tuple(args, kwargs) (line 28)
tuple_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 28, 33), tuple_145, *[range_call_result_151], **kwargs_152)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 13), tuple_136, tuple_call_result_153)
# Adding element type (line 28)

# Call to tuple(...): (line 28)
# Processing the call arguments (line 28)

# Call to range(...): (line 28)
# Processing the call arguments (line 28)
int_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 71), 'int')
int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 75), 'int')
int_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 80), 'int')
# Processing the call keyword arguments (line 28)
kwargs_159 = {}
# Getting the type of 'range' (line 28)
range_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 65), 'range', False)
# Calling range(args, kwargs) (line 28)
range_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 28, 65), range_155, *[int_156, int_157, int_158], **kwargs_159)

# Processing the call keyword arguments (line 28)
kwargs_161 = {}
# Getting the type of 'tuple' (line 28)
tuple_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 59), 'tuple', False)
# Calling tuple(args, kwargs) (line 28)
tuple_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 28, 59), tuple_154, *[range_call_result_160], **kwargs_161)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 13), tuple_136, tuple_call_result_162)
# Adding element type (line 28)

# Call to tuple(...): (line 28)
# Processing the call arguments (line 28)

# Call to range(...): (line 28)
# Processing the call arguments (line 28)
int_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 98), 'int')
int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 103), 'int')
int_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 109), 'int')
# Processing the call keyword arguments (line 28)
kwargs_168 = {}
# Getting the type of 'range' (line 28)
range_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 92), 'range', False)
# Calling range(args, kwargs) (line 28)
range_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 28, 92), range_164, *[int_165, int_166, int_167], **kwargs_168)

# Processing the call keyword arguments (line 28)
kwargs_170 = {}
# Getting the type of 'tuple' (line 28)
tuple_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 86), 'tuple', False)
# Calling tuple(args, kwargs) (line 28)
tuple_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 28, 86), tuple_163, *[range_call_result_169], **kwargs_170)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 13), tuple_136, tuple_call_result_171)

# Assigning a type to the variable 'rookLines' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'rookLines', tuple_136)

# Assigning a BinOp to a Name (line 29):
# Getting the type of 'bishopLines' (line 29)
bishopLines_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'bishopLines')
# Getting the type of 'rookLines' (line 29)
rookLines_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'rookLines')
# Applying the binary operator '+' (line 29)
result_add_174 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 13), '+', bishopLines_172, rookLines_173)

# Assigning a type to the variable 'queenLines' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'queenLines', result_add_174)

# Assigning a Tuple to a Name (line 30):

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
int_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_176)
# Adding element type (line 30)
int_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_177)
# Adding element type (line 30)
int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_178)
# Adding element type (line 30)
int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_179)
# Adding element type (line 30)
int_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_180)
# Adding element type (line 30)
int_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_181)
# Adding element type (line 30)
int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_182)
# Adding element type (line 30)
int_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_175, int_183)

# Assigning a type to the variable 'kingMoves' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'kingMoves', tuple_175)

# Assigning a Tuple to a Name (line 32):

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, tuple_185)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, tuple_186)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, tuple_187)
# Adding element type (line 32)
# Getting the type of 'bishopLines' (line 32)
bishopLines_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'bishopLines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, bishopLines_188)
# Adding element type (line 32)
# Getting the type of 'rookLines' (line 32)
rookLines_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'rookLines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, rookLines_189)
# Adding element type (line 32)
# Getting the type of 'queenLines' (line 32)
queenLines_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 50), 'queenLines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, queenLines_190)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, tuple_191)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 66), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, tuple_192)
# Adding element type (line 32)
# Getting the type of 'queenLines' (line 32)
queenLines_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 70), 'queenLines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, queenLines_193)
# Adding element type (line 32)
# Getting the type of 'rookLines' (line 32)
rookLines_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 82), 'rookLines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, rookLines_194)
# Adding element type (line 32)
# Getting the type of 'bishopLines' (line 32)
bishopLines_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 93), 'bishopLines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, bishopLines_195)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 106), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, tuple_196)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 110), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_184, tuple_197)

# Assigning a type to the variable 'linePieces' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'linePieces', tuple_184)

# Assigning a BinOp to a Name (line 34):

# Obtaining an instance of the builtin type 'list' (line 34)
list_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
# Getting the type of 'None' (line 34)
None_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 29), list_198, None_199)

int_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 38), 'int')
# Applying the binary operator '*' (line 34)
result_mul_201 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 29), '*', list_198, int_200)

# Assigning a type to the variable 'clearCastlingOpportunities' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'clearCastlingOpportunities', result_mul_201)


# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 16), tuple_203, int_204)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 22), tuple_205, int_206)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 16), tuple_203, tuple_205)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_202, tuple_203)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), tuple_207, int_208)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 36), tuple_209, int_210)
# Adding element type (line 35)
int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 36), tuple_209, int_211)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), tuple_207, tuple_209)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_202, tuple_207)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 47), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 47), tuple_212, int_213)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 53), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 53), tuple_214, int_215)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 47), tuple_212, tuple_214)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_202, tuple_212)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 61), tuple_216, int_217)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 68), tuple_218, int_219)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 61), tuple_216, tuple_218)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_202, tuple_216)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 76), tuple_220, int_221)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 83), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 83), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 83), tuple_222, int_223)
# Adding element type (line 35)
int_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 87), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 83), tuple_222, int_224)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 76), tuple_220, tuple_222)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_202, tuple_220)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 94), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 94), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 94), tuple_225, int_226)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 101), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 101), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 101), tuple_227, int_228)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 94), tuple_225, tuple_227)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_202, tuple_225)

# Testing the type of a for loop iterable (line 35)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 0), tuple_202)
# Getting the type of the for loop variable (line 35)
for_loop_var_229 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 0), tuple_202)
# Assigning a type to the variable 'i' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 0), for_loop_var_229))
# Assigning a type to the variable 'v' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 0), for_loop_var_229))
# SSA begins for a for statement (line 35)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Name to a Subscript (line 36):
# Getting the type of 'v' (line 36)
v_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'v')
# Getting the type of 'clearCastlingOpportunities' (line 36)
clearCastlingOpportunities_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 2), 'clearCastlingOpportunities')
# Getting the type of 'i' (line 36)
i_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'i')
# Storing an element on a container (line 36)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 2), clearCastlingOpportunities_231, (i_232, v_230))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Name (line 38):
str_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'str', '.pnbrqkKQRBNP')
# Assigning a type to the variable 'pieces' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'pieces', str_233)

@norecursion
def evaluate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'evaluate'
    module_type_store = module_type_store.open_function_context('evaluate', 40, 0, False)
    
    # Passed parameters checking function
    evaluate.stypy_localization = localization
    evaluate.stypy_type_of_self = None
    evaluate.stypy_type_store = module_type_store
    evaluate.stypy_function_name = 'evaluate'
    evaluate.stypy_param_names_list = ['board']
    evaluate.stypy_varargs_param_name = None
    evaluate.stypy_kwargs_param_name = None
    evaluate.stypy_call_defaults = defaults
    evaluate.stypy_call_varargs = varargs
    evaluate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'evaluate', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'evaluate', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'evaluate(...)' code ##################

    
    # Assigning a Tuple to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    int_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_235)
    # Adding element type (line 41)
    int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_236)
    # Adding element type (line 41)
    int_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_237)
    # Adding element type (line 41)
    int_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_238)
    # Adding element type (line 41)
    int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_239)
    # Adding element type (line 41)
    int_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_240)
    # Adding element type (line 41)
    int_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_241)
    # Adding element type (line 41)
    int_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_242)
    # Adding element type (line 41)
    int_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_243)
    # Adding element type (line 41)
    int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_244)
    # Adding element type (line 41)
    int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_245)
    # Adding element type (line 41)
    int_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 74), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_246)
    # Adding element type (line 41)
    int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 80), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), tuple_234, int_247)
    
    # Assigning a type to the variable 'evals' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 2), 'evals', tuple_234)
    
    # Call to sum(...): (line 42)
    # Processing the call arguments (line 42)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'squares' (line 42)
    squares_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'squares', False)
    comprehension_257 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 14), squares_256)
    # Assigning a type to the variable 'i' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'i', comprehension_257)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 42)
    i_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'i', False)
    # Getting the type of 'board' (line 42)
    board_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'board', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), board_250, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), getitem___251, i_249)
    
    # Getting the type of 'evals' (line 42)
    evals_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'evals', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 14), evals_253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), getitem___254, subscript_call_result_252)
    
    list_258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 14), list_258, subscript_call_result_255)
    # Processing the call keyword arguments (line 42)
    kwargs_259 = {}
    # Getting the type of 'sum' (line 42)
    sum_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 'sum', False)
    # Calling sum(args, kwargs) (line 42)
    sum_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 42, 9), sum_248, *[list_258], **kwargs_259)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 2), 'stypy_return_type', sum_call_result_260)
    
    # ################# End of 'evaluate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'evaluate' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'evaluate'
    return stypy_return_type_261

# Assigning a type to the variable 'evaluate' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'evaluate', evaluate)

@norecursion
def printBoard(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'printBoard'
    module_type_store = module_type_store.open_function_context('printBoard', 44, 0, False)
    
    # Passed parameters checking function
    printBoard.stypy_localization = localization
    printBoard.stypy_type_of_self = None
    printBoard.stypy_type_store = module_type_store
    printBoard.stypy_function_name = 'printBoard'
    printBoard.stypy_param_names_list = ['board']
    printBoard.stypy_varargs_param_name = None
    printBoard.stypy_kwargs_param_name = None
    printBoard.stypy_call_defaults = defaults
    printBoard.stypy_call_varargs = varargs
    printBoard.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'printBoard', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'printBoard', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'printBoard(...)' code ##################

    
    
    # Call to range(...): (line 45)
    # Processing the call arguments (line 45)
    int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'int')
    int_264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'int')
    int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'int')
    # Processing the call keyword arguments (line 45)
    kwargs_266 = {}
    # Getting the type of 'range' (line 45)
    range_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'range', False)
    # Calling range(args, kwargs) (line 45)
    range_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 45, 11), range_262, *[int_263, int_264, int_265], **kwargs_266)
    
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 2), range_call_result_267)
    # Getting the type of the for loop variable (line 45)
    for_loop_var_268 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 2), range_call_result_267)
    # Assigning a type to the variable 'i' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 2), 'i', for_loop_var_268)
    # SSA begins for a for statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 46)
    # Processing the call arguments (line 46)
    int_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'int')
    # Processing the call keyword arguments (line 46)
    kwargs_271 = {}
    # Getting the type of 'range' (line 46)
    range_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'range', False)
    # Calling range(args, kwargs) (line 46)
    range_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), range_269, *[int_270], **kwargs_271)
    
    # Testing the type of a for loop iterable (line 46)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_272)
    # Getting the type of the for loop variable (line 46)
    for_loop_var_273 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_272)
    # Assigning a type to the variable 'j' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'j', for_loop_var_273)
    # SSA begins for a for statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 47):
    # Getting the type of 'i' (line 47)
    i_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'i')
    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'int')
    # Applying the binary operator '*' (line 47)
    result_mul_276 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '*', i_274, int_275)
    
    # Getting the type of 'j' (line 47)
    j_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'j')
    # Applying the binary operator '+' (line 47)
    result_add_278 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '+', result_mul_276, j_277)
    
    # Assigning a type to the variable 'ix' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 6), 'ix', result_add_278)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'printBoard(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'printBoard' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_279)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'printBoard'
    return stypy_return_type_279

# Assigning a type to the variable 'printBoard' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'printBoard', printBoard)

@norecursion
def move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'move'
    module_type_store = module_type_store.open_function_context('move', 51, 0, False)
    
    # Passed parameters checking function
    move.stypy_localization = localization
    move.stypy_type_of_self = None
    move.stypy_type_store = module_type_store
    move.stypy_function_name = 'move'
    move.stypy_param_names_list = ['board', 'mv']
    move.stypy_varargs_param_name = None
    move.stypy_kwargs_param_name = None
    move.stypy_call_defaults = defaults
    move.stypy_call_varargs = varargs
    move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'move', ['board', 'mv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'move', localization, ['board', 'mv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'move(...)' code ##################

    
    # Assigning a BinOp to a Name (line 52):
    # Getting the type of 'mv' (line 52)
    mv_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'mv')
    int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'int')
    # Applying the binary operator '>>' (line 52)
    result_rshift_282 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 8), '>>', mv_280, int_281)
    
    int_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 19), 'int')
    # Applying the binary operator '&' (line 52)
    result_and__284 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 7), '&', result_rshift_282, int_283)
    
    # Assigning a type to the variable 'ix' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 2), 'ix', result_and__284)
    
    # Assigning a Subscript to a Subscript (line 53):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ix' (line 53)
    ix_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'ix')
    # Getting the type of 'board' (line 53)
    board_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'board')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 21), board_286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 53, 21), getitem___287, ix_285)
    
    # Getting the type of 'board' (line 53)
    board_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 2), 'board')
    # Getting the type of 'mv' (line 53)
    mv_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'mv')
    int_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'int')
    # Applying the binary operator '&' (line 53)
    result_and__292 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 8), '&', mv_290, int_291)
    
    # Storing an element on a container (line 53)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 2), board_289, (result_and__292, subscript_call_result_288))
    
    # Assigning a Num to a Subscript (line 54):
    int_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'int')
    # Getting the type of 'board' (line 54)
    board_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 2), 'board')
    # Getting the type of 'ix' (line 54)
    ix_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'ix')
    # Storing an element on a container (line 54)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 2), board_294, (ix_295, int_293))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ix' (line 55)
    ix_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'ix')
    # Getting the type of 'clearCastlingOpportunities' (line 55)
    clearCastlingOpportunities_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 5), 'clearCastlingOpportunities')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 5), clearCastlingOpportunities_297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 55, 5), getitem___298, ix_296)
    
    # Testing the type of an if condition (line 55)
    if_condition_300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 2), subscript_call_result_299)
    # Assigning a type to the variable 'if_condition_300' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 2), 'if_condition_300', if_condition_300)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ix' (line 56)
    ix_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 40), 'ix')
    # Getting the type of 'clearCastlingOpportunities' (line 56)
    clearCastlingOpportunities_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'clearCastlingOpportunities')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 13), clearCastlingOpportunities_302, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 56, 13), getitem___303, ix_301)
    
    # Testing the type of a for loop iterable (line 56)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 4), subscript_call_result_304)
    # Getting the type of the for loop variable (line 56)
    for_loop_var_305 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 4), subscript_call_result_304)
    # Assigning a type to the variable 'i' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'i', for_loop_var_305)
    # SSA begins for a for statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 57):
    # Getting the type of 'iFalse' (line 57)
    iFalse_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'iFalse')
    # Getting the type of 'board' (line 57)
    board_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 6), 'board')
    # Getting the type of 'i' (line 57)
    i_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'i')
    # Storing an element on a container (line 57)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 6), board_307, (i_308, iFalse_306))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 59):
    
    # Call to int(...): (line 59)
    # Processing the call arguments (line 59)
    
    
    # Obtaining the type of the subscript
    int_310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'int')
    # Getting the type of 'board' (line 59)
    board_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'board', False)
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 22), board_311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), getitem___312, int_310)
    
    # Applying the 'not' unary operator (line 59)
    result_not__314 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 18), 'not', subscript_call_result_313)
    
    # Processing the call keyword arguments (line 59)
    kwargs_315 = {}
    # Getting the type of 'int' (line 59)
    int_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'int', False)
    # Calling int(args, kwargs) (line 59)
    int_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 59, 14), int_309, *[result_not__314], **kwargs_315)
    
    # Getting the type of 'board' (line 59)
    board_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 2), 'board')
    int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
    # Storing an element on a container (line 59)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 2), board_317, (int_318, int_call_result_316))
    
    
    # Getting the type of 'mv' (line 60)
    mv_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 6), 'mv')
    int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 11), 'int')
    # Applying the binary operator '&' (line 60)
    result_and__321 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 6), '&', mv_319, int_320)
    
    int_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'int')
    # Applying the binary operator '==' (line 60)
    result_eq_323 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 5), '==', result_and__321, int_322)
    
    # Testing the type of an if condition (line 60)
    if_condition_324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 2), result_eq_323)
    # Assigning a type to the variable 'if_condition_324' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 2), 'if_condition_324', if_condition_324)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'mv' (line 62)
    mv_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 6), 'mv')
    int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 11), 'int')
    # Applying the binary operator '&' (line 62)
    result_and__327 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 6), '&', mv_325, int_326)
    
    # Testing the type of an if condition (line 62)
    if_condition_328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 2), result_and__327)
    # Assigning a type to the variable 'if_condition_328' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 2), 'if_condition_328', if_condition_328)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 63):
    # Getting the type of 'mv' (line 63)
    mv_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'mv')
    int_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
    # Applying the binary operator '&' (line 63)
    result_and__331 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '&', mv_329, int_330)
    
    # Getting the type of 'board' (line 63)
    board_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'board')
    int_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 10), 'int')
    # Storing an element on a container (line 63)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 4), board_332, (int_333, result_and__331))
    # SSA branch for the else part of an if statement (line 62)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 65):
    # Getting the type of 'iNone' (line 65)
    iNone_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'iNone')
    # Getting the type of 'board' (line 65)
    board_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'board')
    int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 10), 'int')
    # Storing an element on a container (line 65)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 4), board_335, (int_336, iNone_334))
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'mv' (line 66)
    mv_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 6), 'mv')
    int_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 11), 'int')
    # Applying the binary operator '&' (line 66)
    result_and__339 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 6), '&', mv_337, int_338)
    
    # Testing the type of an if condition (line 66)
    if_condition_340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 2), result_and__339)
    # Assigning a type to the variable 'if_condition_340' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 2), 'if_condition_340', if_condition_340)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 67):
    # Getting the type of 'mv' (line 67)
    mv_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'mv')
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'int')
    # Applying the binary operator '&' (line 67)
    result_and__343 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '&', mv_341, int_342)
    
    # Assigning a type to the variable 'toix' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'toix', result_and__343)
    
    
    # Getting the type of 'toix' (line 68)
    toix_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'toix')
    int_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'int')
    # Applying the binary operator '==' (line 68)
    result_eq_346 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), '==', toix_344, int_345)
    
    # Testing the type of an if condition (line 68)
    if_condition_347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_eq_346)
    # Assigning a type to the variable 'if_condition_347' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_347', if_condition_347)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 69):
    int_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
    # Getting the type of 'board' (line 69)
    board_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 6), 'board')
    int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 12), 'int')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 6), board_349, (int_350, int_348))
    
    # Assigning a Num to a Subscript (line 70):
    int_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
    # Getting the type of 'board' (line 70)
    board_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 6), 'board')
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'int')
    # Storing an element on a container (line 70)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 6), board_352, (int_353, int_351))
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'toix' (line 71)
    toix_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 9), 'toix')
    int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 17), 'int')
    # Applying the binary operator '==' (line 71)
    result_eq_356 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), '==', toix_354, int_355)
    
    # Testing the type of an if condition (line 71)
    if_condition_357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 9), result_eq_356)
    # Assigning a type to the variable 'if_condition_357' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 9), 'if_condition_357', if_condition_357)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 72):
    int_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'int')
    # Getting the type of 'board' (line 72)
    board_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 6), 'board')
    int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'int')
    # Storing an element on a container (line 72)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 6), board_359, (int_360, int_358))
    
    # Assigning a Num to a Subscript (line 73):
    int_361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'int')
    # Getting the type of 'board' (line 73)
    board_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 6), 'board')
    int_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'int')
    # Storing an element on a container (line 73)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 6), board_362, (int_363, int_361))
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'toix' (line 74)
    toix_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 9), 'toix')
    int_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'int')
    # Applying the binary operator '==' (line 74)
    result_eq_366 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 9), '==', toix_364, int_365)
    
    # Testing the type of an if condition (line 74)
    if_condition_367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 9), result_eq_366)
    # Assigning a type to the variable 'if_condition_367' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 9), 'if_condition_367', if_condition_367)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 75):
    int_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'int')
    # Getting the type of 'board' (line 75)
    board_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 6), 'board')
    int_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'int')
    # Storing an element on a container (line 75)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 6), board_369, (int_370, int_368))
    
    # Assigning a Num to a Subscript (line 76):
    int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'int')
    # Getting the type of 'board' (line 76)
    board_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 6), 'board')
    int_373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'int')
    # Storing an element on a container (line 76)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 6), board_372, (int_373, int_371))
    # SSA branch for the else part of an if statement (line 74)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'toix' (line 77)
    toix_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'toix')
    int_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 17), 'int')
    # Applying the binary operator '==' (line 77)
    result_eq_376 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 9), '==', toix_374, int_375)
    
    # Testing the type of an if condition (line 77)
    if_condition_377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 9), result_eq_376)
    # Assigning a type to the variable 'if_condition_377' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'if_condition_377', if_condition_377)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 78):
    int_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'int')
    # Getting the type of 'board' (line 78)
    board_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 6), 'board')
    int_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 12), 'int')
    # Storing an element on a container (line 78)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 6), board_379, (int_380, int_378))
    
    # Assigning a Num to a Subscript (line 79):
    int_381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'int')
    # Getting the type of 'board' (line 79)
    board_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 6), 'board')
    int_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 12), 'int')
    # Storing an element on a container (line 79)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 6), board_382, (int_383, int_381))
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    str_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 12), 'str', 'faulty castling')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 81, 6), str_384, 'raise parameter', BaseException)
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'mv' (line 82)
    mv_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 5), 'mv')
    int_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 10), 'int')
    # Applying the binary operator '&' (line 82)
    result_and__387 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 5), '&', mv_385, int_386)
    
    # Testing the type of an if condition (line 82)
    if_condition_388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 2), result_and__387)
    # Assigning a type to the variable 'if_condition_388' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 2), 'if_condition_388', if_condition_388)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'int')
    # Getting the type of 'board' (line 83)
    board_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'board')
    # Obtaining the member '__getitem__' of a type (line 83)
    getitem___391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 7), board_390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 83)
    subscript_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 83, 7), getitem___391, int_389)
    
    # Testing the type of an if condition (line 83)
    if_condition_393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), subscript_call_result_392)
    # Assigning a type to the variable 'if_condition_393' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_393', if_condition_393)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 84):
    int_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
    # Getting the type of 'board' (line 84)
    board_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 6), 'board')
    # Getting the type of 'mv' (line 84)
    mv_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'mv')
    int_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 17), 'int')
    int_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'int')
    # Applying the binary operator '+' (line 84)
    result_add_399 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 17), '+', int_397, int_398)
    
    # Applying the binary operator '&' (line 84)
    result_and__400 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '&', mv_396, result_add_399)
    
    # Storing an element on a container (line 84)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 6), board_395, (result_and__400, int_394))
    # SSA branch for the else part of an if statement (line 83)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 86):
    int_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
    # Getting the type of 'board' (line 86)
    board_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 6), 'board')
    # Getting the type of 'mv' (line 86)
    mv_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'mv')
    int_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 17), 'int')
    int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'int')
    # Applying the binary operator '+' (line 86)
    result_add_406 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 17), '+', int_404, int_405)
    
    # Applying the binary operator '&' (line 86)
    result_and__407 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 12), '&', mv_403, result_add_406)
    
    # Storing an element on a container (line 86)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 6), board_402, (result_and__407, int_401))
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'mv' (line 87)
    mv_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 5), 'mv')
    int_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 10), 'int')
    # Applying the binary operator '&' (line 87)
    result_and__410 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 5), '&', mv_408, int_409)
    
    # Testing the type of an if condition (line 87)
    if_condition_411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 2), result_and__410)
    # Assigning a type to the variable 'if_condition_411' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 2), 'if_condition_411', if_condition_411)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 88):
    # Getting the type of 'mv' (line 88)
    mv_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 9), 'mv')
    int_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 14), 'int')
    # Applying the binary operator '&' (line 88)
    result_and__414 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 9), '&', mv_412, int_413)
    
    int_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 27), 'int')
    # Applying the binary operator '>>' (line 88)
    result_rshift_416 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 8), '>>', result_and__414, int_415)
    
    # Assigning a type to the variable 'a' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'a', result_rshift_416)
    
    
    # Getting the type of 'a' (line 89)
    a_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'a')
    int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 13), 'int')
    # Applying the binary operator '>=' (line 89)
    result_ge_419 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 8), '>=', a_417, int_418)
    
    # Testing the type of an if condition (line 89)
    if_condition_420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), result_ge_419)
    # Assigning a type to the variable 'if_condition_420' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_420', if_condition_420)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 90):
    # Getting the type of 'a' (line 90)
    a_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 10), 'a')
    int_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 14), 'int')
    # Applying the binary operator '-' (line 90)
    result_sub_423 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 10), '-', a_421, int_422)
    
    # Assigning a type to the variable 'a' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 6), 'a', result_sub_423)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 91):
    # Getting the type of 'a' (line 91)
    a_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'a')
    # Getting the type of 'board' (line 91)
    board_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'board')
    # Getting the type of 'mv' (line 91)
    mv_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 10), 'mv')
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'int')
    # Applying the binary operator '&' (line 91)
    result_and__428 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 10), '&', mv_426, int_427)
    
    # Storing an element on a container (line 91)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 4), board_425, (result_and__428, a_424))
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'move' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_429)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'move'
    return stypy_return_type_429

# Assigning a type to the variable 'move' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'move', move)

@norecursion
def toString(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'toString'
    module_type_store = module_type_store.open_function_context('toString', 93, 0, False)
    
    # Passed parameters checking function
    toString.stypy_localization = localization
    toString.stypy_type_of_self = None
    toString.stypy_type_store = module_type_store
    toString.stypy_function_name = 'toString'
    toString.stypy_param_names_list = ['move']
    toString.stypy_varargs_param_name = None
    toString.stypy_kwargs_param_name = None
    toString.stypy_call_defaults = defaults
    toString.stypy_call_varargs = varargs
    toString.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'toString', ['move'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'toString', localization, ['move'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'toString(...)' code ##################

    
    # Assigning a BinOp to a Name (line 94):
    # Getting the type of 'move' (line 94)
    move_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'move')
    int_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 16), 'int')
    # Applying the binary operator '>>' (line 94)
    result_rshift_432 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 8), '>>', move_430, int_431)
    
    int_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'int')
    # Applying the binary operator '&' (line 94)
    result_and__434 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '&', result_rshift_432, int_433)
    
    # Assigning a type to the variable 'fr' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 2), 'fr', result_and__434)
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'move' (line 95)
    move_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'move')
    int_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 14), 'int')
    # Applying the binary operator '&' (line 95)
    result_and__437 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 7), '&', move_435, int_436)
    
    # Assigning a type to the variable 'to' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 2), 'to', result_and__437)
    
    # Assigning a Str to a Name (line 96):
    str_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'str', 'abcdefgh')
    # Assigning a type to the variable 'letters' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 2), 'letters', str_438)
    
    # Assigning a Str to a Name (line 97):
    str_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 12), 'str', '12345678')
    # Assigning a type to the variable 'numbers' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 2), 'numbers', str_439)
    
    # Assigning a Str to a Name (line 98):
    str_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'str', '-')
    # Assigning a type to the variable 'mid' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 2), 'mid', str_440)
    
    # Getting the type of 'move' (line 99)
    move_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 6), 'move')
    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 13), 'int')
    # Applying the binary operator '&' (line 99)
    result_and__443 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 6), '&', move_441, int_442)
    
    # Testing the type of an if condition (line 99)
    if_condition_444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 2), result_and__443)
    # Assigning a type to the variable 'if_condition_444' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 2), 'if_condition_444', if_condition_444)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'move' (line 100)
    move_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'move')
    int_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'int')
    # Applying the binary operator '&' (line 100)
    result_and__447 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 8), '&', move_445, int_446)
    
    int_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'int')
    # Applying the binary operator '==' (line 100)
    result_eq_449 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), '==', result_and__447, int_448)
    
    # Testing the type of an if condition (line 100)
    if_condition_450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_eq_449)
    # Assigning a type to the variable 'if_condition_450' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_450', if_condition_450)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'str', 'O-O-O')
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 6), 'stypy_return_type', str_451)
    # SSA branch for the else part of an if statement (line 100)
    module_type_store.open_ssa_branch('else')
    str_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 13), 'str', 'O-O')
    # Assigning a type to the variable 'stypy_return_type' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 6), 'stypy_return_type', str_452)
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'move' (line 104)
    move_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 5), 'move')
    int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 12), 'int')
    # Applying the binary operator '&' (line 104)
    result_and__455 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 5), '&', move_453, int_454)
    
    # Testing the type of an if condition (line 104)
    if_condition_456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 2), result_and__455)
    # Assigning a type to the variable 'if_condition_456' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 2), 'if_condition_456', if_condition_456)
    # SSA begins for if statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 105):
    str_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 10), 'str', 'x')
    # Assigning a type to the variable 'mid' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'mid', str_457)
    # SSA join for if statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 106):
    
    # Obtaining the type of the subscript
    # Getting the type of 'fr' (line 106)
    fr_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'fr')
    int_459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'int')
    # Applying the binary operator '&' (line 106)
    result_and__460 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 19), '&', fr_458, int_459)
    
    # Getting the type of 'letters' (line 106)
    letters_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'letters')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), letters_461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 106, 11), getitem___462, result_and__460)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'fr' (line 106)
    fr_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 37), 'fr')
    int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 43), 'int')
    # Applying the binary operator '>>' (line 106)
    result_rshift_466 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 37), '>>', fr_464, int_465)
    
    # Getting the type of 'numbers' (line 106)
    numbers_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'numbers')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 29), numbers_467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_469 = invoke(stypy.reporting.localization.Localization(__file__, 106, 29), getitem___468, result_rshift_466)
    
    # Applying the binary operator '+' (line 106)
    result_add_470 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), '+', subscript_call_result_463, subscript_call_result_469)
    
    # Getting the type of 'mid' (line 106)
    mid_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), 'mid')
    # Applying the binary operator '+' (line 106)
    result_add_472 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 46), '+', result_add_470, mid_471)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'to' (line 106)
    to_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 62), 'to')
    int_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 67), 'int')
    # Applying the binary operator '&' (line 106)
    result_and__475 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 62), '&', to_473, int_474)
    
    # Getting the type of 'letters' (line 106)
    letters_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 54), 'letters')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 54), letters_476, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 106, 54), getitem___477, result_and__475)
    
    # Applying the binary operator '+' (line 106)
    result_add_479 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 52), '+', result_add_472, subscript_call_result_478)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'to' (line 106)
    to_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 80), 'to')
    int_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 86), 'int')
    # Applying the binary operator '>>' (line 106)
    result_rshift_482 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 80), '>>', to_480, int_481)
    
    # Getting the type of 'numbers' (line 106)
    numbers_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 72), 'numbers')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 72), numbers_483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 106, 72), getitem___484, result_rshift_482)
    
    # Applying the binary operator '+' (line 106)
    result_add_486 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 70), '+', result_add_479, subscript_call_result_485)
    
    # Assigning a type to the variable 'retval' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 2), 'retval', result_add_486)
    # Getting the type of 'retval' (line 107)
    retval_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 9), 'retval')
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 2), 'stypy_return_type', retval_487)
    
    # ################# End of 'toString(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'toString' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_488)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'toString'
    return stypy_return_type_488

# Assigning a type to the variable 'toString' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'toString', toString)

@norecursion
def moveStr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'moveStr'
    module_type_store = module_type_store.open_function_context('moveStr', 109, 0, False)
    
    # Passed parameters checking function
    moveStr.stypy_localization = localization
    moveStr.stypy_type_of_self = None
    moveStr.stypy_type_store = module_type_store
    moveStr.stypy_function_name = 'moveStr'
    moveStr.stypy_param_names_list = ['board', 'strMove']
    moveStr.stypy_varargs_param_name = None
    moveStr.stypy_kwargs_param_name = None
    moveStr.stypy_call_defaults = defaults
    moveStr.stypy_call_varargs = varargs
    moveStr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'moveStr', ['board', 'strMove'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'moveStr', localization, ['board', 'strMove'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'moveStr(...)' code ##################

    
    # Assigning a Call to a Name (line 110):
    
    # Call to pseudoLegalMoves(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'board' (line 110)
    board_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'board', False)
    # Processing the call keyword arguments (line 110)
    kwargs_491 = {}
    # Getting the type of 'pseudoLegalMoves' (line 110)
    pseudoLegalMoves_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 10), 'pseudoLegalMoves', False)
    # Calling pseudoLegalMoves(args, kwargs) (line 110)
    pseudoLegalMoves_call_result_492 = invoke(stypy.reporting.localization.Localization(__file__, 110, 10), pseudoLegalMoves_489, *[board_490], **kwargs_491)
    
    # Assigning a type to the variable 'moves' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 2), 'moves', pseudoLegalMoves_call_result_492)
    
    # Getting the type of 'moves' (line 111)
    moves_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'moves')
    # Testing the type of a for loop iterable (line 111)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 2), moves_493)
    # Getting the type of the for loop variable (line 111)
    for_loop_var_494 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 2), moves_493)
    # Assigning a type to the variable 'm' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 2), 'm', for_loop_var_494)
    # SSA begins for a for statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'strMove' (line 112)
    strMove_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'strMove')
    
    # Call to toString(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'm' (line 112)
    m_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'm', False)
    # Processing the call keyword arguments (line 112)
    kwargs_498 = {}
    # Getting the type of 'toString' (line 112)
    toString_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'toString', False)
    # Calling toString(args, kwargs) (line 112)
    toString_call_result_499 = invoke(stypy.reporting.localization.Localization(__file__, 112, 18), toString_496, *[m_497], **kwargs_498)
    
    # Applying the binary operator '==' (line 112)
    result_eq_500 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 7), '==', strMove_495, toString_call_result_499)
    
    # Testing the type of an if condition (line 112)
    if_condition_501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), result_eq_500)
    # Assigning a type to the variable 'if_condition_501' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_501', if_condition_501)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to move(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'board' (line 113)
    board_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'board', False)
    # Getting the type of 'm' (line 113)
    m_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'm', False)
    # Processing the call keyword arguments (line 113)
    kwargs_505 = {}
    # Getting the type of 'move' (line 113)
    move_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 6), 'move', False)
    # Calling move(args, kwargs) (line 113)
    move_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 113, 6), move_502, *[board_503, m_504], **kwargs_505)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 6), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'moves' (line 115)
    moves_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'moves')
    # Testing the type of a for loop iterable (line 115)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 2), moves_507)
    # Getting the type of the for loop variable (line 115)
    for_loop_var_508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 2), moves_507)
    # Assigning a type to the variable 'm' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 2), 'm', for_loop_var_508)
    # SSA begins for a for statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    pass
    
    # Call to toString(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'm' (line 117)
    m_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'm', False)
    # Processing the call keyword arguments (line 117)
    kwargs_511 = {}
    # Getting the type of 'toString' (line 117)
    toString_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'toString', False)
    # Calling toString(args, kwargs) (line 117)
    toString_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 117, 4), toString_509, *[m_510], **kwargs_511)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'str', 'no move found')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 118, 2), str_513, 'raise parameter', BaseException)
    
    # ################# End of 'moveStr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'moveStr' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_514)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'moveStr'
    return stypy_return_type_514

# Assigning a type to the variable 'moveStr' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'moveStr', moveStr)

@norecursion
def rowAttack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rowAttack'
    module_type_store = module_type_store.open_function_context('rowAttack', 120, 0, False)
    
    # Passed parameters checking function
    rowAttack.stypy_localization = localization
    rowAttack.stypy_type_of_self = None
    rowAttack.stypy_type_store = module_type_store
    rowAttack.stypy_function_name = 'rowAttack'
    rowAttack.stypy_param_names_list = ['board', 'attackers', 'ix', 'dir']
    rowAttack.stypy_varargs_param_name = None
    rowAttack.stypy_kwargs_param_name = None
    rowAttack.stypy_call_defaults = defaults
    rowAttack.stypy_call_varargs = varargs
    rowAttack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rowAttack', ['board', 'attackers', 'ix', 'dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rowAttack', localization, ['board', 'attackers', 'ix', 'dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rowAttack(...)' code ##################

    
    # Assigning a Subscript to a Name (line 121):
    
    # Obtaining the type of the subscript
    int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 18), 'int')
    # Getting the type of 'attackers' (line 121)
    attackers_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'attackers')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), attackers_516, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), getitem___517, int_515)
    
    # Assigning a type to the variable 'own' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 2), 'own', subscript_call_result_518)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'dir' (line 122)
    dir_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'dir')
    comprehension_523 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 12), dir_522)
    # Assigning a type to the variable 'i' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'i', comprehension_523)
    # Getting the type of 'i' (line 122)
    i_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'i')
    # Getting the type of 'ix' (line 122)
    ix_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'ix')
    # Applying the binary operator '+' (line 122)
    result_add_521 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 12), '+', i_519, ix_520)
    
    list_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 12), list_524, result_add_521)
    # Testing the type of a for loop iterable (line 122)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 2), list_524)
    # Getting the type of the for loop variable (line 122)
    for_loop_var_525 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 2), list_524)
    # Assigning a type to the variable 'k' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 2), 'k', for_loop_var_525)
    # SSA begins for a for statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'k' (line 123)
    k_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 7), 'k')
    int_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 11), 'int')
    # Applying the binary operator '&' (line 123)
    result_and__528 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 7), '&', k_526, int_527)
    
    # Testing the type of an if condition (line 123)
    if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 4), result_and__528)
    # Assigning a type to the variable 'if_condition_529' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'if_condition_529', if_condition_529)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 124)
    False_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 6), 'stypy_return_type', False_530)
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 125)
    k_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'k')
    # Getting the type of 'board' (line 125)
    board_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 7), 'board')
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 7), board_532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_534 = invoke(stypy.reporting.localization.Localization(__file__, 125, 7), getitem___533, k_531)
    
    # Testing the type of an if condition (line 125)
    if_condition_535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 4), subscript_call_result_534)
    # Assigning a type to the variable 'if_condition_535' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'if_condition_535', if_condition_535)
    # SSA begins for if statement (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 126)
    k_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'k')
    # Getting the type of 'board' (line 126)
    board_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'board')
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 14), board_537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), getitem___538, k_536)
    
    # Getting the type of 'own' (line 126)
    own_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'own')
    # Applying the binary operator '*' (line 126)
    result_mul_541 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 14), '*', subscript_call_result_539, own_540)
    
    int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 31), 'int')
    # Applying the binary operator '<' (line 126)
    result_lt_543 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 14), '<', result_mul_541, int_542)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 126)
    k_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 44), 'k')
    # Getting the type of 'board' (line 126)
    board_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'board')
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 38), board_545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 126, 38), getitem___546, k_544)
    
    # Getting the type of 'attackers' (line 126)
    attackers_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 50), 'attackers')
    # Applying the binary operator 'in' (line 126)
    result_contains_549 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 38), 'in', subscript_call_result_547, attackers_548)
    
    # Applying the binary operator 'and' (line 126)
    result_and_keyword_550 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 13), 'and', result_lt_543, result_contains_549)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 6), 'stypy_return_type', result_and_keyword_550)
    # SSA join for if statement (line 125)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'rowAttack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rowAttack' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rowAttack'
    return stypy_return_type_551

# Assigning a type to the variable 'rowAttack' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'rowAttack', rowAttack)

@norecursion
def nonpawnAttacks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nonpawnAttacks'
    module_type_store = module_type_store.open_function_context('nonpawnAttacks', 128, 0, False)
    
    # Passed parameters checking function
    nonpawnAttacks.stypy_localization = localization
    nonpawnAttacks.stypy_type_of_self = None
    nonpawnAttacks.stypy_type_store = module_type_store
    nonpawnAttacks.stypy_function_name = 'nonpawnAttacks'
    nonpawnAttacks.stypy_param_names_list = ['board', 'ix', 'color']
    nonpawnAttacks.stypy_varargs_param_name = None
    nonpawnAttacks.stypy_kwargs_param_name = None
    nonpawnAttacks.stypy_call_defaults = defaults
    nonpawnAttacks.stypy_call_varargs = varargs
    nonpawnAttacks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nonpawnAttacks', ['board', 'ix', 'color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nonpawnAttacks', localization, ['board', 'ix', 'color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nonpawnAttacks(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to max(...): (line 129)
    # Processing the call arguments (line 129)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'knightMoves' (line 129)
    knightMoves_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 51), 'knightMoves', False)
    comprehension_564 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 15), knightMoves_563)
    # Assigning a type to the variable 'i' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'i', comprehension_564)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ix' (line 129)
    ix_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'ix', False)
    # Getting the type of 'i' (line 129)
    i_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'i', False)
    # Applying the binary operator '+' (line 129)
    result_add_555 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 21), '+', ix_553, i_554)
    
    # Getting the type of 'board' (line 129)
    board_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'board', False)
    # Obtaining the member '__getitem__' of a type (line 129)
    getitem___557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 15), board_556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 129)
    subscript_call_result_558 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), getitem___557, result_add_555)
    
    # Getting the type of 'color' (line 129)
    color_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'color', False)
    int_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'int')
    # Applying the binary operator '*' (line 129)
    result_mul_561 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 32), '*', color_559, int_560)
    
    # Applying the binary operator '==' (line 129)
    result_eq_562 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), '==', subscript_call_result_558, result_mul_561)
    
    list_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 15), list_565, result_eq_562)
    # Processing the call keyword arguments (line 129)
    kwargs_566 = {}
    # Getting the type of 'max' (line 129)
    max_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 10), 'max', False)
    # Calling max(args, kwargs) (line 129)
    max_call_result_567 = invoke(stypy.reporting.localization.Localization(__file__, 129, 10), max_552, *[list_565], **kwargs_566)
    
    
    # Call to max(...): (line 130)
    # Processing the call arguments (line 130)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bishopLines' (line 130)
    bishopLines_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 90), 'bishopLines', False)
    comprehension_583 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), bishopLines_582)
    # Assigning a type to the variable 'bishopLine' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'bishopLine', comprehension_583)
    
    # Call to rowAttack(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'board' (line 130)
    board_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'board', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 130)
    tuple_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 130)
    # Adding element type (line 130)
    # Getting the type of 'color' (line 130)
    color_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'color', False)
    int_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 41), 'int')
    # Applying the binary operator '*' (line 130)
    result_mul_574 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 33), '*', color_572, int_573)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 33), tuple_571, result_mul_574)
    # Adding element type (line 130)
    # Getting the type of 'color' (line 130)
    color_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 44), 'color', False)
    int_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 52), 'int')
    # Applying the binary operator '*' (line 130)
    result_mul_577 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 44), '*', color_575, int_576)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 33), tuple_571, result_mul_577)
    
    # Getting the type of 'ix' (line 130)
    ix_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 56), 'ix', False)
    # Getting the type of 'bishopLine' (line 130)
    bishopLine_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 60), 'bishopLine', False)
    # Processing the call keyword arguments (line 130)
    kwargs_580 = {}
    # Getting the type of 'rowAttack' (line 130)
    rowAttack_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'rowAttack', False)
    # Calling rowAttack(args, kwargs) (line 130)
    rowAttack_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), rowAttack_569, *[board_570, tuple_571, ix_578, bishopLine_579], **kwargs_580)
    
    list_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_584, rowAttack_call_result_581)
    # Processing the call keyword arguments (line 130)
    kwargs_585 = {}
    # Getting the type of 'max' (line 130)
    max_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 10), 'max', False)
    # Calling max(args, kwargs) (line 130)
    max_call_result_586 = invoke(stypy.reporting.localization.Localization(__file__, 130, 10), max_568, *[list_584], **kwargs_585)
    
    # Applying the binary operator 'or' (line 129)
    result_or_keyword_587 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 10), 'or', max_call_result_567, max_call_result_586)
    
    # Call to max(...): (line 131)
    # Processing the call arguments (line 131)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'rookLines' (line 131)
    rookLines_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 86), 'rookLines', False)
    comprehension_603 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), rookLines_602)
    # Assigning a type to the variable 'rookLine' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'rookLine', comprehension_603)
    
    # Call to rowAttack(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'board' (line 131)
    board_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'board', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 131)
    tuple_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 131)
    # Adding element type (line 131)
    # Getting the type of 'color' (line 131)
    color_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 33), 'color', False)
    int_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 41), 'int')
    # Applying the binary operator '*' (line 131)
    result_mul_594 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 33), '*', color_592, int_593)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 33), tuple_591, result_mul_594)
    # Adding element type (line 131)
    # Getting the type of 'color' (line 131)
    color_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'color', False)
    int_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 52), 'int')
    # Applying the binary operator '*' (line 131)
    result_mul_597 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 44), '*', color_595, int_596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 33), tuple_591, result_mul_597)
    
    # Getting the type of 'ix' (line 131)
    ix_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 56), 'ix', False)
    # Getting the type of 'rookLine' (line 131)
    rookLine_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 60), 'rookLine', False)
    # Processing the call keyword arguments (line 131)
    kwargs_600 = {}
    # Getting the type of 'rowAttack' (line 131)
    rowAttack_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'rowAttack', False)
    # Calling rowAttack(args, kwargs) (line 131)
    rowAttack_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), rowAttack_589, *[board_590, tuple_591, ix_598, rookLine_599], **kwargs_600)
    
    list_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), list_604, rowAttack_call_result_601)
    # Processing the call keyword arguments (line 131)
    kwargs_605 = {}
    # Getting the type of 'max' (line 131)
    max_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 10), 'max', False)
    # Calling max(args, kwargs) (line 131)
    max_call_result_606 = invoke(stypy.reporting.localization.Localization(__file__, 131, 10), max_588, *[list_604], **kwargs_605)
    
    # Applying the binary operator 'or' (line 129)
    result_or_keyword_607 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 10), 'or', result_or_keyword_587, max_call_result_606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 2), 'stypy_return_type', result_or_keyword_607)
    
    # ################# End of 'nonpawnAttacks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nonpawnAttacks' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_608)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nonpawnAttacks'
    return stypy_return_type_608

# Assigning a type to the variable 'nonpawnAttacks' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'nonpawnAttacks', nonpawnAttacks)

# Assigning a Lambda to a Name (line 133):

@norecursion
def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_1'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 133, 22, True)
    # Passed parameters checking function
    _stypy_temp_lambda_1.stypy_localization = localization
    _stypy_temp_lambda_1.stypy_type_of_self = None
    _stypy_temp_lambda_1.stypy_type_store = module_type_store
    _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
    _stypy_temp_lambda_1.stypy_param_names_list = ['board', 'ix']
    _stypy_temp_lambda_1.stypy_varargs_param_name = None
    _stypy_temp_lambda_1.stypy_kwargs_param_name = None
    _stypy_temp_lambda_1.stypy_call_defaults = defaults
    _stypy_temp_lambda_1.stypy_call_varargs = varargs
    _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['board', 'ix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_1', ['board', 'ix'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to nonpawnAttacks(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'board' (line 133)
    board_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 55), 'board', False)
    # Getting the type of 'ix' (line 133)
    ix_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 62), 'ix', False)
    int_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 66), 'int')
    # Processing the call keyword arguments (line 133)
    kwargs_613 = {}
    # Getting the type of 'nonpawnAttacks' (line 133)
    nonpawnAttacks_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'nonpawnAttacks', False)
    # Calling nonpawnAttacks(args, kwargs) (line 133)
    nonpawnAttacks_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), nonpawnAttacks_609, *[board_610, ix_611, int_612], **kwargs_613)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'stypy_return_type', nonpawnAttacks_call_result_614)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_1' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_1'
    return stypy_return_type_615

# Assigning a type to the variable '_stypy_temp_lambda_1' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
# Getting the type of '_stypy_temp_lambda_1' (line 133)
_stypy_temp_lambda_1_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), '_stypy_temp_lambda_1')
# Assigning a type to the variable 'nonpawnBlackAttacks' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'nonpawnBlackAttacks', _stypy_temp_lambda_1_616)

# Assigning a Lambda to a Name (line 134):

@norecursion
def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_2'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 134, 22, True)
    # Passed parameters checking function
    _stypy_temp_lambda_2.stypy_localization = localization
    _stypy_temp_lambda_2.stypy_type_of_self = None
    _stypy_temp_lambda_2.stypy_type_store = module_type_store
    _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
    _stypy_temp_lambda_2.stypy_param_names_list = ['board', 'ix']
    _stypy_temp_lambda_2.stypy_varargs_param_name = None
    _stypy_temp_lambda_2.stypy_kwargs_param_name = None
    _stypy_temp_lambda_2.stypy_call_defaults = defaults
    _stypy_temp_lambda_2.stypy_call_varargs = varargs
    _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['board', 'ix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_2', ['board', 'ix'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to nonpawnAttacks(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'board' (line 134)
    board_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 55), 'board', False)
    # Getting the type of 'ix' (line 134)
    ix_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 62), 'ix', False)
    int_620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 66), 'int')
    # Processing the call keyword arguments (line 134)
    kwargs_621 = {}
    # Getting the type of 'nonpawnAttacks' (line 134)
    nonpawnAttacks_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 40), 'nonpawnAttacks', False)
    # Calling nonpawnAttacks(args, kwargs) (line 134)
    nonpawnAttacks_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 134, 40), nonpawnAttacks_617, *[board_618, ix_619, int_620], **kwargs_621)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'stypy_return_type', nonpawnAttacks_call_result_622)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_2' in the type store
    # Getting the type of 'stypy_return_type' (line 134)
    stypy_return_type_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_623)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_2'
    return stypy_return_type_623

# Assigning a type to the variable '_stypy_temp_lambda_2' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
# Getting the type of '_stypy_temp_lambda_2' (line 134)
_stypy_temp_lambda_2_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), '_stypy_temp_lambda_2')
# Assigning a type to the variable 'nonpawnWhiteAttacks' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'nonpawnWhiteAttacks', _stypy_temp_lambda_2_624)

@norecursion
def pseudoLegalMovesWhite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pseudoLegalMovesWhite'
    module_type_store = module_type_store.open_function_context('pseudoLegalMovesWhite', 136, 0, False)
    
    # Passed parameters checking function
    pseudoLegalMovesWhite.stypy_localization = localization
    pseudoLegalMovesWhite.stypy_type_of_self = None
    pseudoLegalMovesWhite.stypy_type_store = module_type_store
    pseudoLegalMovesWhite.stypy_function_name = 'pseudoLegalMovesWhite'
    pseudoLegalMovesWhite.stypy_param_names_list = ['board']
    pseudoLegalMovesWhite.stypy_varargs_param_name = None
    pseudoLegalMovesWhite.stypy_kwargs_param_name = None
    pseudoLegalMovesWhite.stypy_call_defaults = defaults
    pseudoLegalMovesWhite.stypy_call_varargs = varargs
    pseudoLegalMovesWhite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pseudoLegalMovesWhite', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pseudoLegalMovesWhite', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pseudoLegalMovesWhite(...)' code ##################

    
    # Assigning a Call to a Name (line 137):
    
    # Call to pseudoLegalCapturesWhite(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'board' (line 137)
    board_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'board', False)
    # Processing the call keyword arguments (line 137)
    kwargs_627 = {}
    # Getting the type of 'pseudoLegalCapturesWhite' (line 137)
    pseudoLegalCapturesWhite_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'pseudoLegalCapturesWhite', False)
    # Calling pseudoLegalCapturesWhite(args, kwargs) (line 137)
    pseudoLegalCapturesWhite_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), pseudoLegalCapturesWhite_625, *[board_626], **kwargs_627)
    
    # Assigning a type to the variable 'retval' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 2), 'retval', pseudoLegalCapturesWhite_call_result_628)
    
    # Getting the type of 'squares' (line 138)
    squares_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'squares')
    # Testing the type of a for loop iterable (line 138)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 2), squares_629)
    # Getting the type of the for loop variable (line 138)
    for_loop_var_630 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 2), squares_629)
    # Assigning a type to the variable 'sq' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 2), 'sq', for_loop_var_630)
    # SSA begins for a for statement (line 138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 139):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 139)
    sq_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 14), 'sq')
    # Getting the type of 'board' (line 139)
    board_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), board_632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_634 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), getitem___633, sq_631)
    
    # Assigning a type to the variable 'b' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'b', subscript_call_result_634)
    
    
    # Getting the type of 'b' (line 140)
    b_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 7), 'b')
    int_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 12), 'int')
    # Applying the binary operator '>=' (line 140)
    result_ge_637 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 7), '>=', b_635, int_636)
    
    # Testing the type of an if condition (line 140)
    if_condition_638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 4), result_ge_637)
    # Assigning a type to the variable 'if_condition_638' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'if_condition_638', if_condition_638)
    # SSA begins for if statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 141)
    b_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'b')
    int_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 14), 'int')
    # Applying the binary operator '==' (line 141)
    result_eq_641 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), '==', b_639, int_640)
    
    
    # Getting the type of 'sq' (line 141)
    sq_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'sq')
    int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'int')
    # Applying the binary operator '+' (line 141)
    result_add_644 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 25), '+', sq_642, int_643)
    
    int_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'int')
    # Applying the binary operator '&' (line 141)
    result_and__646 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 25), '&', result_add_644, int_645)
    
    # Applying the 'not' unary operator (line 141)
    result_not__647 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 20), 'not', result_and__646)
    
    # Applying the binary operator 'and' (line 141)
    result_and_keyword_648 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), 'and', result_eq_641, result_not__647)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 141)
    sq_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 51), 'sq')
    int_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 56), 'int')
    # Applying the binary operator '+' (line 141)
    result_add_651 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 51), '+', sq_649, int_650)
    
    # Getting the type of 'board' (line 141)
    board_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 45), 'board')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 45), board_652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 141, 45), getitem___653, result_add_651)
    
    int_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 63), 'int')
    # Applying the binary operator '==' (line 141)
    result_eq_656 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 45), '==', subscript_call_result_654, int_655)
    
    # Applying the binary operator 'and' (line 141)
    result_and_keyword_657 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), 'and', result_and_keyword_648, result_eq_656)
    
    # Testing the type of an if condition (line 141)
    if_condition_658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 6), result_and_keyword_657)
    # Assigning a type to the variable 'if_condition_658' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 6), 'if_condition_658', if_condition_658)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 142)
    sq_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'sq')
    int_660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 17), 'int')
    # Applying the binary operator '>=' (line 142)
    result_ge_661 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), '>=', sq_659, int_660)
    
    
    # Getting the type of 'sq' (line 142)
    sq_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'sq')
    int_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'int')
    # Applying the binary operator '<' (line 142)
    result_lt_664 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 24), '<', sq_662, int_663)
    
    # Applying the binary operator 'and' (line 142)
    result_and_keyword_665 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), 'and', result_ge_661, result_lt_664)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 142)
    sq_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'sq')
    int_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 47), 'int')
    # Applying the binary operator '+' (line 142)
    result_add_668 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 42), '+', sq_666, int_667)
    
    # Getting the type of 'board' (line 142)
    board_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'board')
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 36), board_669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 142, 36), getitem___670, result_add_668)
    
    int_672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 54), 'int')
    # Applying the binary operator '==' (line 142)
    result_eq_673 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 36), '==', subscript_call_result_671, int_672)
    
    # Applying the binary operator 'and' (line 142)
    result_and_keyword_674 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), 'and', result_and_keyword_665, result_eq_673)
    
    # Testing the type of an if condition (line 142)
    if_condition_675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), result_and_keyword_674)
    # Assigning a type to the variable 'if_condition_675' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_675', if_condition_675)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'sq' (line 143)
    sq_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'sq', False)
    int_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'int')
    # Applying the binary operator '*' (line 143)
    result_mul_680 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 24), '*', sq_678, int_679)
    
    int_681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'int')
    # Applying the binary operator '+' (line 143)
    result_add_682 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 24), '+', result_mul_680, int_681)
    
    # Processing the call keyword arguments (line 143)
    kwargs_683 = {}
    # Getting the type of 'retval' (line 143)
    retval_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 143)
    append_677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 10), retval_676, 'append')
    # Calling append(args, kwargs) (line 143)
    append_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 143, 10), append_677, *[result_add_682], **kwargs_683)
    
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'sq' (line 144)
    sq_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'sq', False)
    int_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 27), 'int')
    # Applying the binary operator '*' (line 144)
    result_mul_689 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 22), '*', sq_687, int_688)
    
    int_690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 35), 'int')
    # Applying the binary operator '+' (line 144)
    result_add_691 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 22), '+', result_mul_689, int_690)
    
    # Processing the call keyword arguments (line 144)
    kwargs_692 = {}
    # Getting the type of 'retval' (line 144)
    retval_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'retval', False)
    # Obtaining the member 'append' of a type (line 144)
    append_686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), retval_685, 'append')
    # Calling append(args, kwargs) (line 144)
    append_call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), append_686, *[result_add_691], **kwargs_692)
    
    # SSA branch for the else part of an if statement (line 141)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 145)
    b_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'b')
    int_695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'int')
    # Applying the binary operator '==' (line 145)
    result_eq_696 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '==', b_694, int_695)
    
    # Testing the type of an if condition (line 145)
    if_condition_697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 11), result_eq_696)
    # Assigning a type to the variable 'if_condition_697' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'if_condition_697', if_condition_697)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'knightMoves' (line 146)
    knightMoves_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'knightMoves')
    # Testing the type of a for loop iterable (line 146)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), knightMoves_698)
    # Getting the type of the for loop variable (line 146)
    for_loop_var_699 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), knightMoves_698)
    # Assigning a type to the variable 'k' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'k', for_loop_var_699)
    # SSA begins for a for statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 147)
    k_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'k')
    # Getting the type of 'sq' (line 147)
    sq_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'sq')
    # Applying the binary operator '+' (line 147)
    result_add_702 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), '+', k_700, sq_701)
    
    # Getting the type of 'board' (line 147)
    board_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'board')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 13), board_703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), getitem___704, result_add_702)
    
    int_706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'int')
    # Applying the binary operator '==' (line 147)
    result_eq_707 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 13), '==', subscript_call_result_705, int_706)
    
    # Testing the type of an if condition (line 147)
    if_condition_708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 10), result_eq_707)
    # Assigning a type to the variable 'if_condition_708' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 10), 'if_condition_708', if_condition_708)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'sq' (line 148)
    sq_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'sq', False)
    int_712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 31), 'int')
    # Applying the binary operator '*' (line 148)
    result_mul_713 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 26), '*', sq_711, int_712)
    
    # Getting the type of 'k' (line 148)
    k_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'k', False)
    # Applying the binary operator '+' (line 148)
    result_add_715 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 26), '+', result_mul_713, k_714)
    
    # Processing the call keyword arguments (line 148)
    kwargs_716 = {}
    # Getting the type of 'retval' (line 148)
    retval_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 148)
    append_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), retval_709, 'append')
    # Calling append(args, kwargs) (line 148)
    append_call_result_717 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), append_710, *[result_add_715], **kwargs_716)
    
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 145)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 149)
    b_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'b')
    int_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 16), 'int')
    # Applying the binary operator '==' (line 149)
    result_eq_720 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '==', b_718, int_719)
    
    
    # Getting the type of 'b' (line 149)
    b_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'b')
    int_722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 26), 'int')
    # Applying the binary operator '==' (line 149)
    result_eq_723 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 21), '==', b_721, int_722)
    
    # Applying the binary operator 'or' (line 149)
    result_or_keyword_724 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), 'or', result_eq_720, result_eq_723)
    
    # Testing the type of an if condition (line 149)
    if_condition_725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 11), result_or_keyword_724)
    # Assigning a type to the variable 'if_condition_725' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'if_condition_725', if_condition_725)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'bishopLines' (line 150)
    bishopLines_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'bishopLines')
    # Testing the type of a for loop iterable (line 150)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 8), bishopLines_726)
    # Getting the type of the for loop variable (line 150)
    for_loop_var_727 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 8), bishopLines_726)
    # Assigning a type to the variable 'line' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'line', for_loop_var_727)
    # SSA begins for a for statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 151)
    line_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'line')
    # Testing the type of a for loop iterable (line 151)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 10), line_728)
    # Getting the type of the for loop variable (line 151)
    for_loop_var_729 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 10), line_728)
    # Assigning a type to the variable 'k' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 10), 'k', for_loop_var_729)
    # SSA begins for a for statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'k' (line 152)
    k_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'k')
    # Getting the type of 'sq' (line 152)
    sq_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'sq')
    # Applying the binary operator '+' (line 152)
    result_add_732 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 16), '+', k_730, sq_731)
    
    int_733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'int')
    # Applying the binary operator '&' (line 152)
    result_and__734 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 16), '&', result_add_732, int_733)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 152)
    k_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 40), 'k')
    # Getting the type of 'sq' (line 152)
    sq_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 44), 'sq')
    # Applying the binary operator '+' (line 152)
    result_add_737 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 40), '+', k_735, sq_736)
    
    # Getting the type of 'board' (line 152)
    board_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 34), 'board')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 34), board_738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_740 = invoke(stypy.reporting.localization.Localization(__file__, 152, 34), getitem___739, result_add_737)
    
    int_741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 51), 'int')
    # Applying the binary operator '!=' (line 152)
    result_ne_742 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 34), '!=', subscript_call_result_740, int_741)
    
    # Applying the binary operator 'or' (line 152)
    result_or_keyword_743 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), 'or', result_and__734, result_ne_742)
    
    # Testing the type of an if condition (line 152)
    if_condition_744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), result_or_keyword_743)
    # Assigning a type to the variable 'if_condition_744' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'if_condition_744', if_condition_744)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'sq' (line 154)
    sq_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'sq', False)
    int_748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'int')
    # Applying the binary operator '*' (line 154)
    result_mul_749 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 26), '*', sq_747, int_748)
    
    # Getting the type of 'k' (line 154)
    k_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'k', False)
    # Applying the binary operator '+' (line 154)
    result_add_751 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 26), '+', result_mul_749, k_750)
    
    # Processing the call keyword arguments (line 154)
    kwargs_752 = {}
    # Getting the type of 'retval' (line 154)
    retval_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 154)
    append_746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), retval_745, 'append')
    # Calling append(args, kwargs) (line 154)
    append_call_result_753 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), append_746, *[result_add_751], **kwargs_752)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 155)
    b_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 9), 'b')
    int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 14), 'int')
    # Applying the binary operator '==' (line 155)
    result_eq_756 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 9), '==', b_754, int_755)
    
    
    # Getting the type of 'b' (line 155)
    b_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'b')
    int_758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'int')
    # Applying the binary operator '==' (line 155)
    result_eq_759 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 19), '==', b_757, int_758)
    
    # Applying the binary operator 'or' (line 155)
    result_or_keyword_760 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 9), 'or', result_eq_756, result_eq_759)
    
    # Testing the type of an if condition (line 155)
    if_condition_761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 6), result_or_keyword_760)
    # Assigning a type to the variable 'if_condition_761' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 6), 'if_condition_761', if_condition_761)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'rookLines' (line 156)
    rookLines_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'rookLines')
    # Testing the type of a for loop iterable (line 156)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 8), rookLines_762)
    # Getting the type of the for loop variable (line 156)
    for_loop_var_763 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 8), rookLines_762)
    # Assigning a type to the variable 'line' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'line', for_loop_var_763)
    # SSA begins for a for statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 157)
    line_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'line')
    # Testing the type of a for loop iterable (line 157)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 10), line_764)
    # Getting the type of the for loop variable (line 157)
    for_loop_var_765 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 10), line_764)
    # Assigning a type to the variable 'k' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 10), 'k', for_loop_var_765)
    # SSA begins for a for statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'k' (line 158)
    k_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'k')
    # Getting the type of 'sq' (line 158)
    sq_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'sq')
    # Applying the binary operator '+' (line 158)
    result_add_768 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 16), '+', k_766, sq_767)
    
    int_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'int')
    # Applying the binary operator '&' (line 158)
    result_and__770 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 16), '&', result_add_768, int_769)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 158)
    k_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 40), 'k')
    # Getting the type of 'sq' (line 158)
    sq_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 44), 'sq')
    # Applying the binary operator '+' (line 158)
    result_add_773 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 40), '+', k_771, sq_772)
    
    # Getting the type of 'board' (line 158)
    board_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 'board')
    # Obtaining the member '__getitem__' of a type (line 158)
    getitem___775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 34), board_774, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 158)
    subscript_call_result_776 = invoke(stypy.reporting.localization.Localization(__file__, 158, 34), getitem___775, result_add_773)
    
    int_777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 51), 'int')
    # Applying the binary operator '!=' (line 158)
    result_ne_778 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 34), '!=', subscript_call_result_776, int_777)
    
    # Applying the binary operator 'or' (line 158)
    result_or_keyword_779 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 15), 'or', result_and__770, result_ne_778)
    
    # Testing the type of an if condition (line 158)
    if_condition_780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 12), result_or_keyword_779)
    # Assigning a type to the variable 'if_condition_780' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'if_condition_780', if_condition_780)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'sq' (line 160)
    sq_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'sq', False)
    int_784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 31), 'int')
    # Applying the binary operator '*' (line 160)
    result_mul_785 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 26), '*', sq_783, int_784)
    
    # Getting the type of 'k' (line 160)
    k_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 'k', False)
    # Applying the binary operator '+' (line 160)
    result_add_787 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 26), '+', result_mul_785, k_786)
    
    # Processing the call keyword arguments (line 160)
    kwargs_788 = {}
    # Getting the type of 'retval' (line 160)
    retval_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 160)
    append_782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), retval_781, 'append')
    # Calling append(args, kwargs) (line 160)
    append_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), append_782, *[result_add_787], **kwargs_788)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 155)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 161)
    b_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'b')
    int_791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
    # Applying the binary operator '==' (line 161)
    result_eq_792 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), '==', b_790, int_791)
    
    # Testing the type of an if condition (line 161)
    if_condition_793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 11), result_eq_792)
    # Assigning a type to the variable 'if_condition_793' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'if_condition_793', if_condition_793)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'kingMoves' (line 162)
    kingMoves_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'kingMoves')
    # Testing the type of a for loop iterable (line 162)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 8), kingMoves_794)
    # Getting the type of the for loop variable (line 162)
    for_loop_var_795 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 8), kingMoves_794)
    # Assigning a type to the variable 'k' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'k', for_loop_var_795)
    # SSA begins for a for statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 163)
    k_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'k')
    # Getting the type of 'sq' (line 163)
    sq_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'sq')
    # Applying the binary operator '+' (line 163)
    result_add_798 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 18), '+', k_796, sq_797)
    
    int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 27), 'int')
    # Applying the binary operator '&' (line 163)
    result_and__800 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 18), '&', result_add_798, int_799)
    
    # Applying the 'not' unary operator (line 163)
    result_not__801 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 13), 'not', result_and__800)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 163)
    k_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), 'k')
    # Getting the type of 'sq' (line 163)
    sq_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 47), 'sq')
    # Applying the binary operator '+' (line 163)
    result_add_804 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 43), '+', k_802, sq_803)
    
    # Getting the type of 'board' (line 163)
    board_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'board')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 37), board_805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_807 = invoke(stypy.reporting.localization.Localization(__file__, 163, 37), getitem___806, result_add_804)
    
    int_808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 54), 'int')
    # Applying the binary operator '==' (line 163)
    result_eq_809 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 37), '==', subscript_call_result_807, int_808)
    
    # Applying the binary operator 'and' (line 163)
    result_and_keyword_810 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 13), 'and', result_not__801, result_eq_809)
    
    # Testing the type of an if condition (line 163)
    if_condition_811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 10), result_and_keyword_810)
    # Assigning a type to the variable 'if_condition_811' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 10), 'if_condition_811', if_condition_811)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'sq' (line 164)
    sq_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'sq', False)
    int_815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 31), 'int')
    # Applying the binary operator '*' (line 164)
    result_mul_816 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 26), '*', sq_814, int_815)
    
    # Getting the type of 'k' (line 164)
    k_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 39), 'k', False)
    # Applying the binary operator '+' (line 164)
    result_add_818 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 26), '+', result_mul_816, k_817)
    
    # Processing the call keyword arguments (line 164)
    kwargs_819 = {}
    # Getting the type of 'retval' (line 164)
    retval_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 164)
    append_813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), retval_812, 'append')
    # Calling append(args, kwargs) (line 164)
    append_call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), append_813, *[result_add_818], **kwargs_819)
    
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 140)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    int_821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 12), 'int')
    # Getting the type of 'board' (line 165)
    board_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 6), 'board')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 6), board_822, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_824 = invoke(stypy.reporting.localization.Localization(__file__, 165, 6), getitem___823, int_821)
    
    
    
    # Obtaining the type of the subscript
    int_825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 26), 'int')
    # Getting the type of 'board' (line 165)
    board_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'board')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 20), board_826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_828 = invoke(stypy.reporting.localization.Localization(__file__, 165, 20), getitem___827, int_825)
    
    int_829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
    # Applying the binary operator '==' (line 165)
    result_eq_830 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 20), '==', subscript_call_result_828, int_829)
    
    # Applying the binary operator 'and' (line 165)
    result_and_keyword_831 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 6), 'and', subscript_call_result_824, result_eq_830)
    
    
    # Obtaining the type of the subscript
    int_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 44), 'int')
    # Getting the type of 'board' (line 165)
    board_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 38), 'board')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 38), board_833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_835 = invoke(stypy.reporting.localization.Localization(__file__, 165, 38), getitem___834, int_832)
    
    int_836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 50), 'int')
    # Applying the binary operator '==' (line 165)
    result_eq_837 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 38), '==', subscript_call_result_835, int_836)
    
    # Applying the binary operator 'and' (line 165)
    result_and_keyword_838 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 6), 'and', result_and_keyword_831, result_eq_837)
    
    
    # Obtaining the type of the subscript
    int_839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 62), 'int')
    # Getting the type of 'board' (line 165)
    board_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 56), 'board')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 56), board_840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_842 = invoke(stypy.reporting.localization.Localization(__file__, 165, 56), getitem___841, int_839)
    
    int_843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 68), 'int')
    # Applying the binary operator '==' (line 165)
    result_eq_844 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 56), '==', subscript_call_result_842, int_843)
    
    # Applying the binary operator 'and' (line 165)
    result_and_keyword_845 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 6), 'and', result_and_keyword_838, result_eq_844)
    
    
    int_846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 10), 'int')
    
    # Obtaining the type of the subscript
    int_847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 22), 'int')
    int_848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 25), 'int')
    slice_849 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 16), int_847, int_848, None)
    # Getting the type of 'board' (line 166)
    board_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'board')
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), board_850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_852 = invoke(stypy.reporting.localization.Localization(__file__, 166, 16), getitem___851, slice_849)
    
    # Applying the binary operator 'in' (line 166)
    result_contains_853 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 10), 'in', int_846, subscript_call_result_852)
    
    # Applying the 'not' unary operator (line 166)
    result_not__854 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 6), 'not', result_contains_853)
    
    # Applying the binary operator 'and' (line 165)
    result_and_keyword_855 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 6), 'and', result_and_keyword_845, result_not__854)
    
    
    # Call to nonpawnBlackAttacks(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'board' (line 167)
    board_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'board', False)
    int_858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 37), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_859 = {}
    # Getting the type of 'nonpawnBlackAttacks' (line 167)
    nonpawnBlackAttacks_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 10), 'nonpawnBlackAttacks', False)
    # Calling nonpawnBlackAttacks(args, kwargs) (line 167)
    nonpawnBlackAttacks_call_result_860 = invoke(stypy.reporting.localization.Localization(__file__, 167, 10), nonpawnBlackAttacks_856, *[board_857, int_858], **kwargs_859)
    
    # Applying the 'not' unary operator (line 167)
    result_not__861 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 6), 'not', nonpawnBlackAttacks_call_result_860)
    
    # Applying the binary operator 'and' (line 165)
    result_and_keyword_862 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 6), 'and', result_and_keyword_855, result_not__861)
    
    
    # Call to nonpawnBlackAttacks(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'board' (line 167)
    board_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 68), 'board', False)
    int_865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 75), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_866 = {}
    # Getting the type of 'nonpawnBlackAttacks' (line 167)
    nonpawnBlackAttacks_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 48), 'nonpawnBlackAttacks', False)
    # Calling nonpawnBlackAttacks(args, kwargs) (line 167)
    nonpawnBlackAttacks_call_result_867 = invoke(stypy.reporting.localization.Localization(__file__, 167, 48), nonpawnBlackAttacks_863, *[board_864, int_865], **kwargs_866)
    
    # Applying the 'not' unary operator (line 167)
    result_not__868 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 44), 'not', nonpawnBlackAttacks_call_result_867)
    
    # Applying the binary operator 'and' (line 165)
    result_and_keyword_869 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 6), 'and', result_and_keyword_862, result_not__868)
    
    
    # Call to nonpawnBlackAttacks(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'board' (line 167)
    board_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 106), 'board', False)
    int_872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 113), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_873 = {}
    # Getting the type of 'nonpawnBlackAttacks' (line 167)
    nonpawnBlackAttacks_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 86), 'nonpawnBlackAttacks', False)
    # Calling nonpawnBlackAttacks(args, kwargs) (line 167)
    nonpawnBlackAttacks_call_result_874 = invoke(stypy.reporting.localization.Localization(__file__, 167, 86), nonpawnBlackAttacks_870, *[board_871, int_872], **kwargs_873)
    
    # Applying the 'not' unary operator (line 167)
    result_not__875 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 82), 'not', nonpawnBlackAttacks_call_result_874)
    
    # Applying the binary operator 'and' (line 165)
    result_and_keyword_876 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 6), 'and', result_and_keyword_869, result_not__875)
    
    # Testing the type of an if condition (line 165)
    if_condition_877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 2), result_and_keyword_876)
    # Assigning a type to the variable 'if_condition_877' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 2), 'if_condition_877', if_condition_877)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 168)
    # Processing the call arguments (line 168)
    int_880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'int')
    int_881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 31), 'int')
    int_882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 35), 'int')
    # Applying the binary operator '*' (line 168)
    result_mul_883 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 31), '*', int_881, int_882)
    
    # Applying the binary operator '+' (line 168)
    result_add_884 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 18), '+', int_880, result_mul_883)
    
    int_885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 43), 'int')
    # Applying the binary operator '-' (line 168)
    result_sub_886 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 41), '-', result_add_884, int_885)
    
    # Processing the call keyword arguments (line 168)
    kwargs_887 = {}
    # Getting the type of 'retval' (line 168)
    retval_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'retval', False)
    # Obtaining the member 'append' of a type (line 168)
    append_879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), retval_878, 'append')
    # Calling append(args, kwargs) (line 168)
    append_call_result_888 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), append_879, *[result_sub_886], **kwargs_887)
    
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    int_889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    # Getting the type of 'board' (line 169)
    board_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 6), 'board')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 6), board_890, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_892 = invoke(stypy.reporting.localization.Localization(__file__, 169, 6), getitem___891, int_889)
    
    
    
    # Obtaining the type of the subscript
    int_893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'int')
    # Getting the type of 'board' (line 169)
    board_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'board')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), board_894, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_896 = invoke(stypy.reporting.localization.Localization(__file__, 169, 20), getitem___895, int_893)
    
    int_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_898 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 20), '==', subscript_call_result_896, int_897)
    
    # Applying the binary operator 'and' (line 169)
    result_and_keyword_899 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 6), 'and', subscript_call_result_892, result_eq_898)
    
    
    # Obtaining the type of the subscript
    int_900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 44), 'int')
    # Getting the type of 'board' (line 169)
    board_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 38), 'board')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 38), board_901, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_903 = invoke(stypy.reporting.localization.Localization(__file__, 169, 38), getitem___902, int_900)
    
    int_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 50), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_905 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 38), '==', subscript_call_result_903, int_904)
    
    # Applying the binary operator 'and' (line 169)
    result_and_keyword_906 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 6), 'and', result_and_keyword_899, result_eq_905)
    
    
    int_907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 10), 'int')
    
    # Obtaining the type of the subscript
    int_908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'int')
    int_909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'int')
    slice_910 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 170, 16), int_908, int_909, None)
    # Getting the type of 'board' (line 170)
    board_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'board')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), board_911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_913 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), getitem___912, slice_910)
    
    # Applying the binary operator 'in' (line 170)
    result_contains_914 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 10), 'in', int_907, subscript_call_result_913)
    
    # Applying the 'not' unary operator (line 170)
    result_not__915 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 6), 'not', result_contains_914)
    
    # Applying the binary operator 'and' (line 169)
    result_and_keyword_916 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 6), 'and', result_and_keyword_906, result_not__915)
    
    
    # Call to nonpawnBlackAttacks(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'board' (line 171)
    board_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'board', False)
    int_919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 37), 'int')
    # Processing the call keyword arguments (line 171)
    kwargs_920 = {}
    # Getting the type of 'nonpawnBlackAttacks' (line 171)
    nonpawnBlackAttacks_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 10), 'nonpawnBlackAttacks', False)
    # Calling nonpawnBlackAttacks(args, kwargs) (line 171)
    nonpawnBlackAttacks_call_result_921 = invoke(stypy.reporting.localization.Localization(__file__, 171, 10), nonpawnBlackAttacks_917, *[board_918, int_919], **kwargs_920)
    
    # Applying the 'not' unary operator (line 171)
    result_not__922 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 6), 'not', nonpawnBlackAttacks_call_result_921)
    
    # Applying the binary operator 'and' (line 169)
    result_and_keyword_923 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 6), 'and', result_and_keyword_916, result_not__922)
    
    
    # Call to nonpawnBlackAttacks(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'board' (line 171)
    board_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 68), 'board', False)
    int_926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 75), 'int')
    # Processing the call keyword arguments (line 171)
    kwargs_927 = {}
    # Getting the type of 'nonpawnBlackAttacks' (line 171)
    nonpawnBlackAttacks_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'nonpawnBlackAttacks', False)
    # Calling nonpawnBlackAttacks(args, kwargs) (line 171)
    nonpawnBlackAttacks_call_result_928 = invoke(stypy.reporting.localization.Localization(__file__, 171, 48), nonpawnBlackAttacks_924, *[board_925, int_926], **kwargs_927)
    
    # Applying the 'not' unary operator (line 171)
    result_not__929 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 44), 'not', nonpawnBlackAttacks_call_result_928)
    
    # Applying the binary operator 'and' (line 169)
    result_and_keyword_930 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 6), 'and', result_and_keyword_923, result_not__929)
    
    
    # Call to nonpawnBlackAttacks(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'board' (line 171)
    board_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 106), 'board', False)
    int_933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 113), 'int')
    # Processing the call keyword arguments (line 171)
    kwargs_934 = {}
    # Getting the type of 'nonpawnBlackAttacks' (line 171)
    nonpawnBlackAttacks_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 86), 'nonpawnBlackAttacks', False)
    # Calling nonpawnBlackAttacks(args, kwargs) (line 171)
    nonpawnBlackAttacks_call_result_935 = invoke(stypy.reporting.localization.Localization(__file__, 171, 86), nonpawnBlackAttacks_931, *[board_932, int_933], **kwargs_934)
    
    # Applying the 'not' unary operator (line 171)
    result_not__936 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 82), 'not', nonpawnBlackAttacks_call_result_935)
    
    # Applying the binary operator 'and' (line 169)
    result_and_keyword_937 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 6), 'and', result_and_keyword_930, result_not__936)
    
    # Testing the type of an if condition (line 169)
    if_condition_938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 2), result_and_keyword_937)
    # Assigning a type to the variable 'if_condition_938' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 2), 'if_condition_938', if_condition_938)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 172)
    # Processing the call arguments (line 172)
    int_941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'int')
    int_942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 31), 'int')
    int_943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 35), 'int')
    # Applying the binary operator '*' (line 172)
    result_mul_944 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 31), '*', int_942, int_943)
    
    # Applying the binary operator '+' (line 172)
    result_add_945 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 18), '+', int_941, result_mul_944)
    
    int_946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 43), 'int')
    # Applying the binary operator '+' (line 172)
    result_add_947 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 41), '+', result_add_945, int_946)
    
    # Processing the call keyword arguments (line 172)
    kwargs_948 = {}
    # Getting the type of 'retval' (line 172)
    retval_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'retval', False)
    # Obtaining the member 'append' of a type (line 172)
    append_940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 4), retval_939, 'append')
    # Calling append(args, kwargs) (line 172)
    append_call_result_949 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), append_940, *[result_add_947], **kwargs_948)
    
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'retval' (line 173)
    retval_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 9), 'retval')
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 2), 'stypy_return_type', retval_950)
    
    # ################# End of 'pseudoLegalMovesWhite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pseudoLegalMovesWhite' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_951)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pseudoLegalMovesWhite'
    return stypy_return_type_951

# Assigning a type to the variable 'pseudoLegalMovesWhite' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'pseudoLegalMovesWhite', pseudoLegalMovesWhite)

@norecursion
def pseudoLegalMovesBlack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pseudoLegalMovesBlack'
    module_type_store = module_type_store.open_function_context('pseudoLegalMovesBlack', 175, 0, False)
    
    # Passed parameters checking function
    pseudoLegalMovesBlack.stypy_localization = localization
    pseudoLegalMovesBlack.stypy_type_of_self = None
    pseudoLegalMovesBlack.stypy_type_store = module_type_store
    pseudoLegalMovesBlack.stypy_function_name = 'pseudoLegalMovesBlack'
    pseudoLegalMovesBlack.stypy_param_names_list = ['board']
    pseudoLegalMovesBlack.stypy_varargs_param_name = None
    pseudoLegalMovesBlack.stypy_kwargs_param_name = None
    pseudoLegalMovesBlack.stypy_call_defaults = defaults
    pseudoLegalMovesBlack.stypy_call_varargs = varargs
    pseudoLegalMovesBlack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pseudoLegalMovesBlack', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pseudoLegalMovesBlack', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pseudoLegalMovesBlack(...)' code ##################

    
    # Assigning a Call to a Name (line 176):
    
    # Call to pseudoLegalCapturesBlack(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'board' (line 176)
    board_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'board', False)
    # Processing the call keyword arguments (line 176)
    kwargs_954 = {}
    # Getting the type of 'pseudoLegalCapturesBlack' (line 176)
    pseudoLegalCapturesBlack_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'pseudoLegalCapturesBlack', False)
    # Calling pseudoLegalCapturesBlack(args, kwargs) (line 176)
    pseudoLegalCapturesBlack_call_result_955 = invoke(stypy.reporting.localization.Localization(__file__, 176, 11), pseudoLegalCapturesBlack_952, *[board_953], **kwargs_954)
    
    # Assigning a type to the variable 'retval' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 2), 'retval', pseudoLegalCapturesBlack_call_result_955)
    
    # Getting the type of 'squares' (line 177)
    squares_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'squares')
    # Testing the type of a for loop iterable (line 177)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 2), squares_956)
    # Getting the type of the for loop variable (line 177)
    for_loop_var_957 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 2), squares_956)
    # Assigning a type to the variable 'sq' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 2), 'sq', for_loop_var_957)
    # SSA begins for a for statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 178)
    sq_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 14), 'sq')
    # Getting the type of 'board' (line 178)
    board_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), board_959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_961 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), getitem___960, sq_958)
    
    # Assigning a type to the variable 'b' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'b', subscript_call_result_961)
    
    
    # Getting the type of 'b' (line 179)
    b_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 7), 'b')
    int_963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 11), 'int')
    # Applying the binary operator '<' (line 179)
    result_lt_964 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 7), '<', b_962, int_963)
    
    # Testing the type of an if condition (line 179)
    if_condition_965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 4), result_lt_964)
    # Assigning a type to the variable 'if_condition_965' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'if_condition_965', if_condition_965)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 180)
    b_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 9), 'b')
    int_967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 14), 'int')
    # Applying the binary operator '==' (line 180)
    result_eq_968 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 9), '==', b_966, int_967)
    
    
    # Getting the type of 'sq' (line 180)
    sq_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 26), 'sq')
    int_970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'int')
    # Applying the binary operator '+' (line 180)
    result_add_971 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 26), '+', sq_969, int_970)
    
    int_972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 36), 'int')
    # Applying the binary operator '&' (line 180)
    result_and__973 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 26), '&', result_add_971, int_972)
    
    # Applying the 'not' unary operator (line 180)
    result_not__974 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 21), 'not', result_and__973)
    
    # Applying the binary operator 'and' (line 180)
    result_and_keyword_975 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 9), 'and', result_eq_968, result_not__974)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 180)
    sq_976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 52), 'sq')
    int_977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 57), 'int')
    # Applying the binary operator '-' (line 180)
    result_sub_978 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 52), '-', sq_976, int_977)
    
    # Getting the type of 'board' (line 180)
    board_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 46), 'board')
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 46), board_979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_981 = invoke(stypy.reporting.localization.Localization(__file__, 180, 46), getitem___980, result_sub_978)
    
    int_982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 64), 'int')
    # Applying the binary operator '==' (line 180)
    result_eq_983 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 46), '==', subscript_call_result_981, int_982)
    
    # Applying the binary operator 'and' (line 180)
    result_and_keyword_984 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 9), 'and', result_and_keyword_975, result_eq_983)
    
    # Testing the type of an if condition (line 180)
    if_condition_985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 6), result_and_keyword_984)
    # Assigning a type to the variable 'if_condition_985' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 6), 'if_condition_985', if_condition_985)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 181)
    sq_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'sq')
    int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'int')
    # Applying the binary operator '>=' (line 181)
    result_ge_988 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), '>=', sq_986, int_987)
    
    
    # Getting the type of 'sq' (line 181)
    sq_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'sq')
    int_990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'int')
    # Applying the binary operator '<' (line 181)
    result_lt_991 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 24), '<', sq_989, int_990)
    
    # Applying the binary operator 'and' (line 181)
    result_and_keyword_992 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'and', result_ge_988, result_lt_991)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 181)
    sq_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 43), 'sq')
    int_994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 48), 'int')
    # Applying the binary operator '-' (line 181)
    result_sub_995 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 43), '-', sq_993, int_994)
    
    # Getting the type of 'board' (line 181)
    board_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 37), 'board')
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 37), board_996, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_998 = invoke(stypy.reporting.localization.Localization(__file__, 181, 37), getitem___997, result_sub_995)
    
    int_999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 55), 'int')
    # Applying the binary operator '==' (line 181)
    result_eq_1000 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 37), '==', subscript_call_result_998, int_999)
    
    # Applying the binary operator 'and' (line 181)
    result_and_keyword_1001 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'and', result_and_keyword_992, result_eq_1000)
    
    # Testing the type of an if condition (line 181)
    if_condition_1002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_and_keyword_1001)
    # Assigning a type to the variable 'if_condition_1002' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_1002', if_condition_1002)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'sq' (line 182)
    sq_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'sq', False)
    int_1006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 29), 'int')
    # Applying the binary operator '*' (line 182)
    result_mul_1007 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 24), '*', sq_1005, int_1006)
    
    int_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'int')
    # Applying the binary operator '-' (line 182)
    result_sub_1009 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 24), '-', result_mul_1007, int_1008)
    
    # Processing the call keyword arguments (line 182)
    kwargs_1010 = {}
    # Getting the type of 'retval' (line 182)
    retval_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 182)
    append_1004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 10), retval_1003, 'append')
    # Calling append(args, kwargs) (line 182)
    append_call_result_1011 = invoke(stypy.reporting.localization.Localization(__file__, 182, 10), append_1004, *[result_sub_1009], **kwargs_1010)
    
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'sq' (line 183)
    sq_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'sq', False)
    int_1015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 27), 'int')
    # Applying the binary operator '*' (line 183)
    result_mul_1016 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 22), '*', sq_1014, int_1015)
    
    int_1017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 35), 'int')
    # Applying the binary operator '-' (line 183)
    result_sub_1018 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 22), '-', result_mul_1016, int_1017)
    
    # Processing the call keyword arguments (line 183)
    kwargs_1019 = {}
    # Getting the type of 'retval' (line 183)
    retval_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'retval', False)
    # Obtaining the member 'append' of a type (line 183)
    append_1013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), retval_1012, 'append')
    # Calling append(args, kwargs) (line 183)
    append_call_result_1020 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), append_1013, *[result_sub_1018], **kwargs_1019)
    
    # SSA branch for the else part of an if statement (line 180)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 184)
    b_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'b')
    int_1022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 16), 'int')
    # Applying the binary operator '==' (line 184)
    result_eq_1023 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '==', b_1021, int_1022)
    
    # Testing the type of an if condition (line 184)
    if_condition_1024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 11), result_eq_1023)
    # Assigning a type to the variable 'if_condition_1024' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'if_condition_1024', if_condition_1024)
    # SSA begins for if statement (line 184)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'knightMoves' (line 185)
    knightMoves_1025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'knightMoves')
    # Testing the type of a for loop iterable (line 185)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 8), knightMoves_1025)
    # Getting the type of the for loop variable (line 185)
    for_loop_var_1026 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 8), knightMoves_1025)
    # Assigning a type to the variable 'k' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'k', for_loop_var_1026)
    # SSA begins for a for statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 186)
    k_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'k')
    # Getting the type of 'sq' (line 186)
    sq_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 'sq')
    # Applying the binary operator '+' (line 186)
    result_add_1029 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 19), '+', k_1027, sq_1028)
    
    # Getting the type of 'board' (line 186)
    board_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'board')
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___1031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 13), board_1030, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_1032 = invoke(stypy.reporting.localization.Localization(__file__, 186, 13), getitem___1031, result_add_1029)
    
    int_1033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 30), 'int')
    # Applying the binary operator '==' (line 186)
    result_eq_1034 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 13), '==', subscript_call_result_1032, int_1033)
    
    # Testing the type of an if condition (line 186)
    if_condition_1035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 10), result_eq_1034)
    # Assigning a type to the variable 'if_condition_1035' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 10), 'if_condition_1035', if_condition_1035)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'sq' (line 187)
    sq_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'sq', False)
    int_1039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 31), 'int')
    # Applying the binary operator '*' (line 187)
    result_mul_1040 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 26), '*', sq_1038, int_1039)
    
    # Getting the type of 'k' (line 187)
    k_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 39), 'k', False)
    # Applying the binary operator '+' (line 187)
    result_add_1042 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 26), '+', result_mul_1040, k_1041)
    
    # Processing the call keyword arguments (line 187)
    kwargs_1043 = {}
    # Getting the type of 'retval' (line 187)
    retval_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 187)
    append_1037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), retval_1036, 'append')
    # Calling append(args, kwargs) (line 187)
    append_call_result_1044 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), append_1037, *[result_add_1042], **kwargs_1043)
    
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 184)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 188)
    b_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'b')
    int_1046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 16), 'int')
    # Applying the binary operator '==' (line 188)
    result_eq_1047 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), '==', b_1045, int_1046)
    
    
    # Getting the type of 'b' (line 188)
    b_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 22), 'b')
    int_1049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 27), 'int')
    # Applying the binary operator '==' (line 188)
    result_eq_1050 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 22), '==', b_1048, int_1049)
    
    # Applying the binary operator 'or' (line 188)
    result_or_keyword_1051 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), 'or', result_eq_1047, result_eq_1050)
    
    # Testing the type of an if condition (line 188)
    if_condition_1052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 11), result_or_keyword_1051)
    # Assigning a type to the variable 'if_condition_1052' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'if_condition_1052', if_condition_1052)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'bishopLines' (line 189)
    bishopLines_1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'bishopLines')
    # Testing the type of a for loop iterable (line 189)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 189, 8), bishopLines_1053)
    # Getting the type of the for loop variable (line 189)
    for_loop_var_1054 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 189, 8), bishopLines_1053)
    # Assigning a type to the variable 'line' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'line', for_loop_var_1054)
    # SSA begins for a for statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 190)
    line_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'line')
    # Testing the type of a for loop iterable (line 190)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 10), line_1055)
    # Getting the type of the for loop variable (line 190)
    for_loop_var_1056 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 10), line_1055)
    # Assigning a type to the variable 'k' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 10), 'k', for_loop_var_1056)
    # SSA begins for a for statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'k' (line 191)
    k_1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'k')
    # Getting the type of 'sq' (line 191)
    sq_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'sq')
    # Applying the binary operator '+' (line 191)
    result_add_1059 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '+', k_1057, sq_1058)
    
    int_1060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 25), 'int')
    # Applying the binary operator '&' (line 191)
    result_and__1061 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '&', result_add_1059, int_1060)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 191)
    k_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 40), 'k')
    # Getting the type of 'sq' (line 191)
    sq_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 44), 'sq')
    # Applying the binary operator '+' (line 191)
    result_add_1064 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 40), '+', k_1062, sq_1063)
    
    # Getting the type of 'board' (line 191)
    board_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 34), 'board')
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___1066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 34), board_1065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_1067 = invoke(stypy.reporting.localization.Localization(__file__, 191, 34), getitem___1066, result_add_1064)
    
    int_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 51), 'int')
    # Applying the binary operator '!=' (line 191)
    result_ne_1069 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 34), '!=', subscript_call_result_1067, int_1068)
    
    # Applying the binary operator 'or' (line 191)
    result_or_keyword_1070 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), 'or', result_and__1061, result_ne_1069)
    
    # Testing the type of an if condition (line 191)
    if_condition_1071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 12), result_or_keyword_1070)
    # Assigning a type to the variable 'if_condition_1071' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'if_condition_1071', if_condition_1071)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'sq' (line 193)
    sq_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'sq', False)
    int_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 31), 'int')
    # Applying the binary operator '*' (line 193)
    result_mul_1076 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 26), '*', sq_1074, int_1075)
    
    # Getting the type of 'k' (line 193)
    k_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 39), 'k', False)
    # Applying the binary operator '+' (line 193)
    result_add_1078 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 26), '+', result_mul_1076, k_1077)
    
    # Processing the call keyword arguments (line 193)
    kwargs_1079 = {}
    # Getting the type of 'retval' (line 193)
    retval_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 193)
    append_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), retval_1072, 'append')
    # Calling append(args, kwargs) (line 193)
    append_call_result_1080 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), append_1073, *[result_add_1078], **kwargs_1079)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 184)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'b' (line 195)
    b_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 9), 'b')
    int_1082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 14), 'int')
    # Applying the binary operator '==' (line 195)
    result_eq_1083 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 9), '==', b_1081, int_1082)
    
    
    # Getting the type of 'b' (line 195)
    b_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'b')
    int_1085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'int')
    # Applying the binary operator '==' (line 195)
    result_eq_1086 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 20), '==', b_1084, int_1085)
    
    # Applying the binary operator 'or' (line 195)
    result_or_keyword_1087 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 9), 'or', result_eq_1083, result_eq_1086)
    
    # Testing the type of an if condition (line 195)
    if_condition_1088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 6), result_or_keyword_1087)
    # Assigning a type to the variable 'if_condition_1088' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 6), 'if_condition_1088', if_condition_1088)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'rookLines' (line 196)
    rookLines_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'rookLines')
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 8), rookLines_1089)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_1090 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 8), rookLines_1089)
    # Assigning a type to the variable 'line' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'line', for_loop_var_1090)
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 197)
    line_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'line')
    # Testing the type of a for loop iterable (line 197)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 197, 10), line_1091)
    # Getting the type of the for loop variable (line 197)
    for_loop_var_1092 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 197, 10), line_1091)
    # Assigning a type to the variable 'k' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 10), 'k', for_loop_var_1092)
    # SSA begins for a for statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'k' (line 198)
    k_1093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'k')
    # Getting the type of 'sq' (line 198)
    sq_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'sq')
    # Applying the binary operator '+' (line 198)
    result_add_1095 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 16), '+', k_1093, sq_1094)
    
    int_1096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 25), 'int')
    # Applying the binary operator '&' (line 198)
    result_and__1097 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 16), '&', result_add_1095, int_1096)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 198)
    k_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'k')
    # Getting the type of 'sq' (line 198)
    sq_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 'sq')
    # Applying the binary operator '+' (line 198)
    result_add_1100 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 40), '+', k_1098, sq_1099)
    
    # Getting the type of 'board' (line 198)
    board_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 34), 'board')
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___1102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 34), board_1101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_1103 = invoke(stypy.reporting.localization.Localization(__file__, 198, 34), getitem___1102, result_add_1100)
    
    int_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 51), 'int')
    # Applying the binary operator '!=' (line 198)
    result_ne_1105 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 34), '!=', subscript_call_result_1103, int_1104)
    
    # Applying the binary operator 'or' (line 198)
    result_or_keyword_1106 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 15), 'or', result_and__1097, result_ne_1105)
    
    # Testing the type of an if condition (line 198)
    if_condition_1107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 12), result_or_keyword_1106)
    # Assigning a type to the variable 'if_condition_1107' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'if_condition_1107', if_condition_1107)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'sq' (line 200)
    sq_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'sq', False)
    int_1111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 31), 'int')
    # Applying the binary operator '*' (line 200)
    result_mul_1112 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 26), '*', sq_1110, int_1111)
    
    # Getting the type of 'k' (line 200)
    k_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 39), 'k', False)
    # Applying the binary operator '+' (line 200)
    result_add_1114 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 26), '+', result_mul_1112, k_1113)
    
    # Processing the call keyword arguments (line 200)
    kwargs_1115 = {}
    # Getting the type of 'retval' (line 200)
    retval_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 200)
    append_1109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), retval_1108, 'append')
    # Calling append(args, kwargs) (line 200)
    append_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), append_1109, *[result_add_1114], **kwargs_1115)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 195)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 201)
    b_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'b')
    int_1118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'int')
    # Applying the binary operator '==' (line 201)
    result_eq_1119 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), '==', b_1117, int_1118)
    
    # Testing the type of an if condition (line 201)
    if_condition_1120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 11), result_eq_1119)
    # Assigning a type to the variable 'if_condition_1120' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'if_condition_1120', if_condition_1120)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'kingMoves' (line 202)
    kingMoves_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 'kingMoves')
    # Testing the type of a for loop iterable (line 202)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 202, 8), kingMoves_1121)
    # Getting the type of the for loop variable (line 202)
    for_loop_var_1122 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 202, 8), kingMoves_1121)
    # Assigning a type to the variable 'k' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'k', for_loop_var_1122)
    # SSA begins for a for statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 203)
    k_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'k')
    # Getting the type of 'sq' (line 203)
    sq_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'sq')
    # Applying the binary operator '+' (line 203)
    result_add_1125 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 18), '+', k_1123, sq_1124)
    
    int_1126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 27), 'int')
    # Applying the binary operator '&' (line 203)
    result_and__1127 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 18), '&', result_add_1125, int_1126)
    
    # Applying the 'not' unary operator (line 203)
    result_not__1128 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 13), 'not', result_and__1127)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 203)
    k_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 43), 'k')
    # Getting the type of 'sq' (line 203)
    sq_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'sq')
    # Applying the binary operator '+' (line 203)
    result_add_1131 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 43), '+', k_1129, sq_1130)
    
    # Getting the type of 'board' (line 203)
    board_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 37), 'board')
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___1133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 37), board_1132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_1134 = invoke(stypy.reporting.localization.Localization(__file__, 203, 37), getitem___1133, result_add_1131)
    
    int_1135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 54), 'int')
    # Applying the binary operator '==' (line 203)
    result_eq_1136 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 37), '==', subscript_call_result_1134, int_1135)
    
    # Applying the binary operator 'and' (line 203)
    result_and_keyword_1137 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 13), 'and', result_not__1128, result_eq_1136)
    
    # Testing the type of an if condition (line 203)
    if_condition_1138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 10), result_and_keyword_1137)
    # Assigning a type to the variable 'if_condition_1138' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 10), 'if_condition_1138', if_condition_1138)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'sq' (line 204)
    sq_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'sq', False)
    int_1142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 31), 'int')
    # Applying the binary operator '*' (line 204)
    result_mul_1143 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 26), '*', sq_1141, int_1142)
    
    # Getting the type of 'k' (line 204)
    k_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 39), 'k', False)
    # Applying the binary operator '+' (line 204)
    result_add_1145 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 26), '+', result_mul_1143, k_1144)
    
    # Processing the call keyword arguments (line 204)
    kwargs_1146 = {}
    # Getting the type of 'retval' (line 204)
    retval_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 204)
    append_1140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), retval_1139, 'append')
    # Calling append(args, kwargs) (line 204)
    append_call_result_1147 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), append_1140, *[result_add_1145], **kwargs_1146)
    
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    int_1148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 12), 'int')
    # Getting the type of 'board' (line 205)
    board_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 6), 'board')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___1150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 6), board_1149, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_1151 = invoke(stypy.reporting.localization.Localization(__file__, 205, 6), getitem___1150, int_1148)
    
    
    
    # Obtaining the type of the subscript
    int_1152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'int')
    # Getting the type of 'board' (line 205)
    board_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'board')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___1154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 20), board_1153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), getitem___1154, int_1152)
    
    int_1156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 35), 'int')
    # Applying the binary operator '==' (line 205)
    result_eq_1157 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 20), '==', subscript_call_result_1155, int_1156)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_1158 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 6), 'and', subscript_call_result_1151, result_eq_1157)
    
    
    # Obtaining the type of the subscript
    int_1159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 47), 'int')
    # Getting the type of 'board' (line 205)
    board_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 41), 'board')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___1161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 41), board_1160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_1162 = invoke(stypy.reporting.localization.Localization(__file__, 205, 41), getitem___1161, int_1159)
    
    int_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 56), 'int')
    # Applying the binary operator '==' (line 205)
    result_eq_1164 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 41), '==', subscript_call_result_1162, int_1163)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_1165 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 6), 'and', result_and_keyword_1158, result_eq_1164)
    
    
    # Obtaining the type of the subscript
    int_1166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 68), 'int')
    # Getting the type of 'board' (line 205)
    board_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 62), 'board')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___1168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 62), board_1167, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_1169 = invoke(stypy.reporting.localization.Localization(__file__, 205, 62), getitem___1168, int_1166)
    
    int_1170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 77), 'int')
    # Applying the binary operator '==' (line 205)
    result_eq_1171 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 62), '==', subscript_call_result_1169, int_1170)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_1172 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 6), 'and', result_and_keyword_1165, result_eq_1171)
    
    
    int_1173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 10), 'int')
    
    # Obtaining the type of the subscript
    int_1174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'int')
    int_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 26), 'int')
    slice_1176 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 206, 15), int_1174, int_1175, None)
    # Getting the type of 'board' (line 206)
    board_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___1178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 15), board_1177, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), getitem___1178, slice_1176)
    
    # Applying the binary operator 'in' (line 206)
    result_contains_1180 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 10), 'in', int_1173, subscript_call_result_1179)
    
    # Applying the 'not' unary operator (line 206)
    result_not__1181 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 6), 'not', result_contains_1180)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_1182 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 6), 'and', result_and_keyword_1172, result_not__1181)
    
    
    # Call to nonpawnWhiteAttacks(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'board' (line 207)
    board_1184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 30), 'board', False)
    int_1185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 37), 'int')
    # Processing the call keyword arguments (line 207)
    kwargs_1186 = {}
    # Getting the type of 'nonpawnWhiteAttacks' (line 207)
    nonpawnWhiteAttacks_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 10), 'nonpawnWhiteAttacks', False)
    # Calling nonpawnWhiteAttacks(args, kwargs) (line 207)
    nonpawnWhiteAttacks_call_result_1187 = invoke(stypy.reporting.localization.Localization(__file__, 207, 10), nonpawnWhiteAttacks_1183, *[board_1184, int_1185], **kwargs_1186)
    
    # Applying the 'not' unary operator (line 207)
    result_not__1188 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 6), 'not', nonpawnWhiteAttacks_call_result_1187)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_1189 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 6), 'and', result_and_keyword_1182, result_not__1188)
    
    
    # Call to nonpawnWhiteAttacks(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'board' (line 207)
    board_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 71), 'board', False)
    int_1192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 78), 'int')
    # Processing the call keyword arguments (line 207)
    kwargs_1193 = {}
    # Getting the type of 'nonpawnWhiteAttacks' (line 207)
    nonpawnWhiteAttacks_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 51), 'nonpawnWhiteAttacks', False)
    # Calling nonpawnWhiteAttacks(args, kwargs) (line 207)
    nonpawnWhiteAttacks_call_result_1194 = invoke(stypy.reporting.localization.Localization(__file__, 207, 51), nonpawnWhiteAttacks_1190, *[board_1191, int_1192], **kwargs_1193)
    
    # Applying the 'not' unary operator (line 207)
    result_not__1195 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 47), 'not', nonpawnWhiteAttacks_call_result_1194)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_1196 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 6), 'and', result_and_keyword_1189, result_not__1195)
    
    
    # Call to nonpawnWhiteAttacks(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'board' (line 207)
    board_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 112), 'board', False)
    int_1199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 119), 'int')
    # Processing the call keyword arguments (line 207)
    kwargs_1200 = {}
    # Getting the type of 'nonpawnWhiteAttacks' (line 207)
    nonpawnWhiteAttacks_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 92), 'nonpawnWhiteAttacks', False)
    # Calling nonpawnWhiteAttacks(args, kwargs) (line 207)
    nonpawnWhiteAttacks_call_result_1201 = invoke(stypy.reporting.localization.Localization(__file__, 207, 92), nonpawnWhiteAttacks_1197, *[board_1198, int_1199], **kwargs_1200)
    
    # Applying the 'not' unary operator (line 207)
    result_not__1202 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 88), 'not', nonpawnWhiteAttacks_call_result_1201)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_1203 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 6), 'and', result_and_keyword_1196, result_not__1202)
    
    # Testing the type of an if condition (line 205)
    if_condition_1204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 2), result_and_keyword_1203)
    # Assigning a type to the variable 'if_condition_1204' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 2), 'if_condition_1204', if_condition_1204)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 208)
    # Processing the call arguments (line 208)
    int_1207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 18), 'int')
    int_1208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 31), 'int')
    int_1209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 38), 'int')
    # Applying the binary operator '*' (line 208)
    result_mul_1210 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 31), '*', int_1208, int_1209)
    
    # Applying the binary operator '+' (line 208)
    result_add_1211 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 18), '+', int_1207, result_mul_1210)
    
    int_1212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 46), 'int')
    # Applying the binary operator '-' (line 208)
    result_sub_1213 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 44), '-', result_add_1211, int_1212)
    
    # Processing the call keyword arguments (line 208)
    kwargs_1214 = {}
    # Getting the type of 'retval' (line 208)
    retval_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'retval', False)
    # Obtaining the member 'append' of a type (line 208)
    append_1206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 4), retval_1205, 'append')
    # Calling append(args, kwargs) (line 208)
    append_call_result_1215 = invoke(stypy.reporting.localization.Localization(__file__, 208, 4), append_1206, *[result_sub_1213], **kwargs_1214)
    
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    int_1216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 12), 'int')
    # Getting the type of 'board' (line 209)
    board_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 6), 'board')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___1218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 6), board_1217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_1219 = invoke(stypy.reporting.localization.Localization(__file__, 209, 6), getitem___1218, int_1216)
    
    
    
    # Obtaining the type of the subscript
    int_1220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 26), 'int')
    # Getting the type of 'board' (line 209)
    board_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'board')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___1222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), board_1221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_1223 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), getitem___1222, int_1220)
    
    int_1224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 35), 'int')
    # Applying the binary operator '==' (line 209)
    result_eq_1225 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 20), '==', subscript_call_result_1223, int_1224)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_1226 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 6), 'and', subscript_call_result_1219, result_eq_1225)
    
    
    # Obtaining the type of the subscript
    int_1227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 47), 'int')
    # Getting the type of 'board' (line 209)
    board_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 41), 'board')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___1229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 41), board_1228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 209, 41), getitem___1229, int_1227)
    
    int_1231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 56), 'int')
    # Applying the binary operator '==' (line 209)
    result_eq_1232 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 41), '==', subscript_call_result_1230, int_1231)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_1233 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 6), 'and', result_and_keyword_1226, result_eq_1232)
    
    
    int_1234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 10), 'int')
    
    # Obtaining the type of the subscript
    int_1235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'int')
    int_1236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 27), 'int')
    slice_1237 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 210, 16), int_1235, int_1236, None)
    # Getting the type of 'board' (line 210)
    board_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'board')
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___1239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), board_1238, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_1240 = invoke(stypy.reporting.localization.Localization(__file__, 210, 16), getitem___1239, slice_1237)
    
    # Applying the binary operator 'in' (line 210)
    result_contains_1241 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 10), 'in', int_1234, subscript_call_result_1240)
    
    # Applying the 'not' unary operator (line 210)
    result_not__1242 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 6), 'not', result_contains_1241)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_1243 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 6), 'and', result_and_keyword_1233, result_not__1242)
    
    
    # Call to nonpawnWhiteAttacks(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'board' (line 211)
    board_1245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'board', False)
    int_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 37), 'int')
    # Processing the call keyword arguments (line 211)
    kwargs_1247 = {}
    # Getting the type of 'nonpawnWhiteAttacks' (line 211)
    nonpawnWhiteAttacks_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 10), 'nonpawnWhiteAttacks', False)
    # Calling nonpawnWhiteAttacks(args, kwargs) (line 211)
    nonpawnWhiteAttacks_call_result_1248 = invoke(stypy.reporting.localization.Localization(__file__, 211, 10), nonpawnWhiteAttacks_1244, *[board_1245, int_1246], **kwargs_1247)
    
    # Applying the 'not' unary operator (line 211)
    result_not__1249 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 6), 'not', nonpawnWhiteAttacks_call_result_1248)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_1250 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 6), 'and', result_and_keyword_1243, result_not__1249)
    
    
    # Call to nonpawnWhiteAttacks(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'board' (line 211)
    board_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 71), 'board', False)
    int_1253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 78), 'int')
    # Processing the call keyword arguments (line 211)
    kwargs_1254 = {}
    # Getting the type of 'nonpawnWhiteAttacks' (line 211)
    nonpawnWhiteAttacks_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 51), 'nonpawnWhiteAttacks', False)
    # Calling nonpawnWhiteAttacks(args, kwargs) (line 211)
    nonpawnWhiteAttacks_call_result_1255 = invoke(stypy.reporting.localization.Localization(__file__, 211, 51), nonpawnWhiteAttacks_1251, *[board_1252, int_1253], **kwargs_1254)
    
    # Applying the 'not' unary operator (line 211)
    result_not__1256 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 47), 'not', nonpawnWhiteAttacks_call_result_1255)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_1257 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 6), 'and', result_and_keyword_1250, result_not__1256)
    
    
    # Call to nonpawnWhiteAttacks(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'board' (line 211)
    board_1259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 112), 'board', False)
    int_1260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 119), 'int')
    # Processing the call keyword arguments (line 211)
    kwargs_1261 = {}
    # Getting the type of 'nonpawnWhiteAttacks' (line 211)
    nonpawnWhiteAttacks_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 92), 'nonpawnWhiteAttacks', False)
    # Calling nonpawnWhiteAttacks(args, kwargs) (line 211)
    nonpawnWhiteAttacks_call_result_1262 = invoke(stypy.reporting.localization.Localization(__file__, 211, 92), nonpawnWhiteAttacks_1258, *[board_1259, int_1260], **kwargs_1261)
    
    # Applying the 'not' unary operator (line 211)
    result_not__1263 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 88), 'not', nonpawnWhiteAttacks_call_result_1262)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_1264 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 6), 'and', result_and_keyword_1257, result_not__1263)
    
    # Testing the type of an if condition (line 209)
    if_condition_1265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 2), result_and_keyword_1264)
    # Assigning a type to the variable 'if_condition_1265' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 2), 'if_condition_1265', if_condition_1265)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 212)
    # Processing the call arguments (line 212)
    int_1268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'int')
    int_1269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 31), 'int')
    int_1270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 38), 'int')
    # Applying the binary operator '*' (line 212)
    result_mul_1271 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 31), '*', int_1269, int_1270)
    
    # Applying the binary operator '+' (line 212)
    result_add_1272 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 18), '+', int_1268, result_mul_1271)
    
    int_1273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 46), 'int')
    # Applying the binary operator '+' (line 212)
    result_add_1274 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 44), '+', result_add_1272, int_1273)
    
    # Processing the call keyword arguments (line 212)
    kwargs_1275 = {}
    # Getting the type of 'retval' (line 212)
    retval_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'retval', False)
    # Obtaining the member 'append' of a type (line 212)
    append_1267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 4), retval_1266, 'append')
    # Calling append(args, kwargs) (line 212)
    append_call_result_1276 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), append_1267, *[result_add_1274], **kwargs_1275)
    
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'retval' (line 213)
    retval_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 9), 'retval')
    # Assigning a type to the variable 'stypy_return_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 2), 'stypy_return_type', retval_1277)
    
    # ################# End of 'pseudoLegalMovesBlack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pseudoLegalMovesBlack' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pseudoLegalMovesBlack'
    return stypy_return_type_1278

# Assigning a type to the variable 'pseudoLegalMovesBlack' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'pseudoLegalMovesBlack', pseudoLegalMovesBlack)

@norecursion
def pseudoLegalMoves(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pseudoLegalMoves'
    module_type_store = module_type_store.open_function_context('pseudoLegalMoves', 215, 0, False)
    
    # Passed parameters checking function
    pseudoLegalMoves.stypy_localization = localization
    pseudoLegalMoves.stypy_type_of_self = None
    pseudoLegalMoves.stypy_type_store = module_type_store
    pseudoLegalMoves.stypy_function_name = 'pseudoLegalMoves'
    pseudoLegalMoves.stypy_param_names_list = ['board']
    pseudoLegalMoves.stypy_varargs_param_name = None
    pseudoLegalMoves.stypy_kwargs_param_name = None
    pseudoLegalMoves.stypy_call_defaults = defaults
    pseudoLegalMoves.stypy_call_varargs = varargs
    pseudoLegalMoves.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pseudoLegalMoves', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pseudoLegalMoves', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pseudoLegalMoves(...)' code ##################

    
    
    # Obtaining the type of the subscript
    int_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 11), 'int')
    # Getting the type of 'board' (line 216)
    board_1280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 5), 'board')
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___1281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 5), board_1280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_1282 = invoke(stypy.reporting.localization.Localization(__file__, 216, 5), getitem___1281, int_1279)
    
    # Testing the type of an if condition (line 216)
    if_condition_1283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 2), subscript_call_result_1282)
    # Assigning a type to the variable 'if_condition_1283' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 2), 'if_condition_1283', if_condition_1283)
    # SSA begins for if statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to pseudoLegalMovesWhite(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'board' (line 217)
    board_1285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 33), 'board', False)
    # Processing the call keyword arguments (line 217)
    kwargs_1286 = {}
    # Getting the type of 'pseudoLegalMovesWhite' (line 217)
    pseudoLegalMovesWhite_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'pseudoLegalMovesWhite', False)
    # Calling pseudoLegalMovesWhite(args, kwargs) (line 217)
    pseudoLegalMovesWhite_call_result_1287 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), pseudoLegalMovesWhite_1284, *[board_1285], **kwargs_1286)
    
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', pseudoLegalMovesWhite_call_result_1287)
    # SSA branch for the else part of an if statement (line 216)
    module_type_store.open_ssa_branch('else')
    
    # Call to pseudoLegalMovesBlack(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'board' (line 219)
    board_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'board', False)
    # Processing the call keyword arguments (line 219)
    kwargs_1290 = {}
    # Getting the type of 'pseudoLegalMovesBlack' (line 219)
    pseudoLegalMovesBlack_1288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'pseudoLegalMovesBlack', False)
    # Calling pseudoLegalMovesBlack(args, kwargs) (line 219)
    pseudoLegalMovesBlack_call_result_1291 = invoke(stypy.reporting.localization.Localization(__file__, 219, 11), pseudoLegalMovesBlack_1288, *[board_1289], **kwargs_1290)
    
    # Assigning a type to the variable 'stypy_return_type' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type', pseudoLegalMovesBlack_call_result_1291)
    # SSA join for if statement (line 216)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pseudoLegalMoves(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pseudoLegalMoves' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_1292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1292)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pseudoLegalMoves'
    return stypy_return_type_1292

# Assigning a type to the variable 'pseudoLegalMoves' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'pseudoLegalMoves', pseudoLegalMoves)

@norecursion
def pseudoLegalCapturesWhite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pseudoLegalCapturesWhite'
    module_type_store = module_type_store.open_function_context('pseudoLegalCapturesWhite', 221, 0, False)
    
    # Passed parameters checking function
    pseudoLegalCapturesWhite.stypy_localization = localization
    pseudoLegalCapturesWhite.stypy_type_of_self = None
    pseudoLegalCapturesWhite.stypy_type_store = module_type_store
    pseudoLegalCapturesWhite.stypy_function_name = 'pseudoLegalCapturesWhite'
    pseudoLegalCapturesWhite.stypy_param_names_list = ['board']
    pseudoLegalCapturesWhite.stypy_varargs_param_name = None
    pseudoLegalCapturesWhite.stypy_kwargs_param_name = None
    pseudoLegalCapturesWhite.stypy_call_defaults = defaults
    pseudoLegalCapturesWhite.stypy_call_varargs = varargs
    pseudoLegalCapturesWhite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pseudoLegalCapturesWhite', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pseudoLegalCapturesWhite', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pseudoLegalCapturesWhite(...)' code ##################

    
    # Assigning a List to a Name (line 222):
    
    # Obtaining an instance of the builtin type 'list' (line 222)
    list_1293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 222)
    
    # Assigning a type to the variable 'retval' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 2), 'retval', list_1293)
    
    # Getting the type of 'squares' (line 223)
    squares_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'squares')
    # Testing the type of a for loop iterable (line 223)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 223, 2), squares_1294)
    # Getting the type of the for loop variable (line 223)
    for_loop_var_1295 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 223, 2), squares_1294)
    # Assigning a type to the variable 'sq' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 2), 'sq', for_loop_var_1295)
    # SSA begins for a for statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 224)
    sq_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'sq')
    # Getting the type of 'board' (line 224)
    board_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___1298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), board_1297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_1299 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___1298, sq_1296)
    
    # Assigning a type to the variable 'b' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'b', subscript_call_result_1299)
    
    
    # Getting the type of 'b' (line 225)
    b_1300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 7), 'b')
    int_1301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'int')
    # Applying the binary operator '>=' (line 225)
    result_ge_1302 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 7), '>=', b_1300, int_1301)
    
    # Testing the type of an if condition (line 225)
    if_condition_1303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 4), result_ge_1302)
    # Assigning a type to the variable 'if_condition_1303' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'if_condition_1303', if_condition_1303)
    # SSA begins for if statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'b' (line 226)
    b_1304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 9), 'b')
    int_1305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 14), 'int')
    # Applying the binary operator '==' (line 226)
    result_eq_1306 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 9), '==', b_1304, int_1305)
    
    # Testing the type of an if condition (line 226)
    if_condition_1307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 6), result_eq_1306)
    # Assigning a type to the variable 'if_condition_1307' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 6), 'if_condition_1307', if_condition_1307)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 227)
    sq_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'sq')
    int_1309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 21), 'int')
    # Applying the binary operator '+' (line 227)
    result_add_1310 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 16), '+', sq_1308, int_1309)
    
    int_1311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 26), 'int')
    # Applying the binary operator '&' (line 227)
    result_and__1312 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 16), '&', result_add_1310, int_1311)
    
    # Applying the 'not' unary operator (line 227)
    result_not__1313 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), 'not', result_and__1312)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 227)
    sq_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 42), 'sq')
    int_1315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 47), 'int')
    # Applying the binary operator '+' (line 227)
    result_add_1316 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 42), '+', sq_1314, int_1315)
    
    # Getting the type of 'board' (line 227)
    board_1317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'board')
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___1318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 36), board_1317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_1319 = invoke(stypy.reporting.localization.Localization(__file__, 227, 36), getitem___1318, result_add_1316)
    
    int_1320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 53), 'int')
    # Applying the binary operator '<' (line 227)
    result_lt_1321 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 36), '<', subscript_call_result_1319, int_1320)
    
    # Applying the binary operator 'and' (line 227)
    result_and_keyword_1322 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), 'and', result_not__1313, result_lt_1321)
    
    # Testing the type of an if condition (line 227)
    if_condition_1323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_and_keyword_1322)
    # Assigning a type to the variable 'if_condition_1323' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_1323', if_condition_1323)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 228)
    # Processing the call arguments (line 228)
    int_1326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 24), 'int')
    # Getting the type of 'sq' (line 228)
    sq_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'sq', False)
    int_1328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 42), 'int')
    # Applying the binary operator '*' (line 228)
    result_mul_1329 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 37), '*', sq_1327, int_1328)
    
    # Applying the binary operator '+' (line 228)
    result_add_1330 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 24), '+', int_1326, result_mul_1329)
    
    int_1331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 50), 'int')
    # Applying the binary operator '+' (line 228)
    result_add_1332 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 48), '+', result_add_1330, int_1331)
    
    # Processing the call keyword arguments (line 228)
    kwargs_1333 = {}
    # Getting the type of 'retval' (line 228)
    retval_1324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 228)
    append_1325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 10), retval_1324, 'append')
    # Calling append(args, kwargs) (line 228)
    append_call_result_1334 = invoke(stypy.reporting.localization.Localization(__file__, 228, 10), append_1325, *[result_add_1332], **kwargs_1333)
    
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 229)
    sq_1335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'sq')
    int_1336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 21), 'int')
    # Applying the binary operator '+' (line 229)
    result_add_1337 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 16), '+', sq_1335, int_1336)
    
    int_1338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'int')
    # Applying the binary operator '&' (line 229)
    result_and__1339 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 16), '&', result_add_1337, int_1338)
    
    # Applying the 'not' unary operator (line 229)
    result_not__1340 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'not', result_and__1339)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 229)
    sq_1341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 42), 'sq')
    int_1342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 47), 'int')
    # Applying the binary operator '+' (line 229)
    result_add_1343 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 42), '+', sq_1341, int_1342)
    
    # Getting the type of 'board' (line 229)
    board_1344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 36), 'board')
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___1345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 36), board_1344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_1346 = invoke(stypy.reporting.localization.Localization(__file__, 229, 36), getitem___1345, result_add_1343)
    
    int_1347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 53), 'int')
    # Applying the binary operator '<' (line 229)
    result_lt_1348 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 36), '<', subscript_call_result_1346, int_1347)
    
    # Applying the binary operator 'and' (line 229)
    result_and_keyword_1349 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'and', result_not__1340, result_lt_1348)
    
    # Testing the type of an if condition (line 229)
    if_condition_1350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), result_and_keyword_1349)
    # Assigning a type to the variable 'if_condition_1350' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_1350', if_condition_1350)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 230)
    # Processing the call arguments (line 230)
    int_1353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 24), 'int')
    # Getting the type of 'sq' (line 230)
    sq_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 37), 'sq', False)
    int_1355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 42), 'int')
    # Applying the binary operator '*' (line 230)
    result_mul_1356 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 37), '*', sq_1354, int_1355)
    
    # Applying the binary operator '+' (line 230)
    result_add_1357 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 24), '+', int_1353, result_mul_1356)
    
    int_1358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 50), 'int')
    # Applying the binary operator '+' (line 230)
    result_add_1359 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 48), '+', result_add_1357, int_1358)
    
    # Processing the call keyword arguments (line 230)
    kwargs_1360 = {}
    # Getting the type of 'retval' (line 230)
    retval_1351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 230)
    append_1352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 10), retval_1351, 'append')
    # Calling append(args, kwargs) (line 230)
    append_call_result_1361 = invoke(stypy.reporting.localization.Localization(__file__, 230, 10), append_1352, *[result_add_1359], **kwargs_1360)
    
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 231)
    sq_1362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'sq')
    int_1363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 17), 'int')
    # Applying the binary operator '>=' (line 231)
    result_ge_1364 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), '>=', sq_1362, int_1363)
    
    
    # Getting the type of 'sq' (line 231)
    sq_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'sq')
    int_1366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 29), 'int')
    # Applying the binary operator '<' (line 231)
    result_lt_1367 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 24), '<', sq_1365, int_1366)
    
    # Applying the binary operator 'and' (line 231)
    result_and_keyword_1368 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), 'and', result_ge_1364, result_lt_1367)
    
    
    # Call to abs(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'sq' (line 231)
    sq_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 41), 'sq', False)
    int_1371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 46), 'int')
    # Applying the binary operator '&' (line 231)
    result_and__1372 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 41), '&', sq_1370, int_1371)
    
    
    # Obtaining the type of the subscript
    int_1373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 57), 'int')
    # Getting the type of 'board' (line 231)
    board_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 51), 'board', False)
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___1375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 51), board_1374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_1376 = invoke(stypy.reporting.localization.Localization(__file__, 231, 51), getitem___1375, int_1373)
    
    # Applying the binary operator '-' (line 231)
    result_sub_1377 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 40), '-', result_and__1372, subscript_call_result_1376)
    
    # Processing the call keyword arguments (line 231)
    kwargs_1378 = {}
    # Getting the type of 'abs' (line 231)
    abs_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 36), 'abs', False)
    # Calling abs(args, kwargs) (line 231)
    abs_call_result_1379 = invoke(stypy.reporting.localization.Localization(__file__, 231, 36), abs_1369, *[result_sub_1377], **kwargs_1378)
    
    int_1380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 65), 'int')
    # Applying the binary operator '==' (line 231)
    result_eq_1381 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 36), '==', abs_call_result_1379, int_1380)
    
    # Applying the binary operator 'and' (line 231)
    result_and_keyword_1382 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), 'and', result_and_keyword_1368, result_eq_1381)
    
    # Testing the type of an if condition (line 231)
    if_condition_1383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), result_and_keyword_1382)
    # Assigning a type to the variable 'if_condition_1383' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_1383', if_condition_1383)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 232)
    # Processing the call arguments (line 232)
    int_1386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 24), 'int')
    # Getting the type of 'sq' (line 232)
    sq_1387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 37), 'sq', False)
    int_1388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 42), 'int')
    # Applying the binary operator '*' (line 232)
    result_mul_1389 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 37), '*', sq_1387, int_1388)
    
    # Applying the binary operator '+' (line 232)
    result_add_1390 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 24), '+', int_1386, result_mul_1389)
    
    # Getting the type of 'sq' (line 232)
    sq_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 51), 'sq', False)
    int_1392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 56), 'int')
    # Applying the binary operator '&' (line 232)
    result_and__1393 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 51), '&', sq_1391, int_1392)
    
    # Applying the binary operator '+' (line 232)
    result_add_1394 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 48), '+', result_add_1390, result_and__1393)
    
    int_1395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 64), 'int')
    # Applying the binary operator '+' (line 232)
    result_add_1396 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 62), '+', result_add_1394, int_1395)
    
    
    # Obtaining the type of the subscript
    int_1397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 75), 'int')
    # Getting the type of 'board' (line 232)
    board_1398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 69), 'board', False)
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___1399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 69), board_1398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_1400 = invoke(stypy.reporting.localization.Localization(__file__, 232, 69), getitem___1399, int_1397)
    
    # Applying the binary operator '+' (line 232)
    result_add_1401 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 67), '+', result_add_1396, subscript_call_result_1400)
    
    # Processing the call keyword arguments (line 232)
    kwargs_1402 = {}
    # Getting the type of 'retval' (line 232)
    retval_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 232)
    append_1385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 10), retval_1384, 'append')
    # Calling append(args, kwargs) (line 232)
    append_call_result_1403 = invoke(stypy.reporting.localization.Localization(__file__, 232, 10), append_1385, *[result_add_1401], **kwargs_1402)
    
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 226)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 233)
    b_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'b')
    int_1405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'int')
    # Applying the binary operator '==' (line 233)
    result_eq_1406 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 11), '==', b_1404, int_1405)
    
    # Testing the type of an if condition (line 233)
    if_condition_1407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 11), result_eq_1406)
    # Assigning a type to the variable 'if_condition_1407' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'if_condition_1407', if_condition_1407)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'knightMoves' (line 234)
    knightMoves_1408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'knightMoves')
    # Testing the type of a for loop iterable (line 234)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 234, 8), knightMoves_1408)
    # Getting the type of the for loop variable (line 234)
    for_loop_var_1409 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 234, 8), knightMoves_1408)
    # Assigning a type to the variable 'k' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'k', for_loop_var_1409)
    # SSA begins for a for statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 235)
    sq_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 18), 'sq')
    # Getting the type of 'k' (line 235)
    k_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'k')
    # Applying the binary operator '+' (line 235)
    result_add_1412 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 18), '+', sq_1410, k_1411)
    
    int_1413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 27), 'int')
    # Applying the binary operator '&' (line 235)
    result_and__1414 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 18), '&', result_add_1412, int_1413)
    
    # Applying the 'not' unary operator (line 235)
    result_not__1415 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 13), 'not', result_and__1414)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 235)
    k_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 43), 'k')
    # Getting the type of 'sq' (line 235)
    sq_1417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 47), 'sq')
    # Applying the binary operator '+' (line 235)
    result_add_1418 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 43), '+', k_1416, sq_1417)
    
    # Getting the type of 'board' (line 235)
    board_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 37), 'board')
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___1420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 37), board_1419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_1421 = invoke(stypy.reporting.localization.Localization(__file__, 235, 37), getitem___1420, result_add_1418)
    
    int_1422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 53), 'int')
    # Applying the binary operator '<' (line 235)
    result_lt_1423 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 37), '<', subscript_call_result_1421, int_1422)
    
    # Applying the binary operator 'and' (line 235)
    result_and_keyword_1424 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 13), 'and', result_not__1415, result_lt_1423)
    
    # Testing the type of an if condition (line 235)
    if_condition_1425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 10), result_and_keyword_1424)
    # Assigning a type to the variable 'if_condition_1425' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 10), 'if_condition_1425', if_condition_1425)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 236)
    # Processing the call arguments (line 236)
    int_1428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'int')
    # Getting the type of 'sq' (line 236)
    sq_1429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 39), 'sq', False)
    int_1430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 44), 'int')
    # Applying the binary operator '*' (line 236)
    result_mul_1431 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 39), '*', sq_1429, int_1430)
    
    # Applying the binary operator '+' (line 236)
    result_add_1432 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 26), '+', int_1428, result_mul_1431)
    
    # Getting the type of 'k' (line 236)
    k_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 52), 'k', False)
    # Applying the binary operator '+' (line 236)
    result_add_1434 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 50), '+', result_add_1432, k_1433)
    
    # Processing the call keyword arguments (line 236)
    kwargs_1435 = {}
    # Getting the type of 'retval' (line 236)
    retval_1426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 236)
    append_1427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), retval_1426, 'append')
    # Calling append(args, kwargs) (line 236)
    append_call_result_1436 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), append_1427, *[result_add_1434], **kwargs_1435)
    
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 233)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 237)
    b_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'b')
    int_1438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 16), 'int')
    # Applying the binary operator '==' (line 237)
    result_eq_1439 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 11), '==', b_1437, int_1438)
    
    # Testing the type of an if condition (line 237)
    if_condition_1440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 11), result_eq_1439)
    # Assigning a type to the variable 'if_condition_1440' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'if_condition_1440', if_condition_1440)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'kingMoves' (line 238)
    kingMoves_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'kingMoves')
    # Testing the type of a for loop iterable (line 238)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 238, 8), kingMoves_1441)
    # Getting the type of the for loop variable (line 238)
    for_loop_var_1442 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 238, 8), kingMoves_1441)
    # Assigning a type to the variable 'k' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'k', for_loop_var_1442)
    # SSA begins for a for statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 239)
    k_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 17), 'k')
    # Getting the type of 'sq' (line 239)
    sq_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'sq')
    # Applying the binary operator '+' (line 239)
    result_add_1445 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 17), '+', k_1443, sq_1444)
    
    int_1446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 26), 'int')
    # Applying the binary operator '&' (line 239)
    result_and__1447 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 17), '&', result_add_1445, int_1446)
    
    # Applying the 'not' unary operator (line 239)
    result_not__1448 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 13), 'not', result_and__1447)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 239)
    k_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 42), 'k')
    # Getting the type of 'sq' (line 239)
    sq_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 46), 'sq')
    # Applying the binary operator '+' (line 239)
    result_add_1451 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 42), '+', k_1449, sq_1450)
    
    # Getting the type of 'board' (line 239)
    board_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'board')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___1453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 36), board_1452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_1454 = invoke(stypy.reporting.localization.Localization(__file__, 239, 36), getitem___1453, result_add_1451)
    
    int_1455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 52), 'int')
    # Applying the binary operator '<' (line 239)
    result_lt_1456 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 36), '<', subscript_call_result_1454, int_1455)
    
    # Applying the binary operator 'and' (line 239)
    result_and_keyword_1457 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 13), 'and', result_not__1448, result_lt_1456)
    
    # Testing the type of an if condition (line 239)
    if_condition_1458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 10), result_and_keyword_1457)
    # Assigning a type to the variable 'if_condition_1458' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 10), 'if_condition_1458', if_condition_1458)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 240)
    # Processing the call arguments (line 240)
    int_1461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 26), 'int')
    # Getting the type of 'sq' (line 240)
    sq_1462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 39), 'sq', False)
    int_1463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 44), 'int')
    # Applying the binary operator '*' (line 240)
    result_mul_1464 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 39), '*', sq_1462, int_1463)
    
    # Applying the binary operator '+' (line 240)
    result_add_1465 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 26), '+', int_1461, result_mul_1464)
    
    # Getting the type of 'k' (line 240)
    k_1466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 52), 'k', False)
    # Applying the binary operator '+' (line 240)
    result_add_1467 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 50), '+', result_add_1465, k_1466)
    
    # Processing the call keyword arguments (line 240)
    kwargs_1468 = {}
    # Getting the type of 'retval' (line 240)
    retval_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 240)
    append_1460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), retval_1459, 'append')
    # Calling append(args, kwargs) (line 240)
    append_call_result_1469 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), append_1460, *[result_add_1467], **kwargs_1468)
    
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 237)
    module_type_store.open_ssa_branch('else')
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'b' (line 242)
    b_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 31), 'b')
    # Getting the type of 'linePieces' (line 242)
    linePieces_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'linePieces')
    # Obtaining the member '__getitem__' of a type (line 242)
    getitem___1472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), linePieces_1471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 242)
    subscript_call_result_1473 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), getitem___1472, b_1470)
    
    # Testing the type of a for loop iterable (line 242)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 242, 8), subscript_call_result_1473)
    # Getting the type of the for loop variable (line 242)
    for_loop_var_1474 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 242, 8), subscript_call_result_1473)
    # Assigning a type to the variable 'line' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'line', for_loop_var_1474)
    # SSA begins for a for statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 243)
    line_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'line')
    # Testing the type of a for loop iterable (line 243)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 243, 10), line_1475)
    # Getting the type of the for loop variable (line 243)
    for_loop_var_1476 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 243, 10), line_1475)
    # Assigning a type to the variable 'k' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 10), 'k', for_loop_var_1476)
    # SSA begins for a for statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'sq' (line 244)
    sq_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'sq')
    # Getting the type of 'k' (line 244)
    k_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 21), 'k')
    # Applying the binary operator '+' (line 244)
    result_add_1479 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 16), '+', sq_1477, k_1478)
    
    int_1480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 25), 'int')
    # Applying the binary operator '&' (line 244)
    result_and__1481 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 16), '&', result_add_1479, int_1480)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 244)
    k_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 40), 'k')
    # Getting the type of 'sq' (line 244)
    sq_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 44), 'sq')
    # Applying the binary operator '+' (line 244)
    result_add_1484 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 40), '+', k_1482, sq_1483)
    
    # Getting the type of 'board' (line 244)
    board_1485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 34), 'board')
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___1486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 34), board_1485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_1487 = invoke(stypy.reporting.localization.Localization(__file__, 244, 34), getitem___1486, result_add_1484)
    
    int_1488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 51), 'int')
    # Applying the binary operator '>=' (line 244)
    result_ge_1489 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 34), '>=', subscript_call_result_1487, int_1488)
    
    # Applying the binary operator 'or' (line 244)
    result_or_keyword_1490 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 15), 'or', result_and__1481, result_ge_1489)
    
    # Testing the type of an if condition (line 244)
    if_condition_1491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 12), result_or_keyword_1490)
    # Assigning a type to the variable 'if_condition_1491' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'if_condition_1491', if_condition_1491)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 246)
    k_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'k')
    # Getting the type of 'sq' (line 246)
    sq_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'sq')
    # Applying the binary operator '+' (line 246)
    result_add_1494 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 21), '+', k_1492, sq_1493)
    
    # Getting the type of 'board' (line 246)
    board_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___1496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), board_1495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_1497 = invoke(stypy.reporting.localization.Localization(__file__, 246, 15), getitem___1496, result_add_1494)
    
    int_1498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 31), 'int')
    # Applying the binary operator '<' (line 246)
    result_lt_1499 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 15), '<', subscript_call_result_1497, int_1498)
    
    # Testing the type of an if condition (line 246)
    if_condition_1500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 12), result_lt_1499)
    # Assigning a type to the variable 'if_condition_1500' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'if_condition_1500', if_condition_1500)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 247)
    # Processing the call arguments (line 247)
    int_1503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 28), 'int')
    # Getting the type of 'sq' (line 247)
    sq_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 41), 'sq', False)
    int_1505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 46), 'int')
    # Applying the binary operator '*' (line 247)
    result_mul_1506 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 41), '*', sq_1504, int_1505)
    
    # Applying the binary operator '+' (line 247)
    result_add_1507 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 28), '+', int_1503, result_mul_1506)
    
    # Getting the type of 'k' (line 247)
    k_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 54), 'k', False)
    # Applying the binary operator '+' (line 247)
    result_add_1509 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 52), '+', result_add_1507, k_1508)
    
    # Processing the call keyword arguments (line 247)
    kwargs_1510 = {}
    # Getting the type of 'retval' (line 247)
    retval_1501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'retval', False)
    # Obtaining the member 'append' of a type (line 247)
    append_1502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 14), retval_1501, 'append')
    # Calling append(args, kwargs) (line 247)
    append_call_result_1511 = invoke(stypy.reporting.localization.Localization(__file__, 247, 14), append_1502, *[result_add_1509], **kwargs_1510)
    
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'retval' (line 249)
    retval_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 9), 'retval')
    # Assigning a type to the variable 'stypy_return_type' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 2), 'stypy_return_type', retval_1512)
    
    # ################# End of 'pseudoLegalCapturesWhite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pseudoLegalCapturesWhite' in the type store
    # Getting the type of 'stypy_return_type' (line 221)
    stypy_return_type_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1513)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pseudoLegalCapturesWhite'
    return stypy_return_type_1513

# Assigning a type to the variable 'pseudoLegalCapturesWhite' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'pseudoLegalCapturesWhite', pseudoLegalCapturesWhite)

@norecursion
def pseudoLegalCapturesBlack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pseudoLegalCapturesBlack'
    module_type_store = module_type_store.open_function_context('pseudoLegalCapturesBlack', 251, 0, False)
    
    # Passed parameters checking function
    pseudoLegalCapturesBlack.stypy_localization = localization
    pseudoLegalCapturesBlack.stypy_type_of_self = None
    pseudoLegalCapturesBlack.stypy_type_store = module_type_store
    pseudoLegalCapturesBlack.stypy_function_name = 'pseudoLegalCapturesBlack'
    pseudoLegalCapturesBlack.stypy_param_names_list = ['board']
    pseudoLegalCapturesBlack.stypy_varargs_param_name = None
    pseudoLegalCapturesBlack.stypy_kwargs_param_name = None
    pseudoLegalCapturesBlack.stypy_call_defaults = defaults
    pseudoLegalCapturesBlack.stypy_call_varargs = varargs
    pseudoLegalCapturesBlack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pseudoLegalCapturesBlack', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pseudoLegalCapturesBlack', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pseudoLegalCapturesBlack(...)' code ##################

    
    # Assigning a List to a Name (line 252):
    
    # Obtaining an instance of the builtin type 'list' (line 252)
    list_1514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 252)
    
    # Assigning a type to the variable 'retval' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 2), 'retval', list_1514)
    
    # Getting the type of 'squares' (line 253)
    squares_1515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'squares')
    # Testing the type of a for loop iterable (line 253)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 253, 2), squares_1515)
    # Getting the type of the for loop variable (line 253)
    for_loop_var_1516 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 253, 2), squares_1515)
    # Assigning a type to the variable 'sq' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 2), 'sq', for_loop_var_1516)
    # SSA begins for a for statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 254)
    sq_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 14), 'sq')
    # Getting the type of 'board' (line 254)
    board_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___1519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), board_1518, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_1520 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), getitem___1519, sq_1517)
    
    # Assigning a type to the variable 'b' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'b', subscript_call_result_1520)
    
    
    # Getting the type of 'b' (line 255)
    b_1521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 7), 'b')
    int_1522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 11), 'int')
    # Applying the binary operator '<' (line 255)
    result_lt_1523 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 7), '<', b_1521, int_1522)
    
    # Testing the type of an if condition (line 255)
    if_condition_1524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 4), result_lt_1523)
    # Assigning a type to the variable 'if_condition_1524' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'if_condition_1524', if_condition_1524)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'b' (line 256)
    b_1525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 9), 'b')
    int_1526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 14), 'int')
    # Applying the binary operator '==' (line 256)
    result_eq_1527 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 9), '==', b_1525, int_1526)
    
    # Testing the type of an if condition (line 256)
    if_condition_1528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 6), result_eq_1527)
    # Assigning a type to the variable 'if_condition_1528' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 6), 'if_condition_1528', if_condition_1528)
    # SSA begins for if statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 257)
    sq_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 17), 'sq')
    int_1530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'int')
    # Applying the binary operator '-' (line 257)
    result_sub_1531 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 17), '-', sq_1529, int_1530)
    
    # Getting the type of 'board' (line 257)
    board_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'board')
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___1533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), board_1532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_1534 = invoke(stypy.reporting.localization.Localization(__file__, 257, 11), getitem___1533, result_sub_1531)
    
    int_1535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 29), 'int')
    # Applying the binary operator '>=' (line 257)
    result_ge_1536 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), '>=', subscript_call_result_1534, int_1535)
    
    # Testing the type of an if condition (line 257)
    if_condition_1537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_ge_1536)
    # Assigning a type to the variable 'if_condition_1537' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_1537', if_condition_1537)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 258)
    # Processing the call arguments (line 258)
    int_1540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 24), 'int')
    # Getting the type of 'sq' (line 258)
    sq_1541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 37), 'sq', False)
    int_1542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 42), 'int')
    # Applying the binary operator '*' (line 258)
    result_mul_1543 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 37), '*', sq_1541, int_1542)
    
    # Applying the binary operator '+' (line 258)
    result_add_1544 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 24), '+', int_1540, result_mul_1543)
    
    int_1545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 50), 'int')
    # Applying the binary operator '-' (line 258)
    result_sub_1546 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 48), '-', result_add_1544, int_1545)
    
    # Processing the call keyword arguments (line 258)
    kwargs_1547 = {}
    # Getting the type of 'retval' (line 258)
    retval_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 258)
    append_1539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 10), retval_1538, 'append')
    # Calling append(args, kwargs) (line 258)
    append_call_result_1548 = invoke(stypy.reporting.localization.Localization(__file__, 258, 10), append_1539, *[result_sub_1546], **kwargs_1547)
    
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'sq' (line 259)
    sq_1549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 17), 'sq')
    int_1550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 22), 'int')
    # Applying the binary operator '-' (line 259)
    result_sub_1551 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 17), '-', sq_1549, int_1550)
    
    # Getting the type of 'board' (line 259)
    board_1552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'board')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___1553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), board_1552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_1554 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), getitem___1553, result_sub_1551)
    
    int_1555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 29), 'int')
    # Applying the binary operator '>=' (line 259)
    result_ge_1556 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 11), '>=', subscript_call_result_1554, int_1555)
    
    # Testing the type of an if condition (line 259)
    if_condition_1557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 8), result_ge_1556)
    # Assigning a type to the variable 'if_condition_1557' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'if_condition_1557', if_condition_1557)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 260)
    # Processing the call arguments (line 260)
    int_1560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'int')
    # Getting the type of 'sq' (line 260)
    sq_1561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'sq', False)
    int_1562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 42), 'int')
    # Applying the binary operator '*' (line 260)
    result_mul_1563 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 37), '*', sq_1561, int_1562)
    
    # Applying the binary operator '+' (line 260)
    result_add_1564 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 24), '+', int_1560, result_mul_1563)
    
    int_1565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 50), 'int')
    # Applying the binary operator '-' (line 260)
    result_sub_1566 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 48), '-', result_add_1564, int_1565)
    
    # Processing the call keyword arguments (line 260)
    kwargs_1567 = {}
    # Getting the type of 'retval' (line 260)
    retval_1558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 260)
    append_1559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 10), retval_1558, 'append')
    # Calling append(args, kwargs) (line 260)
    append_call_result_1568 = invoke(stypy.reporting.localization.Localization(__file__, 260, 10), append_1559, *[result_sub_1566], **kwargs_1567)
    
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 261)
    sq_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'sq')
    int_1570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 17), 'int')
    # Applying the binary operator '>=' (line 261)
    result_ge_1571 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), '>=', sq_1569, int_1570)
    
    
    # Getting the type of 'sq' (line 261)
    sq_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'sq')
    int_1573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 29), 'int')
    # Applying the binary operator '<' (line 261)
    result_lt_1574 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 24), '<', sq_1572, int_1573)
    
    # Applying the binary operator 'and' (line 261)
    result_and_keyword_1575 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'and', result_ge_1571, result_lt_1574)
    
    
    # Call to abs(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'sq' (line 261)
    sq_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 41), 'sq', False)
    int_1578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 46), 'int')
    # Applying the binary operator '&' (line 261)
    result_and__1579 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 41), '&', sq_1577, int_1578)
    
    
    # Obtaining the type of the subscript
    int_1580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 57), 'int')
    # Getting the type of 'board' (line 261)
    board_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 51), 'board', False)
    # Obtaining the member '__getitem__' of a type (line 261)
    getitem___1582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 51), board_1581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 261)
    subscript_call_result_1583 = invoke(stypy.reporting.localization.Localization(__file__, 261, 51), getitem___1582, int_1580)
    
    # Applying the binary operator '-' (line 261)
    result_sub_1584 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 40), '-', result_and__1579, subscript_call_result_1583)
    
    # Processing the call keyword arguments (line 261)
    kwargs_1585 = {}
    # Getting the type of 'abs' (line 261)
    abs_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 36), 'abs', False)
    # Calling abs(args, kwargs) (line 261)
    abs_call_result_1586 = invoke(stypy.reporting.localization.Localization(__file__, 261, 36), abs_1576, *[result_sub_1584], **kwargs_1585)
    
    int_1587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 65), 'int')
    # Applying the binary operator '==' (line 261)
    result_eq_1588 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 36), '==', abs_call_result_1586, int_1587)
    
    # Applying the binary operator 'and' (line 261)
    result_and_keyword_1589 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'and', result_and_keyword_1575, result_eq_1588)
    
    # Testing the type of an if condition (line 261)
    if_condition_1590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_and_keyword_1589)
    # Assigning a type to the variable 'if_condition_1590' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_1590', if_condition_1590)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 262)
    # Processing the call arguments (line 262)
    int_1593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'int')
    # Getting the type of 'sq' (line 262)
    sq_1594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 37), 'sq', False)
    int_1595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 42), 'int')
    # Applying the binary operator '*' (line 262)
    result_mul_1596 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 37), '*', sq_1594, int_1595)
    
    # Applying the binary operator '+' (line 262)
    result_add_1597 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 24), '+', int_1593, result_mul_1596)
    
    # Getting the type of 'sq' (line 262)
    sq_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 51), 'sq', False)
    int_1599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 56), 'int')
    # Applying the binary operator '&' (line 262)
    result_and__1600 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 51), '&', sq_1598, int_1599)
    
    # Applying the binary operator '+' (line 262)
    result_add_1601 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 48), '+', result_add_1597, result_and__1600)
    
    int_1602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 64), 'int')
    # Applying the binary operator '-' (line 262)
    result_sub_1603 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 62), '-', result_add_1601, int_1602)
    
    
    # Obtaining the type of the subscript
    int_1604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 75), 'int')
    # Getting the type of 'board' (line 262)
    board_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 69), 'board', False)
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___1606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 69), board_1605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_1607 = invoke(stypy.reporting.localization.Localization(__file__, 262, 69), getitem___1606, int_1604)
    
    # Applying the binary operator '+' (line 262)
    result_add_1608 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 67), '+', result_sub_1603, subscript_call_result_1607)
    
    # Processing the call keyword arguments (line 262)
    kwargs_1609 = {}
    # Getting the type of 'retval' (line 262)
    retval_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 10), 'retval', False)
    # Obtaining the member 'append' of a type (line 262)
    append_1592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 10), retval_1591, 'append')
    # Calling append(args, kwargs) (line 262)
    append_call_result_1610 = invoke(stypy.reporting.localization.Localization(__file__, 262, 10), append_1592, *[result_add_1608], **kwargs_1609)
    
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 256)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 263)
    b_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'b')
    int_1612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'int')
    # Applying the binary operator '==' (line 263)
    result_eq_1613 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 11), '==', b_1611, int_1612)
    
    # Testing the type of an if condition (line 263)
    if_condition_1614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 11), result_eq_1613)
    # Assigning a type to the variable 'if_condition_1614' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'if_condition_1614', if_condition_1614)
    # SSA begins for if statement (line 263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'knightMoves' (line 264)
    knightMoves_1615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'knightMoves')
    # Testing the type of a for loop iterable (line 264)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 264, 8), knightMoves_1615)
    # Getting the type of the for loop variable (line 264)
    for_loop_var_1616 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 264, 8), knightMoves_1615)
    # Assigning a type to the variable 'k' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'k', for_loop_var_1616)
    # SSA begins for a for statement (line 264)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sq' (line 265)
    sq_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'sq')
    # Getting the type of 'k' (line 265)
    k_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'k')
    # Applying the binary operator '+' (line 265)
    result_add_1619 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 18), '+', sq_1617, k_1618)
    
    int_1620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'int')
    # Applying the binary operator '&' (line 265)
    result_and__1621 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 18), '&', result_add_1619, int_1620)
    
    # Applying the 'not' unary operator (line 265)
    result_not__1622 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 13), 'not', result_and__1621)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 265)
    k_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 43), 'k')
    # Getting the type of 'sq' (line 265)
    sq_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 47), 'sq')
    # Applying the binary operator '+' (line 265)
    result_add_1625 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 43), '+', k_1623, sq_1624)
    
    # Getting the type of 'board' (line 265)
    board_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 37), 'board')
    # Obtaining the member '__getitem__' of a type (line 265)
    getitem___1627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 37), board_1626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 265)
    subscript_call_result_1628 = invoke(stypy.reporting.localization.Localization(__file__, 265, 37), getitem___1627, result_add_1625)
    
    int_1629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 54), 'int')
    # Applying the binary operator '>=' (line 265)
    result_ge_1630 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 37), '>=', subscript_call_result_1628, int_1629)
    
    # Applying the binary operator 'and' (line 265)
    result_and_keyword_1631 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 13), 'and', result_not__1622, result_ge_1630)
    
    # Testing the type of an if condition (line 265)
    if_condition_1632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 10), result_and_keyword_1631)
    # Assigning a type to the variable 'if_condition_1632' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 10), 'if_condition_1632', if_condition_1632)
    # SSA begins for if statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 266)
    # Processing the call arguments (line 266)
    int_1635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'int')
    # Getting the type of 'sq' (line 266)
    sq_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 39), 'sq', False)
    int_1637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 44), 'int')
    # Applying the binary operator '*' (line 266)
    result_mul_1638 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 39), '*', sq_1636, int_1637)
    
    # Applying the binary operator '+' (line 266)
    result_add_1639 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 26), '+', int_1635, result_mul_1638)
    
    # Getting the type of 'k' (line 266)
    k_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 52), 'k', False)
    # Applying the binary operator '+' (line 266)
    result_add_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 50), '+', result_add_1639, k_1640)
    
    # Processing the call keyword arguments (line 266)
    kwargs_1642 = {}
    # Getting the type of 'retval' (line 266)
    retval_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 266)
    append_1634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), retval_1633, 'append')
    # Calling append(args, kwargs) (line 266)
    append_call_result_1643 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), append_1634, *[result_add_1641], **kwargs_1642)
    
    # SSA join for if statement (line 265)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 263)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 267)
    b_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'b')
    int_1645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 16), 'int')
    # Applying the binary operator '==' (line 267)
    result_eq_1646 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 11), '==', b_1644, int_1645)
    
    # Testing the type of an if condition (line 267)
    if_condition_1647 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 11), result_eq_1646)
    # Assigning a type to the variable 'if_condition_1647' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'if_condition_1647', if_condition_1647)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'bishopLines' (line 268)
    bishopLines_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'bishopLines')
    # Testing the type of a for loop iterable (line 268)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 268, 8), bishopLines_1648)
    # Getting the type of the for loop variable (line 268)
    for_loop_var_1649 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 268, 8), bishopLines_1648)
    # Assigning a type to the variable 'line' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'line', for_loop_var_1649)
    # SSA begins for a for statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 269)
    line_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'line')
    # Testing the type of a for loop iterable (line 269)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 10), line_1650)
    # Getting the type of the for loop variable (line 269)
    for_loop_var_1651 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 10), line_1650)
    # Assigning a type to the variable 'k' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 10), 'k', for_loop_var_1651)
    # SSA begins for a for statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 270)
    k_1652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 21), 'k')
    # Getting the type of 'sq' (line 270)
    sq_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 25), 'sq')
    # Applying the binary operator '+' (line 270)
    result_add_1654 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 21), '+', k_1652, sq_1653)
    
    # Getting the type of 'board' (line 270)
    board_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___1656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 15), board_1655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_1657 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), getitem___1656, result_add_1654)
    
    int_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 31), 'int')
    # Applying the binary operator '<' (line 270)
    result_lt_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 15), '<', subscript_call_result_1657, int_1658)
    
    # Testing the type of an if condition (line 270)
    if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 12), result_lt_1659)
    # Assigning a type to the variable 'if_condition_1660' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'if_condition_1660', if_condition_1660)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 272)
    k_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 21), 'k')
    # Getting the type of 'sq' (line 272)
    sq_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 25), 'sq')
    # Applying the binary operator '+' (line 272)
    result_add_1663 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 21), '+', k_1661, sq_1662)
    
    # Getting the type of 'board' (line 272)
    board_1664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___1665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), board_1664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_1666 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), getitem___1665, result_add_1663)
    
    int_1667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 32), 'int')
    # Applying the binary operator '>=' (line 272)
    result_ge_1668 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), '>=', subscript_call_result_1666, int_1667)
    
    # Testing the type of an if condition (line 272)
    if_condition_1669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), result_ge_1668)
    # Assigning a type to the variable 'if_condition_1669' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_1669', if_condition_1669)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 273)
    # Processing the call arguments (line 273)
    int_1672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 28), 'int')
    # Getting the type of 'sq' (line 273)
    sq_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 41), 'sq', False)
    int_1674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 46), 'int')
    # Applying the binary operator '*' (line 273)
    result_mul_1675 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 41), '*', sq_1673, int_1674)
    
    # Applying the binary operator '+' (line 273)
    result_add_1676 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 28), '+', int_1672, result_mul_1675)
    
    # Getting the type of 'k' (line 273)
    k_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 54), 'k', False)
    # Applying the binary operator '+' (line 273)
    result_add_1678 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 52), '+', result_add_1676, k_1677)
    
    # Processing the call keyword arguments (line 273)
    kwargs_1679 = {}
    # Getting the type of 'retval' (line 273)
    retval_1670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 14), 'retval', False)
    # Obtaining the member 'append' of a type (line 273)
    append_1671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 14), retval_1670, 'append')
    # Calling append(args, kwargs) (line 273)
    append_call_result_1680 = invoke(stypy.reporting.localization.Localization(__file__, 273, 14), append_1671, *[result_add_1678], **kwargs_1679)
    
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 267)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 275)
    b_1681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'b')
    int_1682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 16), 'int')
    # Applying the binary operator '==' (line 275)
    result_eq_1683 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), '==', b_1681, int_1682)
    
    # Testing the type of an if condition (line 275)
    if_condition_1684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 11), result_eq_1683)
    # Assigning a type to the variable 'if_condition_1684' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'if_condition_1684', if_condition_1684)
    # SSA begins for if statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'rookLines' (line 276)
    rookLines_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'rookLines')
    # Testing the type of a for loop iterable (line 276)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 276, 8), rookLines_1685)
    # Getting the type of the for loop variable (line 276)
    for_loop_var_1686 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 276, 8), rookLines_1685)
    # Assigning a type to the variable 'line' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'line', for_loop_var_1686)
    # SSA begins for a for statement (line 276)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 277)
    line_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'line')
    # Testing the type of a for loop iterable (line 277)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 10), line_1687)
    # Getting the type of the for loop variable (line 277)
    for_loop_var_1688 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 10), line_1687)
    # Assigning a type to the variable 'k' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 10), 'k', for_loop_var_1688)
    # SSA begins for a for statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 278)
    k_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'k')
    # Getting the type of 'sq' (line 278)
    sq_1690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 25), 'sq')
    # Applying the binary operator '+' (line 278)
    result_add_1691 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 21), '+', k_1689, sq_1690)
    
    # Getting the type of 'board' (line 278)
    board_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___1693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 15), board_1692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_1694 = invoke(stypy.reporting.localization.Localization(__file__, 278, 15), getitem___1693, result_add_1691)
    
    int_1695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
    # Applying the binary operator '<' (line 278)
    result_lt_1696 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 15), '<', subscript_call_result_1694, int_1695)
    
    # Testing the type of an if condition (line 278)
    if_condition_1697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 12), result_lt_1696)
    # Assigning a type to the variable 'if_condition_1697' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'if_condition_1697', if_condition_1697)
    # SSA begins for if statement (line 278)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 278)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 280)
    k_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 21), 'k')
    # Getting the type of 'sq' (line 280)
    sq_1699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'sq')
    # Applying the binary operator '+' (line 280)
    result_add_1700 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 21), '+', k_1698, sq_1699)
    
    # Getting the type of 'board' (line 280)
    board_1701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___1702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 15), board_1701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_1703 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), getitem___1702, result_add_1700)
    
    int_1704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 32), 'int')
    # Applying the binary operator '>=' (line 280)
    result_ge_1705 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), '>=', subscript_call_result_1703, int_1704)
    
    # Testing the type of an if condition (line 280)
    if_condition_1706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), result_ge_1705)
    # Assigning a type to the variable 'if_condition_1706' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_1706', if_condition_1706)
    # SSA begins for if statement (line 280)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 281)
    # Processing the call arguments (line 281)
    int_1709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 28), 'int')
    # Getting the type of 'sq' (line 281)
    sq_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 41), 'sq', False)
    int_1711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 46), 'int')
    # Applying the binary operator '*' (line 281)
    result_mul_1712 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 41), '*', sq_1710, int_1711)
    
    # Applying the binary operator '+' (line 281)
    result_add_1713 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 28), '+', int_1709, result_mul_1712)
    
    # Getting the type of 'k' (line 281)
    k_1714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 54), 'k', False)
    # Applying the binary operator '+' (line 281)
    result_add_1715 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 52), '+', result_add_1713, k_1714)
    
    # Processing the call keyword arguments (line 281)
    kwargs_1716 = {}
    # Getting the type of 'retval' (line 281)
    retval_1707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 14), 'retval', False)
    # Obtaining the member 'append' of a type (line 281)
    append_1708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 14), retval_1707, 'append')
    # Calling append(args, kwargs) (line 281)
    append_call_result_1717 = invoke(stypy.reporting.localization.Localization(__file__, 281, 14), append_1708, *[result_add_1715], **kwargs_1716)
    
    # SSA join for if statement (line 280)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 275)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 283)
    b_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'b')
    int_1719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 16), 'int')
    # Applying the binary operator '==' (line 283)
    result_eq_1720 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 11), '==', b_1718, int_1719)
    
    # Testing the type of an if condition (line 283)
    if_condition_1721 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 11), result_eq_1720)
    # Assigning a type to the variable 'if_condition_1721' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'if_condition_1721', if_condition_1721)
    # SSA begins for if statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'queenLines' (line 284)
    queenLines_1722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'queenLines')
    # Testing the type of a for loop iterable (line 284)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 284, 8), queenLines_1722)
    # Getting the type of the for loop variable (line 284)
    for_loop_var_1723 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 284, 8), queenLines_1722)
    # Assigning a type to the variable 'line' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'line', for_loop_var_1723)
    # SSA begins for a for statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 285)
    line_1724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'line')
    # Testing the type of a for loop iterable (line 285)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 285, 10), line_1724)
    # Getting the type of the for loop variable (line 285)
    for_loop_var_1725 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 285, 10), line_1724)
    # Assigning a type to the variable 'k' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 10), 'k', for_loop_var_1725)
    # SSA begins for a for statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 286)
    k_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 21), 'k')
    # Getting the type of 'sq' (line 286)
    sq_1727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'sq')
    # Applying the binary operator '+' (line 286)
    result_add_1728 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 21), '+', k_1726, sq_1727)
    
    # Getting the type of 'board' (line 286)
    board_1729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___1730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), board_1729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_1731 = invoke(stypy.reporting.localization.Localization(__file__, 286, 15), getitem___1730, result_add_1728)
    
    int_1732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 31), 'int')
    # Applying the binary operator '<' (line 286)
    result_lt_1733 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 15), '<', subscript_call_result_1731, int_1732)
    
    # Testing the type of an if condition (line 286)
    if_condition_1734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 12), result_lt_1733)
    # Assigning a type to the variable 'if_condition_1734' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'if_condition_1734', if_condition_1734)
    # SSA begins for if statement (line 286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 286)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 288)
    k_1735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'k')
    # Getting the type of 'sq' (line 288)
    sq_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'sq')
    # Applying the binary operator '+' (line 288)
    result_add_1737 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 21), '+', k_1735, sq_1736)
    
    # Getting the type of 'board' (line 288)
    board_1738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'board')
    # Obtaining the member '__getitem__' of a type (line 288)
    getitem___1739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), board_1738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 288)
    subscript_call_result_1740 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), getitem___1739, result_add_1737)
    
    int_1741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 32), 'int')
    # Applying the binary operator '>=' (line 288)
    result_ge_1742 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 15), '>=', subscript_call_result_1740, int_1741)
    
    # Testing the type of an if condition (line 288)
    if_condition_1743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 12), result_ge_1742)
    # Assigning a type to the variable 'if_condition_1743' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'if_condition_1743', if_condition_1743)
    # SSA begins for if statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 289)
    # Processing the call arguments (line 289)
    int_1746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 28), 'int')
    # Getting the type of 'sq' (line 289)
    sq_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 41), 'sq', False)
    int_1748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 46), 'int')
    # Applying the binary operator '*' (line 289)
    result_mul_1749 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 41), '*', sq_1747, int_1748)
    
    # Applying the binary operator '+' (line 289)
    result_add_1750 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 28), '+', int_1746, result_mul_1749)
    
    # Getting the type of 'k' (line 289)
    k_1751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 54), 'k', False)
    # Applying the binary operator '+' (line 289)
    result_add_1752 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 52), '+', result_add_1750, k_1751)
    
    # Processing the call keyword arguments (line 289)
    kwargs_1753 = {}
    # Getting the type of 'retval' (line 289)
    retval_1744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 14), 'retval', False)
    # Obtaining the member 'append' of a type (line 289)
    append_1745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 14), retval_1744, 'append')
    # Calling append(args, kwargs) (line 289)
    append_call_result_1754 = invoke(stypy.reporting.localization.Localization(__file__, 289, 14), append_1745, *[result_add_1752], **kwargs_1753)
    
    # SSA join for if statement (line 288)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 283)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'b' (line 291)
    b_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 11), 'b')
    int_1756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 16), 'int')
    # Applying the binary operator '==' (line 291)
    result_eq_1757 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 11), '==', b_1755, int_1756)
    
    # Testing the type of an if condition (line 291)
    if_condition_1758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 11), result_eq_1757)
    # Assigning a type to the variable 'if_condition_1758' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 11), 'if_condition_1758', if_condition_1758)
    # SSA begins for if statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'kingMoves' (line 292)
    kingMoves_1759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 'kingMoves')
    # Testing the type of a for loop iterable (line 292)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 292, 8), kingMoves_1759)
    # Getting the type of the for loop variable (line 292)
    for_loop_var_1760 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 292, 8), kingMoves_1759)
    # Assigning a type to the variable 'k' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'k', for_loop_var_1760)
    # SSA begins for a for statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 293)
    k_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'k')
    # Getting the type of 'sq' (line 293)
    sq_1762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 23), 'sq')
    # Applying the binary operator '+' (line 293)
    result_add_1763 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 19), '+', k_1761, sq_1762)
    
    # Getting the type of 'board' (line 293)
    board_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'board')
    # Obtaining the member '__getitem__' of a type (line 293)
    getitem___1765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 13), board_1764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 293)
    subscript_call_result_1766 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), getitem___1765, result_add_1763)
    
    int_1767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 30), 'int')
    # Applying the binary operator '>=' (line 293)
    result_ge_1768 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 13), '>=', subscript_call_result_1766, int_1767)
    
    # Testing the type of an if condition (line 293)
    if_condition_1769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 10), result_ge_1768)
    # Assigning a type to the variable 'if_condition_1769' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 10), 'if_condition_1769', if_condition_1769)
    # SSA begins for if statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 294)
    # Processing the call arguments (line 294)
    int_1772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'int')
    # Getting the type of 'sq' (line 294)
    sq_1773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'sq', False)
    int_1774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 44), 'int')
    # Applying the binary operator '*' (line 294)
    result_mul_1775 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 39), '*', sq_1773, int_1774)
    
    # Applying the binary operator '+' (line 294)
    result_add_1776 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 26), '+', int_1772, result_mul_1775)
    
    # Getting the type of 'k' (line 294)
    k_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 52), 'k', False)
    # Applying the binary operator '+' (line 294)
    result_add_1778 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 50), '+', result_add_1776, k_1777)
    
    # Processing the call keyword arguments (line 294)
    kwargs_1779 = {}
    # Getting the type of 'retval' (line 294)
    retval_1770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'retval', False)
    # Obtaining the member 'append' of a type (line 294)
    append_1771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), retval_1770, 'append')
    # Calling append(args, kwargs) (line 294)
    append_call_result_1780 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), append_1771, *[result_add_1778], **kwargs_1779)
    
    # SSA join for if statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 291)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 283)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 275)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 263)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 256)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'retval' (line 295)
    retval_1781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 9), 'retval')
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 2), 'stypy_return_type', retval_1781)
    
    # ################# End of 'pseudoLegalCapturesBlack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pseudoLegalCapturesBlack' in the type store
    # Getting the type of 'stypy_return_type' (line 251)
    stypy_return_type_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1782)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pseudoLegalCapturesBlack'
    return stypy_return_type_1782

# Assigning a type to the variable 'pseudoLegalCapturesBlack' (line 251)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'pseudoLegalCapturesBlack', pseudoLegalCapturesBlack)

@norecursion
def pseudoLegalCaptures(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pseudoLegalCaptures'
    module_type_store = module_type_store.open_function_context('pseudoLegalCaptures', 297, 0, False)
    
    # Passed parameters checking function
    pseudoLegalCaptures.stypy_localization = localization
    pseudoLegalCaptures.stypy_type_of_self = None
    pseudoLegalCaptures.stypy_type_store = module_type_store
    pseudoLegalCaptures.stypy_function_name = 'pseudoLegalCaptures'
    pseudoLegalCaptures.stypy_param_names_list = ['board']
    pseudoLegalCaptures.stypy_varargs_param_name = None
    pseudoLegalCaptures.stypy_kwargs_param_name = None
    pseudoLegalCaptures.stypy_call_defaults = defaults
    pseudoLegalCaptures.stypy_call_varargs = varargs
    pseudoLegalCaptures.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pseudoLegalCaptures', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pseudoLegalCaptures', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pseudoLegalCaptures(...)' code ##################

    
    
    # Obtaining the type of the subscript
    int_1783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 11), 'int')
    # Getting the type of 'board' (line 298)
    board_1784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 5), 'board')
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___1785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 5), board_1784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_1786 = invoke(stypy.reporting.localization.Localization(__file__, 298, 5), getitem___1785, int_1783)
    
    # Testing the type of an if condition (line 298)
    if_condition_1787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 2), subscript_call_result_1786)
    # Assigning a type to the variable 'if_condition_1787' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 2), 'if_condition_1787', if_condition_1787)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to pseudoLegalCapturesWhite(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'board' (line 299)
    board_1789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'board', False)
    # Processing the call keyword arguments (line 299)
    kwargs_1790 = {}
    # Getting the type of 'pseudoLegalCapturesWhite' (line 299)
    pseudoLegalCapturesWhite_1788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'pseudoLegalCapturesWhite', False)
    # Calling pseudoLegalCapturesWhite(args, kwargs) (line 299)
    pseudoLegalCapturesWhite_call_result_1791 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), pseudoLegalCapturesWhite_1788, *[board_1789], **kwargs_1790)
    
    # Assigning a type to the variable 'stypy_return_type' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type', pseudoLegalCapturesWhite_call_result_1791)
    # SSA branch for the else part of an if statement (line 298)
    module_type_store.open_ssa_branch('else')
    
    # Call to pseudoLegalCapturesBlack(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'board' (line 301)
    board_1793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'board', False)
    # Processing the call keyword arguments (line 301)
    kwargs_1794 = {}
    # Getting the type of 'pseudoLegalCapturesBlack' (line 301)
    pseudoLegalCapturesBlack_1792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'pseudoLegalCapturesBlack', False)
    # Calling pseudoLegalCapturesBlack(args, kwargs) (line 301)
    pseudoLegalCapturesBlack_call_result_1795 = invoke(stypy.reporting.localization.Localization(__file__, 301, 11), pseudoLegalCapturesBlack_1792, *[board_1793], **kwargs_1794)
    
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type', pseudoLegalCapturesBlack_call_result_1795)
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pseudoLegalCaptures(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pseudoLegalCaptures' in the type store
    # Getting the type of 'stypy_return_type' (line 297)
    stypy_return_type_1796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1796)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pseudoLegalCaptures'
    return stypy_return_type_1796

# Assigning a type to the variable 'pseudoLegalCaptures' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'pseudoLegalCaptures', pseudoLegalCaptures)

@norecursion
def legalMoves(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legalMoves'
    module_type_store = module_type_store.open_function_context('legalMoves', 303, 0, False)
    
    # Passed parameters checking function
    legalMoves.stypy_localization = localization
    legalMoves.stypy_type_of_self = None
    legalMoves.stypy_type_store = module_type_store
    legalMoves.stypy_function_name = 'legalMoves'
    legalMoves.stypy_param_names_list = ['board']
    legalMoves.stypy_varargs_param_name = None
    legalMoves.stypy_kwargs_param_name = None
    legalMoves.stypy_call_defaults = defaults
    legalMoves.stypy_call_varargs = varargs
    legalMoves.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legalMoves', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legalMoves', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legalMoves(...)' code ##################

    
    # Assigning a Call to a Name (line 304):
    
    # Call to pseudoLegalMoves(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'board' (line 304)
    board_1798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 30), 'board', False)
    # Processing the call keyword arguments (line 304)
    kwargs_1799 = {}
    # Getting the type of 'pseudoLegalMoves' (line 304)
    pseudoLegalMoves_1797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 13), 'pseudoLegalMoves', False)
    # Calling pseudoLegalMoves(args, kwargs) (line 304)
    pseudoLegalMoves_call_result_1800 = invoke(stypy.reporting.localization.Localization(__file__, 304, 13), pseudoLegalMoves_1797, *[board_1798], **kwargs_1799)
    
    # Assigning a type to the variable 'allMoves' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 2), 'allMoves', pseudoLegalMoves_call_result_1800)
    
    # Assigning a List to a Name (line 305):
    
    # Obtaining an instance of the builtin type 'list' (line 305)
    list_1801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 305)
    
    # Assigning a type to the variable 'retval' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 2), 'retval', list_1801)
    
    # Assigning a Num to a Name (line 307):
    int_1802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 12), 'int')
    # Assigning a type to the variable 'kingVal' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 2), 'kingVal', int_1802)
    
    
    # Obtaining the type of the subscript
    int_1803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 11), 'int')
    # Getting the type of 'board' (line 308)
    board_1804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 5), 'board')
    # Obtaining the member '__getitem__' of a type (line 308)
    getitem___1805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 5), board_1804, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 308)
    subscript_call_result_1806 = invoke(stypy.reporting.localization.Localization(__file__, 308, 5), getitem___1805, int_1803)
    
    # Testing the type of an if condition (line 308)
    if_condition_1807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 2), subscript_call_result_1806)
    # Assigning a type to the variable 'if_condition_1807' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 2), 'if_condition_1807', if_condition_1807)
    # SSA begins for if statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 309):
    
    # Getting the type of 'kingVal' (line 309)
    kingVal_1808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), 'kingVal')
    # Applying the 'usub' unary operator (line 309)
    result___neg___1809 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 14), 'usub', kingVal_1808)
    
    # Assigning a type to the variable 'kingVal' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'kingVal', result___neg___1809)
    # SSA join for if statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'allMoves' (line 310)
    allMoves_1810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'allMoves')
    # Testing the type of a for loop iterable (line 310)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 2), allMoves_1810)
    # Getting the type of the for loop variable (line 310)
    for_loop_var_1811 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 2), allMoves_1810)
    # Assigning a type to the variable 'mv' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 2), 'mv', for_loop_var_1811)
    # SSA begins for a for statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 311):
    
    # Call to copy(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'board' (line 311)
    board_1813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'board', False)
    # Processing the call keyword arguments (line 311)
    kwargs_1814 = {}
    # Getting the type of 'copy' (line 311)
    copy_1812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 13), 'copy', False)
    # Calling copy(args, kwargs) (line 311)
    copy_call_result_1815 = invoke(stypy.reporting.localization.Localization(__file__, 311, 13), copy_1812, *[board_1813], **kwargs_1814)
    
    # Assigning a type to the variable 'board2' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'board2', copy_call_result_1815)
    
    # Call to move(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'board2' (line 312)
    board2_1817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 9), 'board2', False)
    # Getting the type of 'mv' (line 312)
    mv_1818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'mv', False)
    # Processing the call keyword arguments (line 312)
    kwargs_1819 = {}
    # Getting the type of 'move' (line 312)
    move_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'move', False)
    # Calling move(args, kwargs) (line 312)
    move_call_result_1820 = invoke(stypy.reporting.localization.Localization(__file__, 312, 4), move_1816, *[board2_1817, mv_1818], **kwargs_1819)
    
    
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to pseudoLegalCaptures(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'board2' (line 314)
    board2_1831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 43), 'board2', False)
    # Processing the call keyword arguments (line 314)
    kwargs_1832 = {}
    # Getting the type of 'pseudoLegalCaptures' (line 314)
    pseudoLegalCaptures_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 23), 'pseudoLegalCaptures', False)
    # Calling pseudoLegalCaptures(args, kwargs) (line 314)
    pseudoLegalCaptures_call_result_1833 = invoke(stypy.reporting.localization.Localization(__file__, 314, 23), pseudoLegalCaptures_1830, *[board2_1831], **kwargs_1832)
    
    comprehension_1834 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 12), pseudoLegalCaptures_call_result_1833)
    # Assigning a type to the variable 'i' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'i', comprehension_1834)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 314)
    i_1822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 61), 'i')
    int_1823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 65), 'int')
    # Applying the binary operator '&' (line 314)
    result_and__1824 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 61), '&', i_1822, int_1823)
    
    # Getting the type of 'board2' (line 314)
    board2_1825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 54), 'board2')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___1826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 54), board2_1825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_1827 = invoke(stypy.reporting.localization.Localization(__file__, 314, 54), getitem___1826, result_and__1824)
    
    # Getting the type of 'kingVal' (line 314)
    kingVal_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 74), 'kingVal')
    # Applying the binary operator '==' (line 314)
    result_eq_1829 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 54), '==', subscript_call_result_1827, kingVal_1828)
    
    # Getting the type of 'i' (line 314)
    i_1821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'i')
    list_1835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 12), list_1835, i_1821)
    # Applying the 'not' unary operator (line 314)
    result_not__1836 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 7), 'not', list_1835)
    
    # Testing the type of an if condition (line 314)
    if_condition_1837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 4), result_not__1836)
    # Assigning a type to the variable 'if_condition_1837' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'if_condition_1837', if_condition_1837)
    # SSA begins for if statement (line 314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'mv' (line 315)
    mv_1840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'mv', False)
    # Processing the call keyword arguments (line 315)
    kwargs_1841 = {}
    # Getting the type of 'retval' (line 315)
    retval_1838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 6), 'retval', False)
    # Obtaining the member 'append' of a type (line 315)
    append_1839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 6), retval_1838, 'append')
    # Calling append(args, kwargs) (line 315)
    append_call_result_1842 = invoke(stypy.reporting.localization.Localization(__file__, 315, 6), append_1839, *[mv_1840], **kwargs_1841)
    
    # SSA join for if statement (line 314)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'retval' (line 316)
    retval_1843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 9), 'retval')
    # Assigning a type to the variable 'stypy_return_type' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 2), 'stypy_return_type', retval_1843)
    
    # ################# End of 'legalMoves(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legalMoves' in the type store
    # Getting the type of 'stypy_return_type' (line 303)
    stypy_return_type_1844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1844)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legalMoves'
    return stypy_return_type_1844

# Assigning a type to the variable 'legalMoves' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'legalMoves', legalMoves)

@norecursion
def alphaBetaQui(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'alphaBetaQui'
    module_type_store = module_type_store.open_function_context('alphaBetaQui', 318, 0, False)
    
    # Passed parameters checking function
    alphaBetaQui.stypy_localization = localization
    alphaBetaQui.stypy_type_of_self = None
    alphaBetaQui.stypy_type_store = module_type_store
    alphaBetaQui.stypy_function_name = 'alphaBetaQui'
    alphaBetaQui.stypy_param_names_list = ['board', 'alpha', 'beta', 'n']
    alphaBetaQui.stypy_varargs_param_name = None
    alphaBetaQui.stypy_kwargs_param_name = None
    alphaBetaQui.stypy_call_defaults = defaults
    alphaBetaQui.stypy_call_varargs = varargs
    alphaBetaQui.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'alphaBetaQui', ['board', 'alpha', 'beta', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'alphaBetaQui', localization, ['board', 'alpha', 'beta', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'alphaBetaQui(...)' code ##################

    
    # Assigning a Call to a Name (line 319):
    
    # Call to evaluate(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'board' (line 319)
    board_1846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'board', False)
    # Processing the call keyword arguments (line 319)
    kwargs_1847 = {}
    # Getting the type of 'evaluate' (line 319)
    evaluate_1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 6), 'evaluate', False)
    # Calling evaluate(args, kwargs) (line 319)
    evaluate_call_result_1848 = invoke(stypy.reporting.localization.Localization(__file__, 319, 6), evaluate_1845, *[board_1846], **kwargs_1847)
    
    # Assigning a type to the variable 'e' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 2), 'e', evaluate_call_result_1848)
    
    
    
    # Obtaining the type of the subscript
    int_1849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 15), 'int')
    # Getting the type of 'board' (line 320)
    board_1850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 9), 'board')
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___1851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 9), board_1850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_1852 = invoke(stypy.reporting.localization.Localization(__file__, 320, 9), getitem___1851, int_1849)
    
    # Applying the 'not' unary operator (line 320)
    result_not__1853 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 5), 'not', subscript_call_result_1852)
    
    # Testing the type of an if condition (line 320)
    if_condition_1854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 2), result_not__1853)
    # Assigning a type to the variable 'if_condition_1854' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 2), 'if_condition_1854', if_condition_1854)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 321):
    
    # Getting the type of 'e' (line 321)
    e_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 9), 'e')
    # Applying the 'usub' unary operator (line 321)
    result___neg___1856 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 8), 'usub', e_1855)
    
    # Assigning a type to the variable 'e' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'e', result___neg___1856)
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'e' (line 322)
    e_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 5), 'e')
    # Getting the type of 'beta' (line 322)
    beta_1858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 10), 'beta')
    # Applying the binary operator '>=' (line 322)
    result_ge_1859 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 5), '>=', e_1857, beta_1858)
    
    # Testing the type of an if condition (line 322)
    if_condition_1860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 2), result_ge_1859)
    # Assigning a type to the variable 'if_condition_1860' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 2), 'if_condition_1860', if_condition_1860)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 323)
    tuple_1861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 323)
    # Adding element type (line 323)
    # Getting the type of 'beta' (line 323)
    beta_1862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 12), tuple_1861, beta_1862)
    # Adding element type (line 323)
    # Getting the type of 'iNone' (line 323)
    iNone_1863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 18), 'iNone')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 12), tuple_1861, iNone_1863)
    
    # Assigning a type to the variable 'stypy_return_type' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type', tuple_1861)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'e' (line 324)
    e_1864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 6), 'e')
    # Getting the type of 'alpha' (line 324)
    alpha_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 10), 'alpha')
    # Applying the binary operator '>' (line 324)
    result_gt_1866 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 6), '>', e_1864, alpha_1865)
    
    # Testing the type of an if condition (line 324)
    if_condition_1867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 2), result_gt_1866)
    # Assigning a type to the variable 'if_condition_1867' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 2), 'if_condition_1867', if_condition_1867)
    # SSA begins for if statement (line 324)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 325):
    # Getting the type of 'e' (line 325)
    e_1868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'e')
    # Assigning a type to the variable 'alpha' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'alpha', e_1868)
    # SSA join for if statement (line 324)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 326):
    # Getting the type of 'iNone' (line 326)
    iNone_1869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'iNone')
    # Assigning a type to the variable 'bestMove' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 2), 'bestMove', iNone_1869)
    
    
    # Getting the type of 'n' (line 327)
    n_1870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 5), 'n')
    int_1871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 10), 'int')
    # Applying the binary operator '>=' (line 327)
    result_ge_1872 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 5), '>=', n_1870, int_1871)
    
    # Testing the type of an if condition (line 327)
    if_condition_1873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 2), result_ge_1872)
    # Assigning a type to the variable 'if_condition_1873' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 2), 'if_condition_1873', if_condition_1873)
    # SSA begins for if statement (line 327)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to pseudoLegalCaptures(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'board' (line 329)
    board_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 34), 'board', False)
    # Processing the call keyword arguments (line 329)
    kwargs_1876 = {}
    # Getting the type of 'pseudoLegalCaptures' (line 329)
    pseudoLegalCaptures_1874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 14), 'pseudoLegalCaptures', False)
    # Calling pseudoLegalCaptures(args, kwargs) (line 329)
    pseudoLegalCaptures_call_result_1877 = invoke(stypy.reporting.localization.Localization(__file__, 329, 14), pseudoLegalCaptures_1874, *[board_1875], **kwargs_1876)
    
    # Testing the type of a for loop iterable (line 329)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 329, 4), pseudoLegalCaptures_call_result_1877)
    # Getting the type of the for loop variable (line 329)
    for_loop_var_1878 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 329, 4), pseudoLegalCaptures_call_result_1877)
    # Assigning a type to the variable 'mv' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'mv', for_loop_var_1878)
    # SSA begins for a for statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 330):
    
    # Call to copy(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'board' (line 330)
    board_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'board', False)
    # Processing the call keyword arguments (line 330)
    kwargs_1881 = {}
    # Getting the type of 'copy' (line 330)
    copy_1879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'copy', False)
    # Calling copy(args, kwargs) (line 330)
    copy_call_result_1882 = invoke(stypy.reporting.localization.Localization(__file__, 330, 17), copy_1879, *[board_1880], **kwargs_1881)
    
    # Assigning a type to the variable 'newboard' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 6), 'newboard', copy_call_result_1882)
    
    # Call to move(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'newboard' (line 331)
    newboard_1884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'newboard', False)
    # Getting the type of 'mv' (line 331)
    mv_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'mv', False)
    # Processing the call keyword arguments (line 331)
    kwargs_1886 = {}
    # Getting the type of 'move' (line 331)
    move_1883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 6), 'move', False)
    # Calling move(args, kwargs) (line 331)
    move_call_result_1887 = invoke(stypy.reporting.localization.Localization(__file__, 331, 6), move_1883, *[newboard_1884, mv_1885], **kwargs_1886)
    
    
    # Assigning a Call to a Name (line 332):
    
    # Call to alphaBetaQui(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'newboard' (line 332)
    newboard_1889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 27), 'newboard', False)
    
    # Getting the type of 'beta' (line 332)
    beta_1890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 38), 'beta', False)
    # Applying the 'usub' unary operator (line 332)
    result___neg___1891 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 37), 'usub', beta_1890)
    
    
    # Getting the type of 'alpha' (line 332)
    alpha_1892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 45), 'alpha', False)
    # Applying the 'usub' unary operator (line 332)
    result___neg___1893 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 44), 'usub', alpha_1892)
    
    # Getting the type of 'n' (line 332)
    n_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 52), 'n', False)
    int_1895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 56), 'int')
    # Applying the binary operator '-' (line 332)
    result_sub_1896 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 52), '-', n_1894, int_1895)
    
    # Processing the call keyword arguments (line 332)
    kwargs_1897 = {}
    # Getting the type of 'alphaBetaQui' (line 332)
    alphaBetaQui_1888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'alphaBetaQui', False)
    # Calling alphaBetaQui(args, kwargs) (line 332)
    alphaBetaQui_call_result_1898 = invoke(stypy.reporting.localization.Localization(__file__, 332, 14), alphaBetaQui_1888, *[newboard_1889, result___neg___1891, result___neg___1893, result_sub_1896], **kwargs_1897)
    
    # Assigning a type to the variable 'value' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 6), 'value', alphaBetaQui_call_result_1898)
    
    # Assigning a Tuple to a Name (line 333):
    
    # Obtaining an instance of the builtin type 'tuple' (line 333)
    tuple_1899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 333)
    # Adding element type (line 333)
    
    
    # Obtaining the type of the subscript
    int_1900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 22), 'int')
    # Getting the type of 'value' (line 333)
    value_1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'value')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___1902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), value_1901, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_1903 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), getitem___1902, int_1900)
    
    # Applying the 'usub' unary operator (line 333)
    result___neg___1904 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 15), 'usub', subscript_call_result_1903)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 15), tuple_1899, result___neg___1904)
    # Adding element type (line 333)
    
    # Obtaining the type of the subscript
    int_1905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 32), 'int')
    # Getting the type of 'value' (line 333)
    value_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 26), 'value')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___1907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 26), value_1906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_1908 = invoke(stypy.reporting.localization.Localization(__file__, 333, 26), getitem___1907, int_1905)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 15), tuple_1899, subscript_call_result_1908)
    
    # Assigning a type to the variable 'value' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 6), 'value', tuple_1899)
    
    
    
    # Obtaining the type of the subscript
    int_1909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 15), 'int')
    # Getting the type of 'value' (line 334)
    value_1910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 9), 'value')
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___1911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 9), value_1910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_1912 = invoke(stypy.reporting.localization.Localization(__file__, 334, 9), getitem___1911, int_1909)
    
    # Getting the type of 'beta' (line 334)
    beta_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 21), 'beta')
    # Applying the binary operator '>=' (line 334)
    result_ge_1914 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 9), '>=', subscript_call_result_1912, beta_1913)
    
    # Testing the type of an if condition (line 334)
    if_condition_1915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 6), result_ge_1914)
    # Assigning a type to the variable 'if_condition_1915' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 6), 'if_condition_1915', if_condition_1915)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 335)
    tuple_1916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 335)
    # Adding element type (line 335)
    # Getting the type of 'beta' (line 335)
    beta_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 16), tuple_1916, beta_1917)
    # Adding element type (line 335)
    # Getting the type of 'mv' (line 335)
    mv_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'mv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 16), tuple_1916, mv_1918)
    
    # Assigning a type to the variable 'stypy_return_type' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'stypy_return_type', tuple_1916)
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_1919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'int')
    # Getting the type of 'value' (line 336)
    value_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 10), 'value')
    # Obtaining the member '__getitem__' of a type (line 336)
    getitem___1921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 10), value_1920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 336)
    subscript_call_result_1922 = invoke(stypy.reporting.localization.Localization(__file__, 336, 10), getitem___1921, int_1919)
    
    # Getting the type of 'alpha' (line 336)
    alpha_1923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'alpha')
    # Applying the binary operator '>' (line 336)
    result_gt_1924 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 10), '>', subscript_call_result_1922, alpha_1923)
    
    # Testing the type of an if condition (line 336)
    if_condition_1925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 6), result_gt_1924)
    # Assigning a type to the variable 'if_condition_1925' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 6), 'if_condition_1925', if_condition_1925)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    int_1926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 22), 'int')
    # Getting the type of 'value' (line 337)
    value_1927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'value')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___1928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 16), value_1927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_1929 = invoke(stypy.reporting.localization.Localization(__file__, 337, 16), getitem___1928, int_1926)
    
    # Assigning a type to the variable 'alpha' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'alpha', subscript_call_result_1929)
    
    # Assigning a Name to a Name (line 338):
    # Getting the type of 'mv' (line 338)
    mv_1930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'mv')
    # Assigning a type to the variable 'bestMove' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'bestMove', mv_1930)
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 327)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 339)
    tuple_1931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 339)
    # Adding element type (line 339)
    # Getting the type of 'alpha' (line 339)
    alpha_1932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 10), 'alpha')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 10), tuple_1931, alpha_1932)
    # Adding element type (line 339)
    # Getting the type of 'bestMove' (line 339)
    bestMove_1933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'bestMove')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 10), tuple_1931, bestMove_1933)
    
    # Assigning a type to the variable 'stypy_return_type' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 2), 'stypy_return_type', tuple_1931)
    
    # ################# End of 'alphaBetaQui(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'alphaBetaQui' in the type store
    # Getting the type of 'stypy_return_type' (line 318)
    stypy_return_type_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1934)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'alphaBetaQui'
    return stypy_return_type_1934

# Assigning a type to the variable 'alphaBetaQui' (line 318)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'alphaBetaQui', alphaBetaQui)

@norecursion
def alphaBeta(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'alphaBeta'
    module_type_store = module_type_store.open_function_context('alphaBeta', 341, 0, False)
    
    # Passed parameters checking function
    alphaBeta.stypy_localization = localization
    alphaBeta.stypy_type_of_self = None
    alphaBeta.stypy_type_store = module_type_store
    alphaBeta.stypy_function_name = 'alphaBeta'
    alphaBeta.stypy_param_names_list = ['board', 'alpha', 'beta', 'n']
    alphaBeta.stypy_varargs_param_name = None
    alphaBeta.stypy_kwargs_param_name = None
    alphaBeta.stypy_call_defaults = defaults
    alphaBeta.stypy_call_varargs = varargs
    alphaBeta.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'alphaBeta', ['board', 'alpha', 'beta', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'alphaBeta', localization, ['board', 'alpha', 'beta', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'alphaBeta(...)' code ##################

    
    
    # Getting the type of 'n' (line 342)
    n_1935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 5), 'n')
    int_1936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 10), 'int')
    # Applying the binary operator '==' (line 342)
    result_eq_1937 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 5), '==', n_1935, int_1936)
    
    # Testing the type of an if condition (line 342)
    if_condition_1938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 2), result_eq_1937)
    # Assigning a type to the variable 'if_condition_1938' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 2), 'if_condition_1938', if_condition_1938)
    # SSA begins for if statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to alphaBetaQui(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'board' (line 343)
    board_1940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'board', False)
    # Getting the type of 'alpha' (line 343)
    alpha_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'alpha', False)
    # Getting the type of 'beta' (line 343)
    beta_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 38), 'beta', False)
    # Getting the type of 'n' (line 343)
    n_1943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 44), 'n', False)
    # Processing the call keyword arguments (line 343)
    kwargs_1944 = {}
    # Getting the type of 'alphaBetaQui' (line 343)
    alphaBetaQui_1939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 'alphaBetaQui', False)
    # Calling alphaBetaQui(args, kwargs) (line 343)
    alphaBetaQui_call_result_1945 = invoke(stypy.reporting.localization.Localization(__file__, 343, 11), alphaBetaQui_1939, *[board_1940, alpha_1941, beta_1942, n_1943], **kwargs_1944)
    
    # Assigning a type to the variable 'stypy_return_type' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type', alphaBetaQui_call_result_1945)
    # SSA join for if statement (line 342)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'iNone' (line 345)
    iNone_1946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 13), 'iNone')
    # Assigning a type to the variable 'bestMove' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 2), 'bestMove', iNone_1946)
    
    
    # Call to legalMoves(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'board' (line 347)
    board_1948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'board', False)
    # Processing the call keyword arguments (line 347)
    kwargs_1949 = {}
    # Getting the type of 'legalMoves' (line 347)
    legalMoves_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'legalMoves', False)
    # Calling legalMoves(args, kwargs) (line 347)
    legalMoves_call_result_1950 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), legalMoves_1947, *[board_1948], **kwargs_1949)
    
    # Testing the type of a for loop iterable (line 347)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 347, 2), legalMoves_call_result_1950)
    # Getting the type of the for loop variable (line 347)
    for_loop_var_1951 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 347, 2), legalMoves_call_result_1950)
    # Assigning a type to the variable 'mv' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 2), 'mv', for_loop_var_1951)
    # SSA begins for a for statement (line 347)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 348):
    
    # Call to copy(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'board' (line 348)
    board_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'board', False)
    # Processing the call keyword arguments (line 348)
    kwargs_1954 = {}
    # Getting the type of 'copy' (line 348)
    copy_1952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 15), 'copy', False)
    # Calling copy(args, kwargs) (line 348)
    copy_call_result_1955 = invoke(stypy.reporting.localization.Localization(__file__, 348, 15), copy_1952, *[board_1953], **kwargs_1954)
    
    # Assigning a type to the variable 'newboard' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'newboard', copy_call_result_1955)
    
    # Call to move(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'newboard' (line 349)
    newboard_1957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 9), 'newboard', False)
    # Getting the type of 'mv' (line 349)
    mv_1958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'mv', False)
    # Processing the call keyword arguments (line 349)
    kwargs_1959 = {}
    # Getting the type of 'move' (line 349)
    move_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'move', False)
    # Calling move(args, kwargs) (line 349)
    move_call_result_1960 = invoke(stypy.reporting.localization.Localization(__file__, 349, 4), move_1956, *[newboard_1957, mv_1958], **kwargs_1959)
    
    
    # Assigning a Call to a Name (line 350):
    
    # Call to alphaBeta(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'newboard' (line 350)
    newboard_1962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'newboard', False)
    
    # Getting the type of 'beta' (line 350)
    beta_1963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 33), 'beta', False)
    # Applying the 'usub' unary operator (line 350)
    result___neg___1964 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 32), 'usub', beta_1963)
    
    
    # Getting the type of 'alpha' (line 350)
    alpha_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 40), 'alpha', False)
    # Applying the 'usub' unary operator (line 350)
    result___neg___1966 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 39), 'usub', alpha_1965)
    
    # Getting the type of 'n' (line 350)
    n_1967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 47), 'n', False)
    int_1968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 51), 'int')
    # Applying the binary operator '-' (line 350)
    result_sub_1969 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 47), '-', n_1967, int_1968)
    
    # Processing the call keyword arguments (line 350)
    kwargs_1970 = {}
    # Getting the type of 'alphaBeta' (line 350)
    alphaBeta_1961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'alphaBeta', False)
    # Calling alphaBeta(args, kwargs) (line 350)
    alphaBeta_call_result_1971 = invoke(stypy.reporting.localization.Localization(__file__, 350, 12), alphaBeta_1961, *[newboard_1962, result___neg___1964, result___neg___1966, result_sub_1969], **kwargs_1970)
    
    # Assigning a type to the variable 'value' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'value', alphaBeta_call_result_1971)
    
    # Assigning a Tuple to a Name (line 351):
    
    # Obtaining an instance of the builtin type 'tuple' (line 351)
    tuple_1972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 351)
    # Adding element type (line 351)
    
    
    # Obtaining the type of the subscript
    int_1973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 20), 'int')
    # Getting the type of 'value' (line 351)
    value_1974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 14), 'value')
    # Obtaining the member '__getitem__' of a type (line 351)
    getitem___1975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 14), value_1974, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 351)
    subscript_call_result_1976 = invoke(stypy.reporting.localization.Localization(__file__, 351, 14), getitem___1975, int_1973)
    
    # Applying the 'usub' unary operator (line 351)
    result___neg___1977 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 13), 'usub', subscript_call_result_1976)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 13), tuple_1972, result___neg___1977)
    # Adding element type (line 351)
    
    # Obtaining the type of the subscript
    int_1978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 30), 'int')
    # Getting the type of 'value' (line 351)
    value_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'value')
    # Obtaining the member '__getitem__' of a type (line 351)
    getitem___1980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), value_1979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 351)
    subscript_call_result_1981 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), getitem___1980, int_1978)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 13), tuple_1972, subscript_call_result_1981)
    
    # Assigning a type to the variable 'value' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'value', tuple_1972)
    
    
    
    # Obtaining the type of the subscript
    int_1982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 13), 'int')
    # Getting the type of 'value' (line 352)
    value_1983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 7), 'value')
    # Obtaining the member '__getitem__' of a type (line 352)
    getitem___1984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 7), value_1983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 352)
    subscript_call_result_1985 = invoke(stypy.reporting.localization.Localization(__file__, 352, 7), getitem___1984, int_1982)
    
    # Getting the type of 'beta' (line 352)
    beta_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 19), 'beta')
    # Applying the binary operator '>=' (line 352)
    result_ge_1987 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 7), '>=', subscript_call_result_1985, beta_1986)
    
    # Testing the type of an if condition (line 352)
    if_condition_1988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 4), result_ge_1987)
    # Assigning a type to the variable 'if_condition_1988' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'if_condition_1988', if_condition_1988)
    # SSA begins for if statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 353)
    tuple_1989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 353)
    # Adding element type (line 353)
    # Getting the type of 'beta' (line 353)
    beta_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 14), tuple_1989, beta_1990)
    # Adding element type (line 353)
    # Getting the type of 'mv' (line 353)
    mv_1991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'mv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 14), tuple_1989, mv_1991)
    
    # Assigning a type to the variable 'stypy_return_type' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 6), 'stypy_return_type', tuple_1989)
    # SSA join for if statement (line 352)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_1992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 14), 'int')
    # Getting the type of 'value' (line 354)
    value_1993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'value')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___1994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), value_1993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_1995 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___1994, int_1992)
    
    # Getting the type of 'alpha' (line 354)
    alpha_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'alpha')
    # Applying the binary operator '>' (line 354)
    result_gt_1997 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 8), '>', subscript_call_result_1995, alpha_1996)
    
    # Testing the type of an if condition (line 354)
    if_condition_1998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 4), result_gt_1997)
    # Assigning a type to the variable 'if_condition_1998' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'if_condition_1998', if_condition_1998)
    # SSA begins for if statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 355):
    
    # Obtaining the type of the subscript
    int_1999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 20), 'int')
    # Getting the type of 'value' (line 355)
    value_2000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'value')
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___2001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 14), value_2000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_2002 = invoke(stypy.reporting.localization.Localization(__file__, 355, 14), getitem___2001, int_1999)
    
    # Assigning a type to the variable 'alpha' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 6), 'alpha', subscript_call_result_2002)
    
    # Assigning a Name to a Name (line 356):
    # Getting the type of 'mv' (line 356)
    mv_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 17), 'mv')
    # Assigning a type to the variable 'bestMove' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 6), 'bestMove', mv_2003)
    # SSA join for if statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 357)
    tuple_2004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 357)
    # Adding element type (line 357)
    # Getting the type of 'alpha' (line 357)
    alpha_2005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 10), 'alpha')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 10), tuple_2004, alpha_2005)
    # Adding element type (line 357)
    # Getting the type of 'bestMove' (line 357)
    bestMove_2006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'bestMove')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 10), tuple_2004, bestMove_2006)
    
    # Assigning a type to the variable 'stypy_return_type' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 2), 'stypy_return_type', tuple_2004)
    
    # ################# End of 'alphaBeta(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'alphaBeta' in the type store
    # Getting the type of 'stypy_return_type' (line 341)
    stypy_return_type_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2007)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'alphaBeta'
    return stypy_return_type_2007

# Assigning a type to the variable 'alphaBeta' (line 341)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'alphaBeta', alphaBeta)

@norecursion
def speedTest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'speedTest'
    module_type_store = module_type_store.open_function_context('speedTest', 359, 0, False)
    
    # Passed parameters checking function
    speedTest.stypy_localization = localization
    speedTest.stypy_type_of_self = None
    speedTest.stypy_type_store = module_type_store
    speedTest.stypy_function_name = 'speedTest'
    speedTest.stypy_param_names_list = []
    speedTest.stypy_varargs_param_name = None
    speedTest.stypy_kwargs_param_name = None
    speedTest.stypy_call_defaults = defaults
    speedTest.stypy_call_varargs = varargs
    speedTest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'speedTest', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'speedTest', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'speedTest(...)' code ##################

    
    # Assigning a Call to a Name (line 360):
    
    # Call to list(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'setup' (line 360)
    setup_2009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'setup', False)
    # Processing the call keyword arguments (line 360)
    kwargs_2010 = {}
    # Getting the type of 'list' (line 360)
    list_2008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 10), 'list', False)
    # Calling list(args, kwargs) (line 360)
    list_call_result_2011 = invoke(stypy.reporting.localization.Localization(__file__, 360, 10), list_2008, *[setup_2009], **kwargs_2010)
    
    # Assigning a type to the variable 'board' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 2), 'board', list_call_result_2011)
    
    # Call to moveStr(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'board' (line 361)
    board_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 10), 'board', False)
    str_2014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 17), 'str', 'c2-c4')
    # Processing the call keyword arguments (line 361)
    kwargs_2015 = {}
    # Getting the type of 'moveStr' (line 361)
    moveStr_2012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 2), 'moveStr', False)
    # Calling moveStr(args, kwargs) (line 361)
    moveStr_call_result_2016 = invoke(stypy.reporting.localization.Localization(__file__, 361, 2), moveStr_2012, *[board_2013, str_2014], **kwargs_2015)
    
    
    # Call to moveStr(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of 'board' (line 362)
    board_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 10), 'board', False)
    str_2019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 17), 'str', 'e7-e5')
    # Processing the call keyword arguments (line 362)
    kwargs_2020 = {}
    # Getting the type of 'moveStr' (line 362)
    moveStr_2017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 2), 'moveStr', False)
    # Calling moveStr(args, kwargs) (line 362)
    moveStr_call_result_2021 = invoke(stypy.reporting.localization.Localization(__file__, 362, 2), moveStr_2017, *[board_2018, str_2019], **kwargs_2020)
    
    
    # Call to moveStr(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'board' (line 363)
    board_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 10), 'board', False)
    str_2024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 17), 'str', 'd2-d4')
    # Processing the call keyword arguments (line 363)
    kwargs_2025 = {}
    # Getting the type of 'moveStr' (line 363)
    moveStr_2022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 2), 'moveStr', False)
    # Calling moveStr(args, kwargs) (line 363)
    moveStr_call_result_2026 = invoke(stypy.reporting.localization.Localization(__file__, 363, 2), moveStr_2022, *[board_2023, str_2024], **kwargs_2025)
    
    
    # Assigning a Call to a Name (line 365):
    
    # Call to alphaBeta(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'board' (line 365)
    board_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 18), 'board', False)
    int_2029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 25), 'int')
    int_2030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 36), 'int')
    int_2031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 46), 'int')
    # Processing the call keyword arguments (line 365)
    kwargs_2032 = {}
    # Getting the type of 'alphaBeta' (line 365)
    alphaBeta_2027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'alphaBeta', False)
    # Calling alphaBeta(args, kwargs) (line 365)
    alphaBeta_call_result_2033 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), alphaBeta_2027, *[board_2028, int_2029, int_2030, int_2031], **kwargs_2032)
    
    # Assigning a type to the variable 'res' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 2), 'res', alphaBeta_call_result_2033)
    
    # Call to moveStr(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'board' (line 367)
    board_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 10), 'board', False)
    str_2036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 17), 'str', 'd7-d6')
    # Processing the call keyword arguments (line 367)
    kwargs_2037 = {}
    # Getting the type of 'moveStr' (line 367)
    moveStr_2034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 2), 'moveStr', False)
    # Calling moveStr(args, kwargs) (line 367)
    moveStr_call_result_2038 = invoke(stypy.reporting.localization.Localization(__file__, 367, 2), moveStr_2034, *[board_2035, str_2036], **kwargs_2037)
    
    
    # Assigning a Call to a Name (line 368):
    
    # Call to alphaBeta(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'board' (line 368)
    board_2040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 18), 'board', False)
    int_2041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 25), 'int')
    int_2042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 36), 'int')
    int_2043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 46), 'int')
    # Processing the call keyword arguments (line 368)
    kwargs_2044 = {}
    # Getting the type of 'alphaBeta' (line 368)
    alphaBeta_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'alphaBeta', False)
    # Calling alphaBeta(args, kwargs) (line 368)
    alphaBeta_call_result_2045 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), alphaBeta_2039, *[board_2040, int_2041, int_2042, int_2043], **kwargs_2044)
    
    # Assigning a type to the variable 'res' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 2), 'res', alphaBeta_call_result_2045)
    
    # ################# End of 'speedTest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'speedTest' in the type store
    # Getting the type of 'stypy_return_type' (line 359)
    stypy_return_type_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2046)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'speedTest'
    return stypy_return_type_2046

# Assigning a type to the variable 'speedTest' (line 359)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'speedTest', speedTest)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 371, 0, False)
    
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

    
    # Call to speedTest(...): (line 372)
    # Processing the call keyword arguments (line 372)
    kwargs_2048 = {}
    # Getting the type of 'speedTest' (line 372)
    speedTest_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 2), 'speedTest', False)
    # Calling speedTest(args, kwargs) (line 372)
    speedTest_call_result_2049 = invoke(stypy.reporting.localization.Localization(__file__, 372, 2), speedTest_2047, *[], **kwargs_2048)
    
    # Getting the type of 'True' (line 373)
    True_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 9), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 2), 'stypy_return_type', True_2050)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 371)
    stypy_return_type_2051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2051)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_2051

# Assigning a type to the variable 'run' (line 371)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'run', run)

# Call to run(...): (line 375)
# Processing the call keyword arguments (line 375)
kwargs_2053 = {}
# Getting the type of 'run' (line 375)
run_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 0), 'run', False)
# Calling run(args, kwargs) (line 375)
run_call_result_2054 = invoke(stypy.reporting.localization.Localization(__file__, 375, 0), run_2052, *[], **kwargs_2053)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

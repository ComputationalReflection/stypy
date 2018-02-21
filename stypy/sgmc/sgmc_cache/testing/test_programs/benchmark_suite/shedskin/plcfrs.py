
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A natural language parser for PLCFRS (probabilistic linear context-free
3: rewriting systems). PLCFRS is an extension of context-free grammar which
4: rewrites tuples of strings instead of strings; this allows it to produce
5: parse trees with discontinuous constituents.
6: 
7: Copyright 2011 Andreas van Cranenburgh <andreas@unstable.nl>
8: This program is free software: you can redistribute it and/or modify
9: it under the terms of the GNU General Public License as published by
10: the Free Software Foundation, either version 3 of the License, or
11: (at your option) any later version.
12: 
13: This program is distributed in the hope that it will be useful,
14: but WITHOUT ANY WARRANTY; without even the implied warranty of
15: MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
16: GNU General Public License for more details.
17: 
18: You should have received a copy of the GNU General Public License
19: along with this program.  If not, see <http://www.gnu.org/licenses/>.
20: '''
21: 
22: from sys import argv, stderr
23: from math import exp, log
24: from array import array
25: from heapq import heappush, heappop, heapify
26: 
27: 
28: def parse(sent, grammar, tags, start, exhaustive):
29:     ''' parse sentence, a list of tokens, optionally with gold tags, and
30:     produce a chart, either exhaustive or up until the viterbi parse.
31:     '''
32:     unary = grammar.unary
33:     lbinary = grammar.lbinary
34:     rbinary = grammar.rbinary
35:     lexical = grammar.lexical
36:     toid = grammar.toid
37:     tolabel = grammar.tolabel
38:     goal = ChartItem(start, (1 << len(sent)) - 1)
39:     maxA = 0
40:     blocked = 0
41:     Cx = [{} for _ in toid]
42:     C = {}
43:     A = agenda()
44: 
45:     # scan: assign part-of-speech tags
46:     Epsilon = toid["Epsilon"]
47:     for i, w in enumerate(sent):
48:         recognized = False
49:         for terminal in lexical.get(w, []):
50:             if not tags or tags[i] == tolabel[terminal.lhs].split("@")[0]:
51:                 item = ChartItem(terminal.lhs, 1 << i)
52:                 I = ChartItem(Epsilon, i)
53:                 z = terminal.prob
54:                 A[item] = Edge(z, z, z, I, None)
55:                 C[item] = []
56:                 recognized = True
57:         if not recognized and tags and tags[i] in toid:
58:             item = ChartItem(toid[tags[i]], 1 << i)
59:             I = ChartItem(Epsilon, i)
60:             A[item] = Edge(0.0, 0.0, 0.0, I, None)
61:             C[item] = []
62:             recognized = True
63:         elif not recognized:
64:             ##            print "not covered:", tags[i] if tags else w
65:             return C, None
66: 
67:     # parsing
68:     while A:
69:         item, edge = A.popitem()
70:         C[item].append(edge)
71:         Cx[item.label][item] = edge
72: 
73:         if item == goal:
74:             if exhaustive:
75:                 continue
76:             else:
77:                 break
78:         for rule in unary[item.label]:
79:             blocked += process_edge(
80:                 ChartItem(rule.lhs, item.vec),
81:                 Edge(edge.inside + rule.prob, edge.inside + rule.prob,
82:                      rule.prob, item, None), A, C, exhaustive)
83:         for rule in lbinary[item.label]:
84:             for sibling in Cx[rule.rhs2]:
85:                 e = Cx[rule.rhs2][sibling]
86:                 if (item.vec & sibling.vec == 0
87:                         and concat(rule, item.vec, sibling.vec)):
88:                     blocked += process_edge(
89:                         ChartItem(rule.lhs, item.vec ^ sibling.vec),
90:                         Edge(edge.inside + e.inside + rule.prob,
91:                              edge.inside + e.inside + rule.prob,
92:                              rule.prob, item, sibling), A, C, exhaustive)
93:         for rule in rbinary[item.label]:
94:             for sibling in Cx[rule.rhs1]:
95:                 e = Cx[rule.rhs1][sibling]
96:                 if (sibling.vec & item.vec == 0
97:                         and concat(rule, sibling.vec, item.vec)):
98:                     blocked += process_edge(
99:                         ChartItem(rule.lhs, sibling.vec ^ item.vec),
100:                         Edge(e.inside + edge.inside + rule.prob,
101:                              e.inside + edge.inside + rule.prob,
102:                              rule.prob, sibling, item), A, C, exhaustive)
103:         if len(A) > maxA: maxA = len(A)
104:         # if len(A) % 10000 == 0:
105:         #    print "agenda max %d, now %d, items %d" % (maxA, len(A), len(C))
106:     ##    stderr.write("agenda max %d, now %d, items %d (%d labels), " % (
107:     ##                                maxA, len(A), len(C), len(filter(None, Cx))))
108:     ##    stderr.write("edges %d, blocked %d\n"
109:     ##							% (sum(map(len, C.values())), blocked))
110:     if goal not in C: goal = None
111:     return (C, goal)
112: 
113: 
114: def process_edge(newitem, newedge, A, C, exhaustive):
115:     if newitem not in C and newitem not in A:
116:         # prune improbable edges
117:         if newedge.score > 300.0: return 1
118:         # haven't seen this item before, add to agenda
119:         A[newitem] = newedge
120:         C[newitem] = []
121:     elif newitem in A and newedge.inside < A[newitem].inside:
122:         # item has lower score, update agenda
123:         C[newitem].append(A[newitem])
124:         A[newitem] = newedge
125:     elif exhaustive:
126:         # item is suboptimal, only add to exhaustive chart
127:         C[newitem].append(newedge)
128:     return 0
129: 
130: 
131: def concat(rule, lvec, rvec):
132:     lpos = nextset(lvec, 0)
133:     rpos = nextset(rvec, 0)
134:     # this algorithm was taken from rparse, FastYFComposer.
135:     for x in range(len(rule.args)):
136:         m = rule.lengths[x] - 1
137:         for n in range(m + 1):
138:             if testbit(rule.args[x], n):
139:                 # check if there are any bits left, and
140:                 # if any bits on the right should have gone before
141:                 # ones on this side
142:                 if rpos == -1 or (lpos != -1 and lpos <= rpos):
143:                     return False
144:                 # jump to next gap
145:                 rpos = nextunset(rvec, rpos)
146:                 if lpos != -1 and lpos < rpos:
147:                     return False
148:                 # there should be a gap if and only if
149:                 # this is the last element of this argument
150:                 if n == m:
151:                     if testbit(lvec, rpos):
152:                         return False
153:                 elif not testbit(lvec, rpos):
154:                     return False
155:                 # jump to next argument
156:                 rpos = nextset(rvec, rpos)
157:             else:
158:                 # vice versa to the above
159:                 if lpos == -1 or (rpos != -1 and rpos <= lpos):
160:                     return False
161:                 lpos = nextunset(lvec, lpos)
162:                 if rpos != -1 and rpos < lpos:
163:                     return False
164:                 if n == m:
165:                     if testbit(rvec, lpos):
166:                         return False
167:                 elif not testbit(rvec, lpos):
168:                     return False
169:                 lpos = nextset(lvec, lpos)
170:             # else: raise ValueError("non-binary element in yieldfunction")
171:     if lpos != -1 or rpos != -1:
172:         return False
173:     # everything looks all right
174:     return True
175: 
176: 
177: def mostprobablederivation(chart, start, tolabel):
178:     ''' produce a string representation of the viterbi parse in bracket
179:     notation'''
180:     edge = min(chart[start])
181:     return getmpd(chart, start, tolabel), edge.inside
182: 
183: 
184: def getmpd(chart, start, tolabel):
185:     edge = min(chart[start])
186:     if edge.right and edge.right.label:  # binary
187:         return "(%s %s %s)" % (tolabel[start.label],
188:                                getmpd(chart, edge.left, tolabel),
189:                                getmpd(chart, edge.right, tolabel))
190:     else:  # unary or terminal
191:         return "(%s %s)" % (tolabel[start.label],
192:                             getmpd(chart, edge.left, tolabel)
193:                             if edge.left.label else str(edge.left.vec))
194: 
195: 
196: def binrepr(a, sent):
197:     return "".join(reversed(bin(a.vec)[2:].rjust(len(sent), "0")))
198: 
199: 
200: def pprint_chart(chart, sent, tolabel):
201:     ##    print "chart:"
202:     for n, a in sorted((bitcount(a.vec), a) for a in chart):
203:         if not chart[a]: continue
204:         ##        print "%s[%s] =>" % (tolabel[a.label], binrepr(a, sent))
205:         binrepr(a, sent)
206:         for edge in chart[a]:
207:             ##            print "%g\t%g" % (exp(-edge.inside), exp(-edge.prob)),
208:             (exp(-edge.inside), exp(-edge.prob))
209:             if edge.left.label:
210:                 pass
211:             ##                print "\t%s[%s]" % (tolabel[edge.left.label],
212:             ##                                    binrepr(edge.left, sent)),
213:             else:
214:                 pass
215:             ##                print "\t", repr(sent[edge.left.vec]),
216:             if edge.right:
217:                 pass
218: 
219: 
220: ##                print "\t%s[%s]" % (tolabel[edge.right.label],
221: ##                                    binrepr(edge.right, sent)),
222: ##            print
223: ##        print
224: 
225: def do(sent, grammar):
226:     ##    print "sentence", sent
227:     chart, start = parse(sent.split(), grammar, None, grammar.toid['S'], False)
228:     pprint_chart(chart, sent.split(), grammar.tolabel)
229:     if start:
230:         t, p = mostprobablederivation(chart, start, grammar.tolabel)
231:         # print exp(-p), t, '\n'
232:         exp(-p)
233:     else:
234:         pass  # print "no parse"
235:     return start is not None
236: 
237: 
238: def read_srcg_grammar(rulefile, lexiconfile):
239:     ''' Reads a grammar as produced by write_srcg_grammar. '''
240:     srules = [line[:len(line) - 1].split('\t') for line in open(rulefile)]
241:     slexicon = [line[:len(line) - 1].split('\t') for line in open(lexiconfile)]
242:     rules = [((tuple(a[:len(a) - 2]), tuple(tuple(map(int, b))
243:                                             for b in a[len(a) - 2].split(","))),
244:               float(a[len(a) - 1])) for a in srules]
245:     lexicon = [((tuple(a[:len(a) - 2]), a[len(a) - 2]), float(a[len(a) - 1]))
246:                for a in slexicon]
247:     return rules, lexicon
248: 
249: 
250: def splitgrammar(grammar, lexicon):
251:     ''' split the grammar into various lookup tables, mapping nonterminal
252:     labels to numeric identifiers. Also negates log-probabilities to
253:     accommodate min-heaps.
254:     Can only represent ordered SRCG rules (monotone LCFRS). '''
255:     # get a list of all nonterminals; make sure Epsilon and ROOT are first,
256:     # and assign them unique IDs
257:     nonterminals = list(enumerate(["Epsilon", "ROOT"]
258:                                   + sorted(set(nt for (rule, yf), weight in grammar for nt in rule)
259:                                            - set(["Epsilon", "ROOT"]))))
260:     toid = dict((lhs, n) for n, lhs in nonterminals)
261:     tolabel = dict((n, lhs) for n, lhs in nonterminals)
262:     bylhs = [[] for _ in nonterminals]
263:     unary = [[] for _ in nonterminals]
264:     lbinary = [[] for _ in nonterminals]
265:     rbinary = [[] for _ in nonterminals]
266:     lexical = {}
267:     arity = array('B', [0] * len(nonterminals))
268:     for (tag, word), w in lexicon:
269:         t = Terminal(toid[tag[0]], toid[tag[1]], 0, word, abs(w))
270:         assert arity[t.lhs] in (0, 1)
271:         arity[t.lhs] = 1
272:         lexical.setdefault(word, []).append(t)
273:     for (rule, yf), w in grammar:
274:         args, lengths = yfarray(yf)
275:         assert yf == arraytoyf(args, lengths)  # unbinarized rule => error
276:         # cyclic unary productions
277:         if len(rule) == 2 and w == 0.0: w += 0.00000001
278:         r = Rule(toid[rule[0]], toid[rule[1]],
279:                  toid[rule[2]] if len(rule) == 3 else 0, args, lengths, abs(w))
280:         if arity[r.lhs] == 0:
281:             arity[r.lhs] = len(args)
282:         assert arity[r.lhs] == len(args)
283:         if len(rule) == 2:
284:             unary[r.rhs1].append(r)
285:             bylhs[r.lhs].append(r)
286:         elif len(rule) == 3:
287:             lbinary[r.rhs1].append(r)
288:             rbinary[r.rhs2].append(r)
289:             bylhs[r.lhs].append(r)
290:         else:
291:             raise ValueError("grammar not binarized: %r" % r)
292:     # assert 0 not in arity[1:]
293:     return Grammar(unary, lbinary, rbinary, lexical, bylhs, toid, tolabel)
294: 
295: 
296: def yfarray(yf):
297:     ''' convert a yield function represented as a 2D sequence to an array
298:     object. '''
299:     # I for 32 bits (int), H for 16 bits (short), B for 8 bits (char)
300:     vectype = 'I';
301:     vecsize = 32  # 8 * array(vectype).itemsize
302:     lentype = 'H';
303:     lensize = 16  # 8 * array(lentype).itemsize
304:     assert len(yf) <= lensize  # arity too high?
305:     assert all(len(a) <= vecsize for a in yf)  # too many variables?
306:     initializer = [sum(1 << n for n, b in enumerate(a) if b) for a in yf]
307:     args = array('I', initializer)
308:     lengths = array('H', map(len, yf))
309:     return args, lengths
310: 
311: 
312: def arraytoyf(args, lengths):
313:     return tuple(tuple(1 if a & (1 << m) else 0 for m in range(n))
314:                  for n, a in zip(lengths, args))
315: 
316: 
317: # bit operations
318: def nextset(a, pos):
319:     ''' First set bit, starting from pos '''
320:     result = pos
321:     if a >> result:
322:         while (a >> result) & 1 == 0:
323:             result += 1
324:         return result
325:     return -1
326: 
327: 
328: def nextunset(a, pos):
329:     ''' First unset bit, starting from pos '''
330:     result = pos
331:     while (a >> result) & 1:
332:         result += 1
333:     return result
334: 
335: 
336: def bitcount(a):
337:     ''' Number of set bits (1s) '''
338:     count = 0
339:     while a:
340:         a &= a - 1
341:         count += 1
342:     return count
343: 
344: 
345: def testbit(a, offset):
346:     ''' Mask a particular bit, return nonzero if set '''
347:     return a & (1 << offset)
348: 
349: 
350: # various data types
351: class Grammar(object):
352:     __slots__ = ('unary', 'lbinary', 'rbinary', 'lexical',
353:                  'bylhs', 'toid', 'tolabel')
354: 
355:     def __init__(self, unary, lbinary, rbinary, lexical, bylhs, toid, tolabel):
356:         self.unary = unary
357:         self.lbinary = lbinary
358:         self.rbinary = rbinary
359:         self.lexical = lexical
360:         self.bylhs = bylhs
361:         self.toid = toid
362:         self.tolabel = tolabel
363: 
364: 
365: class ChartItem:
366:     __slots__ = ("label", "vec")
367: 
368:     def __init__(self, label, vec):
369:         self.label = label  # the category of this item (NP/PP/VP etc)
370:         self.vec = vec  # bitvector describing the spans of this item
371: 
372:     def __hash__(self):
373:         # form some reason this does not work well w/shedskin:
374:         # h = self.label ^ (self.vec << 31) ^ (self.vec >> 31)
375:         # the DJB hash function:
376:         h = ((5381 << 5) + 5381) * 33 ^ self.label
377:         h = ((h << 5) + h) * 33 ^ self.vec
378:         return -2 if h == -1 else h
379: 
380:     def __eq__(self, other):
381:         if other is None: return False
382:         return self.label == other.label and self.vec == other.vec
383: 
384: 
385: class Edge:
386:     __slots__ = ('score', 'inside', 'prob', 'left', 'right')
387: 
388:     def __init__(self, score, inside, prob, left, right):
389:         self.score = score;
390:         self.inside = inside;
391:         self.prob = prob
392:         self.left = left;
393:         self.right = right
394: 
395:     def __lt__(self, other):
396:         # the ordering only depends on inside probability
397:         # (or on estimate of outside score when added)
398:         return self.score < other.score
399: 
400:     def __gt__(self, other):
401:         return self.score > other.score
402: 
403:     def __eq__(self, other):
404:         return (self.inside == other.inside
405:                 and self.prob == other.prob
406:                 and self.left == other.right
407:                 and self.right == other.right)
408: 
409: 
410: class Terminal:
411:     __slots__ = ('lhs', 'rhs1', 'rhs2', 'word', 'prob')
412: 
413:     def __init__(self, lhs, rhs1, rhs2, word, prob):
414:         self.lhs = lhs;
415:         self.rhs1 = rhs1;
416:         self.rhs2 = rhs2
417:         self.word = word;
418:         self.prob = prob
419: 
420: 
421: class Rule:
422:     __slots__ = ('lhs', 'rhs1', 'rhs2', 'prob',
423:                  'args', 'lengths', '_args', 'lengths')
424: 
425:     def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
426:         self.lhs = lhs;
427:         self.rhs1 = rhs1;
428:         self.rhs2 = rhs2
429:         self.args = args;
430:         self.lengths = lengths;
431:         self.prob = prob
432:         self._args = self.args;
433:         self._lengths = self.lengths
434: 
435: 
436: # the agenda (priority queue)
437: class Entry(object):
438:     __slots__ = ('key', 'value', 'count')
439: 
440:     def __init__(self, key, value, count):
441:         self.key = key  # the `task'
442:         self.value = value  # the priority
443:         self.count = count  # unqiue identifier to resolve ties
444: 
445:     def __eq__(self, other):
446:         return self.count == other.count
447: 
448:     def __lt__(self, other):
449:         return self.value < other.value or (self.value == other.value
450:                                             and self.count < other.count)
451: 
452: 
453: INVALID = 0
454: 
455: 
456: class agenda(object):
457:     def __init__(self):
458:         self.heap = []  # the priority queue list
459:         self.mapping = {}  # mapping of keys to entries
460:         self.counter = 1  # unique sequence count
461: 
462:     def __setitem__(self, key, value):
463:         if key in self.mapping:
464:             oldentry = self.mapping[key]
465:             entry = Entry(key, value, oldentry.count)
466:             self.mapping[key] = entry
467:             heappush(self.heap, entry)
468:             oldentry.count = INVALID
469:         else:
470:             entry = Entry(key, value, self.counter)
471:             self.counter += 1
472:             self.mapping[key] = entry
473:             heappush(self.heap, entry)
474: 
475:     def __getitem__(self, key):
476:         return self.mapping[key].value
477: 
478:     def __contains__(self, key):
479:         return key in self.mapping
480: 
481:     def __len__(self):
482:         return len(self.mapping)
483: 
484:     def popitem(self):
485:         entry = heappop(self.heap)
486:         while entry.count is INVALID:
487:             entry = heappop(self.heap)
488:         del self.mapping[entry.key]
489:         return entry.key, entry.value
490: 
491: 
492: def batch(rulefile, lexiconfile, sentfile):
493:     rules, lexicon = read_srcg_grammar(rulefile, lexiconfile)
494:     root = rules[0][0][0][0]
495:     grammar = splitgrammar(rules, lexicon)
496:     lines = open(sentfile).read().splitlines()
497:     sents = [[a.split("/") for a in sent.split()] for sent in lines]
498:     for wordstags in sents:
499:         sent = [a[0] for a in wordstags]
500:         tags = [a[1] for a in wordstags]
501:         stderr.write("parsing: %s\n" % " ".join(sent))
502:         chart, start = parse(sent, grammar, tags, grammar.toid[root], False)
503:         if start:
504:             t, p = mostprobablederivation(chart, start, grammar.tolabel)
505:             ##            print "p=%g\n%s\n\n" % (exp(-p), t)
506:             exp(-p)
507:         else:
508:             pass  # print "no parse\n"
509: 
510: 
511: def demo():
512:     rules = [
513:         ((('S', 'VP2', 'VMFIN'), ((0, 1, 0),)), log(1.0)),
514:         ((('VP2', 'VP2', 'VAINF'), ((0,), (0, 1))), log(0.5)),
515:         ((('VP2', 'PROAV', 'VVPP'), ((0,), (1,))), log(0.5)),
516:         ((('VP2', 'VP2'), ((0,), (0,))), log(0.1))]
517:     lexicon = [
518:         ((('PROAV', 'Epsilon'), 'Darueber'), 0.0),
519:         ((('VAINF', 'Epsilon'), 'werden'), 0.0),
520:         ((('VMFIN', 'Epsilon'), 'muss'), 0.0),
521:         ((('VVPP', 'Epsilon'), 'nachgedacht'), 0.0)]
522:     grammar = splitgrammar(rules, lexicon)
523: 
524:     chart, start = parse("Darueber muss nachgedacht werden".split(),
525:                          grammar, "PROAV VMFIN VVPP VAINF".split(), grammar.toid['S'], False)
526:     pprint_chart(chart, "Darueber muss nachgedacht werden".split(),
527:                  grammar.tolabel)
528:     assert (mostprobablederivation(chart, start, grammar.tolabel) ==
529:             ('(S (VP2 (VP2 (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))', -log(0.25)))
530:     assert do("Darueber muss nachgedacht werden", grammar)
531:     assert do("Darueber muss nachgedacht werden werden", grammar)
532:     assert do("Darueber muss nachgedacht werden werden werden", grammar)
533:     # print "ungrammatical sentence:"
534:     assert not do("werden nachgedacht muss Darueber", grammar)
535:     # print "(as expected)\n"
536: 
537: 
538: def run():
539:     ##    if len(argv) == 4:
540:     ##        batch(argv[1], argv[2], argv[3])
541:     ##    else:
542:     for i in range(100):
543:         demo()
544:     ##        print '''usage: %s grammar lexicon sentences
545:     ##
546:     ##grammar is a tab-separated text file with one rule per line, in this format:
547:     ##
548:     ##LHS	RHS1	RHS2	YIELD-FUNC	LOGPROB
549:     ##e.g., S	NP	VP	[01,10]	0.1
550:     ##
551:     ##LHS, RHS1, and RHS2 are strings specifying the labels of this rule.
552:     ##The yield function is described by a list of bit vectors such as [01,10],
553:     ##where 0 is a variable that refers to a contribution by RHS1, and 1 refers to
554:     ##one by RHS2. Adjacent variables are concatenated, comma-separated components
555:     ##indicate discontinuities.
556:     ##The final element of a rule is its log probability.
557:     ##The LHS of the first rule will be used as the start symbol.
558:     ##
559:     ##lexicon is also tab-separated, in this format:
560:     ##
561:     ##WORD	Epsilon	TAG	LOGPROB
562:     ##e.g., nachgedacht	Epsilon	VVPP	0.1
563:     ##
564:     ##Finally, sentences is a file with one sentence per line, consisting of a space
565:     ##separated list of word/tag pairs, for example:
566:     ##
567:     ##Darueber/PROAV muss/VMFIN nachgedacht/VVPP werden/VAINF
568:     ##
569:     ##The output consists of Viterbi parse trees where terminals have been replaced
570:     ##by indices; this makes it possible to express discontinuities in otherwise
571:     ##context-free trees.''' % argv[0]
572:     return True
573: 
574: 
575: run()
576: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\nA natural language parser for PLCFRS (probabilistic linear context-free\nrewriting systems). PLCFRS is an extension of context-free grammar which\nrewrites tuples of strings instead of strings; this allows it to produce\nparse trees with discontinuous constituents.\n\nCopyright 2011 Andreas van Cranenburgh <andreas@unstable.nl>\nThis program is free software: you can redistribute it and/or modify\nit under the terms of the GNU General Public License as published by\nthe Free Software Foundation, either version 3 of the License, or\n(at your option) any later version.\n\nThis program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\nYou should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from sys import argv, stderr' statement (line 22)
try:
    from sys import argv, stderr

except:
    argv = UndefinedType
    stderr = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'sys', None, module_type_store, ['argv', 'stderr'], [argv, stderr])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from math import exp, log' statement (line 23)
try:
    from math import exp, log

except:
    exp = UndefinedType
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'math', None, module_type_store, ['exp', 'log'], [exp, log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from array import array' statement (line 24)
try:
    from array import array

except:
    array = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'array', None, module_type_store, ['array'], [array])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from heapq import heappush, heappop, heapify' statement (line 25)
try:
    from heapq import heappush, heappop, heapify

except:
    heappush = UndefinedType
    heappop = UndefinedType
    heapify = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'heapq', None, module_type_store, ['heappush', 'heappop', 'heapify'], [heappush, heappop, heapify])


@norecursion
def parse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse'
    module_type_store = module_type_store.open_function_context('parse', 28, 0, False)
    
    # Passed parameters checking function
    parse.stypy_localization = localization
    parse.stypy_type_of_self = None
    parse.stypy_type_store = module_type_store
    parse.stypy_function_name = 'parse'
    parse.stypy_param_names_list = ['sent', 'grammar', 'tags', 'start', 'exhaustive']
    parse.stypy_varargs_param_name = None
    parse.stypy_kwargs_param_name = None
    parse.stypy_call_defaults = defaults
    parse.stypy_call_varargs = varargs
    parse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse', ['sent', 'grammar', 'tags', 'start', 'exhaustive'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse', localization, ['sent', 'grammar', 'tags', 'start', 'exhaustive'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse(...)' code ##################

    str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', ' parse sentence, a list of tokens, optionally with gold tags, and\n    produce a chart, either exhaustive or up until the viterbi parse.\n    ')
    
    # Assigning a Attribute to a Name (line 32):
    
    # Assigning a Attribute to a Name (line 32):
    # Getting the type of 'grammar' (line 32)
    grammar_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'grammar')
    # Obtaining the member 'unary' of a type (line 32)
    unary_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), grammar_27, 'unary')
    # Assigning a type to the variable 'unary' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'unary', unary_28)
    
    # Assigning a Attribute to a Name (line 33):
    
    # Assigning a Attribute to a Name (line 33):
    # Getting the type of 'grammar' (line 33)
    grammar_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'grammar')
    # Obtaining the member 'lbinary' of a type (line 33)
    lbinary_30 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 14), grammar_29, 'lbinary')
    # Assigning a type to the variable 'lbinary' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'lbinary', lbinary_30)
    
    # Assigning a Attribute to a Name (line 34):
    
    # Assigning a Attribute to a Name (line 34):
    # Getting the type of 'grammar' (line 34)
    grammar_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'grammar')
    # Obtaining the member 'rbinary' of a type (line 34)
    rbinary_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 14), grammar_31, 'rbinary')
    # Assigning a type to the variable 'rbinary' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'rbinary', rbinary_32)
    
    # Assigning a Attribute to a Name (line 35):
    
    # Assigning a Attribute to a Name (line 35):
    # Getting the type of 'grammar' (line 35)
    grammar_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'grammar')
    # Obtaining the member 'lexical' of a type (line 35)
    lexical_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), grammar_33, 'lexical')
    # Assigning a type to the variable 'lexical' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'lexical', lexical_34)
    
    # Assigning a Attribute to a Name (line 36):
    
    # Assigning a Attribute to a Name (line 36):
    # Getting the type of 'grammar' (line 36)
    grammar_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'grammar')
    # Obtaining the member 'toid' of a type (line 36)
    toid_36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), grammar_35, 'toid')
    # Assigning a type to the variable 'toid' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'toid', toid_36)
    
    # Assigning a Attribute to a Name (line 37):
    
    # Assigning a Attribute to a Name (line 37):
    # Getting the type of 'grammar' (line 37)
    grammar_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'grammar')
    # Obtaining the member 'tolabel' of a type (line 37)
    tolabel_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 14), grammar_37, 'tolabel')
    # Assigning a type to the variable 'tolabel' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tolabel', tolabel_38)
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to ChartItem(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'start' (line 38)
    start_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'start', False)
    int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 29), 'int')
    
    # Call to len(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'sent' (line 38)
    sent_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 38), 'sent', False)
    # Processing the call keyword arguments (line 38)
    kwargs_44 = {}
    # Getting the type of 'len' (line 38)
    len_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'len', False)
    # Calling len(args, kwargs) (line 38)
    len_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 38, 34), len_42, *[sent_43], **kwargs_44)
    
    # Applying the binary operator '<<' (line 38)
    result_lshift_46 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 29), '<<', int_41, len_call_result_45)
    
    int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 47), 'int')
    # Applying the binary operator '-' (line 38)
    result_sub_48 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 28), '-', result_lshift_46, int_47)
    
    # Processing the call keyword arguments (line 38)
    kwargs_49 = {}
    # Getting the type of 'ChartItem' (line 38)
    ChartItem_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'ChartItem', False)
    # Calling ChartItem(args, kwargs) (line 38)
    ChartItem_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), ChartItem_39, *[start_40, result_sub_48], **kwargs_49)
    
    # Assigning a type to the variable 'goal' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'goal', ChartItem_call_result_50)
    
    # Assigning a Num to a Name (line 39):
    
    # Assigning a Num to a Name (line 39):
    int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'int')
    # Assigning a type to the variable 'maxA' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'maxA', int_51)
    
    # Assigning a Num to a Name (line 40):
    
    # Assigning a Num to a Name (line 40):
    int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'int')
    # Assigning a type to the variable 'blocked' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'blocked', int_52)
    
    # Assigning a ListComp to a Name (line 41):
    
    # Assigning a ListComp to a Name (line 41):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'toid' (line 41)
    toid_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'toid')
    comprehension_55 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), toid_54)
    # Assigning a type to the variable '_' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), '_', comprehension_55)
    
    # Obtaining an instance of the builtin type 'dict' (line 41)
    dict_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 41)
    
    list_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_56, dict_53)
    # Assigning a type to the variable 'Cx' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'Cx', list_56)
    
    # Assigning a Dict to a Name (line 42):
    
    # Assigning a Dict to a Name (line 42):
    
    # Obtaining an instance of the builtin type 'dict' (line 42)
    dict_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 42)
    
    # Assigning a type to the variable 'C' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'C', dict_57)
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to agenda(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_59 = {}
    # Getting the type of 'agenda' (line 43)
    agenda_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'agenda', False)
    # Calling agenda(args, kwargs) (line 43)
    agenda_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), agenda_58, *[], **kwargs_59)
    
    # Assigning a type to the variable 'A' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'A', agenda_call_result_60)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    str_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'str', 'Epsilon')
    # Getting the type of 'toid' (line 46)
    toid_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'toid')
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 14), toid_62, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), getitem___63, str_61)
    
    # Assigning a type to the variable 'Epsilon' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'Epsilon', subscript_call_result_64)
    
    
    # Call to enumerate(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'sent' (line 47)
    sent_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'sent', False)
    # Processing the call keyword arguments (line 47)
    kwargs_67 = {}
    # Getting the type of 'enumerate' (line 47)
    enumerate_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 47)
    enumerate_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), enumerate_65, *[sent_66], **kwargs_67)
    
    # Assigning a type to the variable 'enumerate_call_result_68' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'enumerate_call_result_68', enumerate_call_result_68)
    # Testing if the for loop is going to be iterated (line 47)
    # Testing the type of a for loop iterable (line 47)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 4), enumerate_call_result_68)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 47, 4), enumerate_call_result_68):
        # Getting the type of the for loop variable (line 47)
        for_loop_var_69 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 4), enumerate_call_result_68)
        # Assigning a type to the variable 'i' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), for_loop_var_69))
        # Assigning a type to the variable 'w' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), for_loop_var_69))
        # SSA begins for a for statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 48):
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'False' (line 48)
        False_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'False')
        # Assigning a type to the variable 'recognized' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'recognized', False_70)
        
        
        # Call to get(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'w' (line 49)
        w_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'w', False)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        
        # Processing the call keyword arguments (line 49)
        kwargs_75 = {}
        # Getting the type of 'lexical' (line 49)
        lexical_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'lexical', False)
        # Obtaining the member 'get' of a type (line 49)
        get_72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), lexical_71, 'get')
        # Calling get(args, kwargs) (line 49)
        get_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), get_72, *[w_73, list_74], **kwargs_75)
        
        # Assigning a type to the variable 'get_call_result_76' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'get_call_result_76', get_call_result_76)
        # Testing if the for loop is going to be iterated (line 49)
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), get_call_result_76)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 49, 8), get_call_result_76):
            # Getting the type of the for loop variable (line 49)
            for_loop_var_77 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), get_call_result_76)
            # Assigning a type to the variable 'terminal' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'terminal', for_loop_var_77)
            # SSA begins for a for statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'tags' (line 50)
            tags_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'tags')
            # Applying the 'not' unary operator (line 50)
            result_not__79 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'not', tags_78)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 50)
            i_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'i')
            # Getting the type of 'tags' (line 50)
            tags_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'tags')
            # Obtaining the member '__getitem__' of a type (line 50)
            getitem___82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 27), tags_81, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 50)
            subscript_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 50, 27), getitem___82, i_80)
            
            
            # Obtaining the type of the subscript
            int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 71), 'int')
            
            # Call to split(...): (line 50)
            # Processing the call arguments (line 50)
            str_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 66), 'str', '@')
            # Processing the call keyword arguments (line 50)
            kwargs_92 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'terminal' (line 50)
            terminal_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 46), 'terminal', False)
            # Obtaining the member 'lhs' of a type (line 50)
            lhs_86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 46), terminal_85, 'lhs')
            # Getting the type of 'tolabel' (line 50)
            tolabel_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'tolabel', False)
            # Obtaining the member '__getitem__' of a type (line 50)
            getitem___88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 38), tolabel_87, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 50)
            subscript_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 50, 38), getitem___88, lhs_86)
            
            # Obtaining the member 'split' of a type (line 50)
            split_90 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 38), subscript_call_result_89, 'split')
            # Calling split(args, kwargs) (line 50)
            split_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 50, 38), split_90, *[str_91], **kwargs_92)
            
            # Obtaining the member '__getitem__' of a type (line 50)
            getitem___94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 38), split_call_result_93, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 50)
            subscript_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 50, 38), getitem___94, int_84)
            
            # Applying the binary operator '==' (line 50)
            result_eq_96 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 27), '==', subscript_call_result_83, subscript_call_result_95)
            
            # Applying the binary operator 'or' (line 50)
            result_or_keyword_97 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'or', result_not__79, result_eq_96)
            
            # Testing if the type of an if condition is none (line 50)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 50, 12), result_or_keyword_97):
                pass
            else:
                
                # Testing the type of an if condition (line 50)
                if_condition_98 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), result_or_keyword_97)
                # Assigning a type to the variable 'if_condition_98' (line 50)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_condition_98', if_condition_98)
                # SSA begins for if statement (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 51):
                
                # Assigning a Call to a Name (line 51):
                
                # Call to ChartItem(...): (line 51)
                # Processing the call arguments (line 51)
                # Getting the type of 'terminal' (line 51)
                terminal_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'terminal', False)
                # Obtaining the member 'lhs' of a type (line 51)
                lhs_101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 33), terminal_100, 'lhs')
                int_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'int')
                # Getting the type of 'i' (line 51)
                i_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 52), 'i', False)
                # Applying the binary operator '<<' (line 51)
                result_lshift_104 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 47), '<<', int_102, i_103)
                
                # Processing the call keyword arguments (line 51)
                kwargs_105 = {}
                # Getting the type of 'ChartItem' (line 51)
                ChartItem_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'ChartItem', False)
                # Calling ChartItem(args, kwargs) (line 51)
                ChartItem_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 51, 23), ChartItem_99, *[lhs_101, result_lshift_104], **kwargs_105)
                
                # Assigning a type to the variable 'item' (line 51)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'item', ChartItem_call_result_106)
                
                # Assigning a Call to a Name (line 52):
                
                # Assigning a Call to a Name (line 52):
                
                # Call to ChartItem(...): (line 52)
                # Processing the call arguments (line 52)
                # Getting the type of 'Epsilon' (line 52)
                Epsilon_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 30), 'Epsilon', False)
                # Getting the type of 'i' (line 52)
                i_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'i', False)
                # Processing the call keyword arguments (line 52)
                kwargs_110 = {}
                # Getting the type of 'ChartItem' (line 52)
                ChartItem_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'ChartItem', False)
                # Calling ChartItem(args, kwargs) (line 52)
                ChartItem_call_result_111 = invoke(stypy.reporting.localization.Localization(__file__, 52, 20), ChartItem_107, *[Epsilon_108, i_109], **kwargs_110)
                
                # Assigning a type to the variable 'I' (line 52)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'I', ChartItem_call_result_111)
                
                # Assigning a Attribute to a Name (line 53):
                
                # Assigning a Attribute to a Name (line 53):
                # Getting the type of 'terminal' (line 53)
                terminal_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'terminal')
                # Obtaining the member 'prob' of a type (line 53)
                prob_113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), terminal_112, 'prob')
                # Assigning a type to the variable 'z' (line 53)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'z', prob_113)
                
                # Assigning a Call to a Subscript (line 54):
                
                # Assigning a Call to a Subscript (line 54):
                
                # Call to Edge(...): (line 54)
                # Processing the call arguments (line 54)
                # Getting the type of 'z' (line 54)
                z_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'z', False)
                # Getting the type of 'z' (line 54)
                z_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'z', False)
                # Getting the type of 'z' (line 54)
                z_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'z', False)
                # Getting the type of 'I' (line 54)
                I_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'I', False)
                # Getting the type of 'None' (line 54)
                None_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'None', False)
                # Processing the call keyword arguments (line 54)
                kwargs_120 = {}
                # Getting the type of 'Edge' (line 54)
                Edge_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'Edge', False)
                # Calling Edge(args, kwargs) (line 54)
                Edge_call_result_121 = invoke(stypy.reporting.localization.Localization(__file__, 54, 26), Edge_114, *[z_115, z_116, z_117, I_118, None_119], **kwargs_120)
                
                # Getting the type of 'A' (line 54)
                A_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'A')
                # Getting the type of 'item' (line 54)
                item_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'item')
                # Storing an element on a container (line 54)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 16), A_122, (item_123, Edge_call_result_121))
                
                # Assigning a List to a Subscript (line 55):
                
                # Assigning a List to a Subscript (line 55):
                
                # Obtaining an instance of the builtin type 'list' (line 55)
                list_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'list')
                # Adding type elements to the builtin type 'list' instance (line 55)
                
                # Getting the type of 'C' (line 55)
                C_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'C')
                # Getting the type of 'item' (line 55)
                item_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'item')
                # Storing an element on a container (line 55)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 16), C_125, (item_126, list_124))
                
                # Assigning a Name to a Name (line 56):
                
                # Assigning a Name to a Name (line 56):
                # Getting the type of 'True' (line 56)
                True_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'True')
                # Assigning a type to the variable 'recognized' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'recognized', True_127)
                # SSA join for if statement (line 50)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'recognized' (line 57)
        recognized_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'recognized')
        # Applying the 'not' unary operator (line 57)
        result_not__129 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), 'not', recognized_128)
        
        # Getting the type of 'tags' (line 57)
        tags_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'tags')
        # Applying the binary operator 'and' (line 57)
        result_and_keyword_131 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), 'and', result_not__129, tags_130)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 57)
        i_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 44), 'i')
        # Getting the type of 'tags' (line 57)
        tags_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 39), 'tags')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 39), tags_133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_135 = invoke(stypy.reporting.localization.Localization(__file__, 57, 39), getitem___134, i_132)
        
        # Getting the type of 'toid' (line 57)
        toid_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 50), 'toid')
        # Applying the binary operator 'in' (line 57)
        result_contains_137 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 39), 'in', subscript_call_result_135, toid_136)
        
        # Applying the binary operator 'and' (line 57)
        result_and_keyword_138 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), 'and', result_and_keyword_131, result_contains_137)
        
        # Testing if the type of an if condition is none (line 57)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 8), result_and_keyword_138):
            
            # Getting the type of 'recognized' (line 63)
            recognized_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'recognized')
            # Applying the 'not' unary operator (line 63)
            result_not__173 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 13), 'not', recognized_172)
            
            # Testing if the type of an if condition is none (line 63)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 63, 13), result_not__173):
                pass
            else:
                
                # Testing the type of an if condition (line 63)
                if_condition_174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 13), result_not__173)
                # Assigning a type to the variable 'if_condition_174' (line 63)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'if_condition_174', if_condition_174)
                # SSA begins for if statement (line 63)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 65)
                tuple_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 65)
                # Adding element type (line 65)
                # Getting the type of 'C' (line 65)
                C_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'C')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), tuple_175, C_176)
                # Adding element type (line 65)
                # Getting the type of 'None' (line 65)
                None_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'None')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), tuple_175, None_177)
                
                # Assigning a type to the variable 'stypy_return_type' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', tuple_175)
                # SSA join for if statement (line 63)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 57)
            if_condition_139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), result_and_keyword_138)
            # Assigning a type to the variable 'if_condition_139' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_139', if_condition_139)
            # SSA begins for if statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 58):
            
            # Assigning a Call to a Name (line 58):
            
            # Call to ChartItem(...): (line 58)
            # Processing the call arguments (line 58)
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 58)
            i_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 39), 'i', False)
            # Getting the type of 'tags' (line 58)
            tags_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'tags', False)
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 34), tags_142, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 58, 34), getitem___143, i_141)
            
            # Getting the type of 'toid' (line 58)
            toid_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 29), 'toid', False)
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 29), toid_145, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 58, 29), getitem___146, subscript_call_result_144)
            
            int_148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 44), 'int')
            # Getting the type of 'i' (line 58)
            i_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 49), 'i', False)
            # Applying the binary operator '<<' (line 58)
            result_lshift_150 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 44), '<<', int_148, i_149)
            
            # Processing the call keyword arguments (line 58)
            kwargs_151 = {}
            # Getting the type of 'ChartItem' (line 58)
            ChartItem_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'ChartItem', False)
            # Calling ChartItem(args, kwargs) (line 58)
            ChartItem_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), ChartItem_140, *[subscript_call_result_147, result_lshift_150], **kwargs_151)
            
            # Assigning a type to the variable 'item' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'item', ChartItem_call_result_152)
            
            # Assigning a Call to a Name (line 59):
            
            # Assigning a Call to a Name (line 59):
            
            # Call to ChartItem(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'Epsilon' (line 59)
            Epsilon_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'Epsilon', False)
            # Getting the type of 'i' (line 59)
            i_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 35), 'i', False)
            # Processing the call keyword arguments (line 59)
            kwargs_156 = {}
            # Getting the type of 'ChartItem' (line 59)
            ChartItem_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'ChartItem', False)
            # Calling ChartItem(args, kwargs) (line 59)
            ChartItem_call_result_157 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), ChartItem_153, *[Epsilon_154, i_155], **kwargs_156)
            
            # Assigning a type to the variable 'I' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'I', ChartItem_call_result_157)
            
            # Assigning a Call to a Subscript (line 60):
            
            # Assigning a Call to a Subscript (line 60):
            
            # Call to Edge(...): (line 60)
            # Processing the call arguments (line 60)
            float_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'float')
            float_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'float')
            float_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 37), 'float')
            # Getting the type of 'I' (line 60)
            I_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 42), 'I', False)
            # Getting the type of 'None' (line 60)
            None_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'None', False)
            # Processing the call keyword arguments (line 60)
            kwargs_164 = {}
            # Getting the type of 'Edge' (line 60)
            Edge_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'Edge', False)
            # Calling Edge(args, kwargs) (line 60)
            Edge_call_result_165 = invoke(stypy.reporting.localization.Localization(__file__, 60, 22), Edge_158, *[float_159, float_160, float_161, I_162, None_163], **kwargs_164)
            
            # Getting the type of 'A' (line 60)
            A_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'A')
            # Getting the type of 'item' (line 60)
            item_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'item')
            # Storing an element on a container (line 60)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), A_166, (item_167, Edge_call_result_165))
            
            # Assigning a List to a Subscript (line 61):
            
            # Assigning a List to a Subscript (line 61):
            
            # Obtaining an instance of the builtin type 'list' (line 61)
            list_168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 61)
            
            # Getting the type of 'C' (line 61)
            C_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'C')
            # Getting the type of 'item' (line 61)
            item_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'item')
            # Storing an element on a container (line 61)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 12), C_169, (item_170, list_168))
            
            # Assigning a Name to a Name (line 62):
            
            # Assigning a Name to a Name (line 62):
            # Getting the type of 'True' (line 62)
            True_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'True')
            # Assigning a type to the variable 'recognized' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'recognized', True_171)
            # SSA branch for the else part of an if statement (line 57)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'recognized' (line 63)
            recognized_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'recognized')
            # Applying the 'not' unary operator (line 63)
            result_not__173 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 13), 'not', recognized_172)
            
            # Testing if the type of an if condition is none (line 63)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 63, 13), result_not__173):
                pass
            else:
                
                # Testing the type of an if condition (line 63)
                if_condition_174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 13), result_not__173)
                # Assigning a type to the variable 'if_condition_174' (line 63)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'if_condition_174', if_condition_174)
                # SSA begins for if statement (line 63)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 65)
                tuple_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 65)
                # Adding element type (line 65)
                # Getting the type of 'C' (line 65)
                C_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'C')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), tuple_175, C_176)
                # Adding element type (line 65)
                # Getting the type of 'None' (line 65)
                None_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'None')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), tuple_175, None_177)
                
                # Assigning a type to the variable 'stypy_return_type' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', tuple_175)
                # SSA join for if statement (line 63)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 57)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'A' (line 68)
    A_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 10), 'A')
    # Assigning a type to the variable 'A_178' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'A_178', A_178)
    # Testing if the while is going to be iterated (line 68)
    # Testing the type of an if condition (line 68)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), A_178)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 68, 4), A_178):
        # SSA begins for while statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 69):
        
        # Assigning a Call to a Name:
        
        # Call to popitem(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_181 = {}
        # Getting the type of 'A' (line 69)
        A_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'A', False)
        # Obtaining the member 'popitem' of a type (line 69)
        popitem_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 21), A_179, 'popitem')
        # Calling popitem(args, kwargs) (line 69)
        popitem_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), popitem_180, *[], **kwargs_181)
        
        # Assigning a type to the variable 'call_assignment_1' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'call_assignment_1', popitem_call_result_182)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
        # Processing the call keyword arguments
        kwargs_186 = {}
        # Getting the type of 'call_assignment_1' (line 69)
        call_assignment_1_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), call_assignment_1_183, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___184, *[int_185], **kwargs_186)
        
        # Assigning a type to the variable 'call_assignment_2' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'call_assignment_2', getitem___call_result_187)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'call_assignment_2' (line 69)
        call_assignment_2_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'call_assignment_2')
        # Assigning a type to the variable 'item' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'item', call_assignment_2_188)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
        # Processing the call keyword arguments
        kwargs_192 = {}
        # Getting the type of 'call_assignment_1' (line 69)
        call_assignment_1_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), call_assignment_1_189, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___190, *[int_191], **kwargs_192)
        
        # Assigning a type to the variable 'call_assignment_3' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'call_assignment_3', getitem___call_result_193)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'call_assignment_3' (line 69)
        call_assignment_3_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'call_assignment_3')
        # Assigning a type to the variable 'edge' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'edge', call_assignment_3_194)
        
        # Call to append(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'edge' (line 70)
        edge_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'edge', False)
        # Processing the call keyword arguments (line 70)
        kwargs_201 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 70)
        item_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 10), 'item', False)
        # Getting the type of 'C' (line 70)
        C_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'C', False)
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), C_196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), getitem___197, item_195)
        
        # Obtaining the member 'append' of a type (line 70)
        append_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), subscript_call_result_198, 'append')
        # Calling append(args, kwargs) (line 70)
        append_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), append_199, *[edge_200], **kwargs_201)
        
        
        # Assigning a Name to a Subscript (line 71):
        
        # Assigning a Name to a Subscript (line 71):
        # Getting the type of 'edge' (line 71)
        edge_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 'edge')
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 71)
        item_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'item')
        # Obtaining the member 'label' of a type (line 71)
        label_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), item_204, 'label')
        # Getting the type of 'Cx' (line 71)
        Cx_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'Cx')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), Cx_206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___207, label_205)
        
        # Getting the type of 'item' (line 71)
        item_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'item')
        # Storing an element on a container (line 71)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 8), subscript_call_result_208, (item_209, edge_203))
        
        # Getting the type of 'item' (line 73)
        item_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'item')
        # Getting the type of 'goal' (line 73)
        goal_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'goal')
        # Applying the binary operator '==' (line 73)
        result_eq_212 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '==', item_210, goal_211)
        
        # Testing if the type of an if condition is none (line 73)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_212):
            pass
        else:
            
            # Testing the type of an if condition (line 73)
            if_condition_213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_212)
            # Assigning a type to the variable 'if_condition_213' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_213', if_condition_213)
            # SSA begins for if statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'exhaustive' (line 74)
            exhaustive_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'exhaustive')
            # Testing if the type of an if condition is none (line 74)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 12), exhaustive_214):
                pass
            else:
                
                # Testing the type of an if condition (line 74)
                if_condition_215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 12), exhaustive_214)
                # Assigning a type to the variable 'if_condition_215' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'if_condition_215', if_condition_215)
                # SSA begins for if statement (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA branch for the else part of an if statement (line 74)
                module_type_store.open_ssa_branch('else')
                # SSA join for if statement (line 74)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 78)
        item_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'item')
        # Obtaining the member 'label' of a type (line 78)
        label_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 26), item_216, 'label')
        # Getting the type of 'unary' (line 78)
        unary_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'unary')
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), unary_218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 78, 20), getitem___219, label_217)
        
        # Assigning a type to the variable 'subscript_call_result_220' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'subscript_call_result_220', subscript_call_result_220)
        # Testing if the for loop is going to be iterated (line 78)
        # Testing the type of a for loop iterable (line 78)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 8), subscript_call_result_220)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 8), subscript_call_result_220):
            # Getting the type of the for loop variable (line 78)
            for_loop_var_221 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 8), subscript_call_result_220)
            # Assigning a type to the variable 'rule' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'rule', for_loop_var_221)
            # SSA begins for a for statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'blocked' (line 79)
            blocked_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'blocked')
            
            # Call to process_edge(...): (line 79)
            # Processing the call arguments (line 79)
            
            # Call to ChartItem(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'rule' (line 80)
            rule_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'rule', False)
            # Obtaining the member 'lhs' of a type (line 80)
            lhs_226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 26), rule_225, 'lhs')
            # Getting the type of 'item' (line 80)
            item_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'item', False)
            # Obtaining the member 'vec' of a type (line 80)
            vec_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 36), item_227, 'vec')
            # Processing the call keyword arguments (line 80)
            kwargs_229 = {}
            # Getting the type of 'ChartItem' (line 80)
            ChartItem_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'ChartItem', False)
            # Calling ChartItem(args, kwargs) (line 80)
            ChartItem_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), ChartItem_224, *[lhs_226, vec_228], **kwargs_229)
            
            
            # Call to Edge(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'edge' (line 81)
            edge_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'edge', False)
            # Obtaining the member 'inside' of a type (line 81)
            inside_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 21), edge_232, 'inside')
            # Getting the type of 'rule' (line 81)
            rule_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 35), 'rule', False)
            # Obtaining the member 'prob' of a type (line 81)
            prob_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 35), rule_234, 'prob')
            # Applying the binary operator '+' (line 81)
            result_add_236 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 21), '+', inside_233, prob_235)
            
            # Getting the type of 'edge' (line 81)
            edge_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 'edge', False)
            # Obtaining the member 'inside' of a type (line 81)
            inside_238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 46), edge_237, 'inside')
            # Getting the type of 'rule' (line 81)
            rule_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 60), 'rule', False)
            # Obtaining the member 'prob' of a type (line 81)
            prob_240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 60), rule_239, 'prob')
            # Applying the binary operator '+' (line 81)
            result_add_241 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 46), '+', inside_238, prob_240)
            
            # Getting the type of 'rule' (line 82)
            rule_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'rule', False)
            # Obtaining the member 'prob' of a type (line 82)
            prob_243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), rule_242, 'prob')
            # Getting the type of 'item' (line 82)
            item_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'item', False)
            # Getting the type of 'None' (line 82)
            None_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'None', False)
            # Processing the call keyword arguments (line 81)
            kwargs_246 = {}
            # Getting the type of 'Edge' (line 81)
            Edge_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'Edge', False)
            # Calling Edge(args, kwargs) (line 81)
            Edge_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), Edge_231, *[result_add_236, result_add_241, prob_243, item_244, None_245], **kwargs_246)
            
            # Getting the type of 'A' (line 82)
            A_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 45), 'A', False)
            # Getting the type of 'C' (line 82)
            C_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'C', False)
            # Getting the type of 'exhaustive' (line 82)
            exhaustive_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'exhaustive', False)
            # Processing the call keyword arguments (line 79)
            kwargs_251 = {}
            # Getting the type of 'process_edge' (line 79)
            process_edge_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'process_edge', False)
            # Calling process_edge(args, kwargs) (line 79)
            process_edge_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 79, 23), process_edge_223, *[ChartItem_call_result_230, Edge_call_result_247, A_248, C_249, exhaustive_250], **kwargs_251)
            
            # Applying the binary operator '+=' (line 79)
            result_iadd_253 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 12), '+=', blocked_222, process_edge_call_result_252)
            # Assigning a type to the variable 'blocked' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'blocked', result_iadd_253)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 83)
        item_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'item')
        # Obtaining the member 'label' of a type (line 83)
        label_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 28), item_254, 'label')
        # Getting the type of 'lbinary' (line 83)
        lbinary_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'lbinary')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), lbinary_256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_258 = invoke(stypy.reporting.localization.Localization(__file__, 83, 20), getitem___257, label_255)
        
        # Assigning a type to the variable 'subscript_call_result_258' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'subscript_call_result_258', subscript_call_result_258)
        # Testing if the for loop is going to be iterated (line 83)
        # Testing the type of a for loop iterable (line 83)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 8), subscript_call_result_258)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 83, 8), subscript_call_result_258):
            # Getting the type of the for loop variable (line 83)
            for_loop_var_259 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 8), subscript_call_result_258)
            # Assigning a type to the variable 'rule' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'rule', for_loop_var_259)
            # SSA begins for a for statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'rule' (line 84)
            rule_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'rule')
            # Obtaining the member 'rhs2' of a type (line 84)
            rhs2_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), rule_260, 'rhs2')
            # Getting the type of 'Cx' (line 84)
            Cx_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'Cx')
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), Cx_262, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), getitem___263, rhs2_261)
            
            # Assigning a type to the variable 'subscript_call_result_264' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'subscript_call_result_264', subscript_call_result_264)
            # Testing if the for loop is going to be iterated (line 84)
            # Testing the type of a for loop iterable (line 84)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 12), subscript_call_result_264)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 12), subscript_call_result_264):
                # Getting the type of the for loop variable (line 84)
                for_loop_var_265 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 12), subscript_call_result_264)
                # Assigning a type to the variable 'sibling' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'sibling', for_loop_var_265)
                # SSA begins for a for statement (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 85):
                
                # Assigning a Subscript to a Name (line 85):
                
                # Obtaining the type of the subscript
                # Getting the type of 'sibling' (line 85)
                sibling_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'sibling')
                
                # Obtaining the type of the subscript
                # Getting the type of 'rule' (line 85)
                rule_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'rule')
                # Obtaining the member 'rhs2' of a type (line 85)
                rhs2_268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), rule_267, 'rhs2')
                # Getting the type of 'Cx' (line 85)
                Cx_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'Cx')
                # Obtaining the member '__getitem__' of a type (line 85)
                getitem___270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), Cx_269, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 85)
                subscript_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), getitem___270, rhs2_268)
                
                # Obtaining the member '__getitem__' of a type (line 85)
                getitem___272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), subscript_call_result_271, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 85)
                subscript_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), getitem___272, sibling_266)
                
                # Assigning a type to the variable 'e' (line 85)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'e', subscript_call_result_273)
                
                # Evaluating a boolean operation
                
                # Getting the type of 'item' (line 86)
                item_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'item')
                # Obtaining the member 'vec' of a type (line 86)
                vec_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), item_274, 'vec')
                # Getting the type of 'sibling' (line 86)
                sibling_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 31), 'sibling')
                # Obtaining the member 'vec' of a type (line 86)
                vec_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 31), sibling_276, 'vec')
                # Applying the binary operator '&' (line 86)
                result_and__278 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 20), '&', vec_275, vec_277)
                
                int_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 46), 'int')
                # Applying the binary operator '==' (line 86)
                result_eq_280 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 20), '==', result_and__278, int_279)
                
                
                # Call to concat(...): (line 87)
                # Processing the call arguments (line 87)
                # Getting the type of 'rule' (line 87)
                rule_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'rule', False)
                # Getting the type of 'item' (line 87)
                item_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'item', False)
                # Obtaining the member 'vec' of a type (line 87)
                vec_284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 41), item_283, 'vec')
                # Getting the type of 'sibling' (line 87)
                sibling_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 51), 'sibling', False)
                # Obtaining the member 'vec' of a type (line 87)
                vec_286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 51), sibling_285, 'vec')
                # Processing the call keyword arguments (line 87)
                kwargs_287 = {}
                # Getting the type of 'concat' (line 87)
                concat_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'concat', False)
                # Calling concat(args, kwargs) (line 87)
                concat_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 87, 28), concat_281, *[rule_282, vec_284, vec_286], **kwargs_287)
                
                # Applying the binary operator 'and' (line 86)
                result_and_keyword_289 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 20), 'and', result_eq_280, concat_call_result_288)
                
                # Testing if the type of an if condition is none (line 86)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 16), result_and_keyword_289):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 86)
                    if_condition_290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 16), result_and_keyword_289)
                    # Assigning a type to the variable 'if_condition_290' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'if_condition_290', if_condition_290)
                    # SSA begins for if statement (line 86)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'blocked' (line 88)
                    blocked_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'blocked')
                    
                    # Call to process_edge(...): (line 88)
                    # Processing the call arguments (line 88)
                    
                    # Call to ChartItem(...): (line 89)
                    # Processing the call arguments (line 89)
                    # Getting the type of 'rule' (line 89)
                    rule_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'rule', False)
                    # Obtaining the member 'lhs' of a type (line 89)
                    lhs_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 34), rule_294, 'lhs')
                    # Getting the type of 'item' (line 89)
                    item_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), 'item', False)
                    # Obtaining the member 'vec' of a type (line 89)
                    vec_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 44), item_296, 'vec')
                    # Getting the type of 'sibling' (line 89)
                    sibling_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'sibling', False)
                    # Obtaining the member 'vec' of a type (line 89)
                    vec_299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 55), sibling_298, 'vec')
                    # Applying the binary operator '^' (line 89)
                    result_xor_300 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 44), '^', vec_297, vec_299)
                    
                    # Processing the call keyword arguments (line 89)
                    kwargs_301 = {}
                    # Getting the type of 'ChartItem' (line 89)
                    ChartItem_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'ChartItem', False)
                    # Calling ChartItem(args, kwargs) (line 89)
                    ChartItem_call_result_302 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), ChartItem_293, *[lhs_295, result_xor_300], **kwargs_301)
                    
                    
                    # Call to Edge(...): (line 90)
                    # Processing the call arguments (line 90)
                    # Getting the type of 'edge' (line 90)
                    edge_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'edge', False)
                    # Obtaining the member 'inside' of a type (line 90)
                    inside_305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 29), edge_304, 'inside')
                    # Getting the type of 'e' (line 90)
                    e_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 43), 'e', False)
                    # Obtaining the member 'inside' of a type (line 90)
                    inside_307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 43), e_306, 'inside')
                    # Applying the binary operator '+' (line 90)
                    result_add_308 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 29), '+', inside_305, inside_307)
                    
                    # Getting the type of 'rule' (line 90)
                    rule_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 54), 'rule', False)
                    # Obtaining the member 'prob' of a type (line 90)
                    prob_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 54), rule_309, 'prob')
                    # Applying the binary operator '+' (line 90)
                    result_add_311 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 52), '+', result_add_308, prob_310)
                    
                    # Getting the type of 'edge' (line 91)
                    edge_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'edge', False)
                    # Obtaining the member 'inside' of a type (line 91)
                    inside_313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), edge_312, 'inside')
                    # Getting the type of 'e' (line 91)
                    e_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'e', False)
                    # Obtaining the member 'inside' of a type (line 91)
                    inside_315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 43), e_314, 'inside')
                    # Applying the binary operator '+' (line 91)
                    result_add_316 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '+', inside_313, inside_315)
                    
                    # Getting the type of 'rule' (line 91)
                    rule_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 54), 'rule', False)
                    # Obtaining the member 'prob' of a type (line 91)
                    prob_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 54), rule_317, 'prob')
                    # Applying the binary operator '+' (line 91)
                    result_add_319 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 52), '+', result_add_316, prob_318)
                    
                    # Getting the type of 'rule' (line 92)
                    rule_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'rule', False)
                    # Obtaining the member 'prob' of a type (line 92)
                    prob_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 29), rule_320, 'prob')
                    # Getting the type of 'item' (line 92)
                    item_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 40), 'item', False)
                    # Getting the type of 'sibling' (line 92)
                    sibling_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'sibling', False)
                    # Processing the call keyword arguments (line 90)
                    kwargs_324 = {}
                    # Getting the type of 'Edge' (line 90)
                    Edge_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'Edge', False)
                    # Calling Edge(args, kwargs) (line 90)
                    Edge_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 90, 24), Edge_303, *[result_add_311, result_add_319, prob_321, item_322, sibling_323], **kwargs_324)
                    
                    # Getting the type of 'A' (line 92)
                    A_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 56), 'A', False)
                    # Getting the type of 'C' (line 92)
                    C_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 59), 'C', False)
                    # Getting the type of 'exhaustive' (line 92)
                    exhaustive_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 62), 'exhaustive', False)
                    # Processing the call keyword arguments (line 88)
                    kwargs_329 = {}
                    # Getting the type of 'process_edge' (line 88)
                    process_edge_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'process_edge', False)
                    # Calling process_edge(args, kwargs) (line 88)
                    process_edge_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 88, 31), process_edge_292, *[ChartItem_call_result_302, Edge_call_result_325, A_326, C_327, exhaustive_328], **kwargs_329)
                    
                    # Applying the binary operator '+=' (line 88)
                    result_iadd_331 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 20), '+=', blocked_291, process_edge_call_result_330)
                    # Assigning a type to the variable 'blocked' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'blocked', result_iadd_331)
                    
                    # SSA join for if statement (line 86)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 93)
        item_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'item')
        # Obtaining the member 'label' of a type (line 93)
        label_333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), item_332, 'label')
        # Getting the type of 'rbinary' (line 93)
        rbinary_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'rbinary')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 20), rbinary_334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 93, 20), getitem___335, label_333)
        
        # Assigning a type to the variable 'subscript_call_result_336' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'subscript_call_result_336', subscript_call_result_336)
        # Testing if the for loop is going to be iterated (line 93)
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), subscript_call_result_336)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 93, 8), subscript_call_result_336):
            # Getting the type of the for loop variable (line 93)
            for_loop_var_337 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), subscript_call_result_336)
            # Assigning a type to the variable 'rule' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'rule', for_loop_var_337)
            # SSA begins for a for statement (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'rule' (line 94)
            rule_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), 'rule')
            # Obtaining the member 'rhs1' of a type (line 94)
            rhs1_339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 30), rule_338, 'rhs1')
            # Getting the type of 'Cx' (line 94)
            Cx_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'Cx')
            # Obtaining the member '__getitem__' of a type (line 94)
            getitem___341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 27), Cx_340, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 94)
            subscript_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 94, 27), getitem___341, rhs1_339)
            
            # Assigning a type to the variable 'subscript_call_result_342' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'subscript_call_result_342', subscript_call_result_342)
            # Testing if the for loop is going to be iterated (line 94)
            # Testing the type of a for loop iterable (line 94)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 94, 12), subscript_call_result_342)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 94, 12), subscript_call_result_342):
                # Getting the type of the for loop variable (line 94)
                for_loop_var_343 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 94, 12), subscript_call_result_342)
                # Assigning a type to the variable 'sibling' (line 94)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'sibling', for_loop_var_343)
                # SSA begins for a for statement (line 94)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 95):
                
                # Assigning a Subscript to a Name (line 95):
                
                # Obtaining the type of the subscript
                # Getting the type of 'sibling' (line 95)
                sibling_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'sibling')
                
                # Obtaining the type of the subscript
                # Getting the type of 'rule' (line 95)
                rule_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'rule')
                # Obtaining the member 'rhs1' of a type (line 95)
                rhs1_346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), rule_345, 'rhs1')
                # Getting the type of 'Cx' (line 95)
                Cx_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'Cx')
                # Obtaining the member '__getitem__' of a type (line 95)
                getitem___348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 20), Cx_347, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 95)
                subscript_call_result_349 = invoke(stypy.reporting.localization.Localization(__file__, 95, 20), getitem___348, rhs1_346)
                
                # Obtaining the member '__getitem__' of a type (line 95)
                getitem___350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 20), subscript_call_result_349, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 95)
                subscript_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 95, 20), getitem___350, sibling_344)
                
                # Assigning a type to the variable 'e' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'e', subscript_call_result_351)
                
                # Evaluating a boolean operation
                
                # Getting the type of 'sibling' (line 96)
                sibling_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'sibling')
                # Obtaining the member 'vec' of a type (line 96)
                vec_353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), sibling_352, 'vec')
                # Getting the type of 'item' (line 96)
                item_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'item')
                # Obtaining the member 'vec' of a type (line 96)
                vec_355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 34), item_354, 'vec')
                # Applying the binary operator '&' (line 96)
                result_and__356 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 20), '&', vec_353, vec_355)
                
                int_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 46), 'int')
                # Applying the binary operator '==' (line 96)
                result_eq_358 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 20), '==', result_and__356, int_357)
                
                
                # Call to concat(...): (line 97)
                # Processing the call arguments (line 97)
                # Getting the type of 'rule' (line 97)
                rule_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 35), 'rule', False)
                # Getting the type of 'sibling' (line 97)
                sibling_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 41), 'sibling', False)
                # Obtaining the member 'vec' of a type (line 97)
                vec_362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 41), sibling_361, 'vec')
                # Getting the type of 'item' (line 97)
                item_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 54), 'item', False)
                # Obtaining the member 'vec' of a type (line 97)
                vec_364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 54), item_363, 'vec')
                # Processing the call keyword arguments (line 97)
                kwargs_365 = {}
                # Getting the type of 'concat' (line 97)
                concat_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'concat', False)
                # Calling concat(args, kwargs) (line 97)
                concat_call_result_366 = invoke(stypy.reporting.localization.Localization(__file__, 97, 28), concat_359, *[rule_360, vec_362, vec_364], **kwargs_365)
                
                # Applying the binary operator 'and' (line 96)
                result_and_keyword_367 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 20), 'and', result_eq_358, concat_call_result_366)
                
                # Testing if the type of an if condition is none (line 96)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 16), result_and_keyword_367):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 96)
                    if_condition_368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 16), result_and_keyword_367)
                    # Assigning a type to the variable 'if_condition_368' (line 96)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'if_condition_368', if_condition_368)
                    # SSA begins for if statement (line 96)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'blocked' (line 98)
                    blocked_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'blocked')
                    
                    # Call to process_edge(...): (line 98)
                    # Processing the call arguments (line 98)
                    
                    # Call to ChartItem(...): (line 99)
                    # Processing the call arguments (line 99)
                    # Getting the type of 'rule' (line 99)
                    rule_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'rule', False)
                    # Obtaining the member 'lhs' of a type (line 99)
                    lhs_373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 34), rule_372, 'lhs')
                    # Getting the type of 'sibling' (line 99)
                    sibling_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 44), 'sibling', False)
                    # Obtaining the member 'vec' of a type (line 99)
                    vec_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 44), sibling_374, 'vec')
                    # Getting the type of 'item' (line 99)
                    item_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 58), 'item', False)
                    # Obtaining the member 'vec' of a type (line 99)
                    vec_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 58), item_376, 'vec')
                    # Applying the binary operator '^' (line 99)
                    result_xor_378 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 44), '^', vec_375, vec_377)
                    
                    # Processing the call keyword arguments (line 99)
                    kwargs_379 = {}
                    # Getting the type of 'ChartItem' (line 99)
                    ChartItem_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'ChartItem', False)
                    # Calling ChartItem(args, kwargs) (line 99)
                    ChartItem_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 99, 24), ChartItem_371, *[lhs_373, result_xor_378], **kwargs_379)
                    
                    
                    # Call to Edge(...): (line 100)
                    # Processing the call arguments (line 100)
                    # Getting the type of 'e' (line 100)
                    e_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'e', False)
                    # Obtaining the member 'inside' of a type (line 100)
                    inside_383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 29), e_382, 'inside')
                    # Getting the type of 'edge' (line 100)
                    edge_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'edge', False)
                    # Obtaining the member 'inside' of a type (line 100)
                    inside_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 40), edge_384, 'inside')
                    # Applying the binary operator '+' (line 100)
                    result_add_386 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 29), '+', inside_383, inside_385)
                    
                    # Getting the type of 'rule' (line 100)
                    rule_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 54), 'rule', False)
                    # Obtaining the member 'prob' of a type (line 100)
                    prob_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 54), rule_387, 'prob')
                    # Applying the binary operator '+' (line 100)
                    result_add_389 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 52), '+', result_add_386, prob_388)
                    
                    # Getting the type of 'e' (line 101)
                    e_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'e', False)
                    # Obtaining the member 'inside' of a type (line 101)
                    inside_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 29), e_390, 'inside')
                    # Getting the type of 'edge' (line 101)
                    edge_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'edge', False)
                    # Obtaining the member 'inside' of a type (line 101)
                    inside_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 40), edge_392, 'inside')
                    # Applying the binary operator '+' (line 101)
                    result_add_394 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 29), '+', inside_391, inside_393)
                    
                    # Getting the type of 'rule' (line 101)
                    rule_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 54), 'rule', False)
                    # Obtaining the member 'prob' of a type (line 101)
                    prob_396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 54), rule_395, 'prob')
                    # Applying the binary operator '+' (line 101)
                    result_add_397 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 52), '+', result_add_394, prob_396)
                    
                    # Getting the type of 'rule' (line 102)
                    rule_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'rule', False)
                    # Obtaining the member 'prob' of a type (line 102)
                    prob_399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 29), rule_398, 'prob')
                    # Getting the type of 'sibling' (line 102)
                    sibling_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'sibling', False)
                    # Getting the type of 'item' (line 102)
                    item_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'item', False)
                    # Processing the call keyword arguments (line 100)
                    kwargs_402 = {}
                    # Getting the type of 'Edge' (line 100)
                    Edge_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'Edge', False)
                    # Calling Edge(args, kwargs) (line 100)
                    Edge_call_result_403 = invoke(stypy.reporting.localization.Localization(__file__, 100, 24), Edge_381, *[result_add_389, result_add_397, prob_399, sibling_400, item_401], **kwargs_402)
                    
                    # Getting the type of 'A' (line 102)
                    A_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 56), 'A', False)
                    # Getting the type of 'C' (line 102)
                    C_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 59), 'C', False)
                    # Getting the type of 'exhaustive' (line 102)
                    exhaustive_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 62), 'exhaustive', False)
                    # Processing the call keyword arguments (line 98)
                    kwargs_407 = {}
                    # Getting the type of 'process_edge' (line 98)
                    process_edge_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'process_edge', False)
                    # Calling process_edge(args, kwargs) (line 98)
                    process_edge_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 98, 31), process_edge_370, *[ChartItem_call_result_380, Edge_call_result_403, A_404, C_405, exhaustive_406], **kwargs_407)
                    
                    # Applying the binary operator '+=' (line 98)
                    result_iadd_409 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 20), '+=', blocked_369, process_edge_call_result_408)
                    # Assigning a type to the variable 'blocked' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'blocked', result_iadd_409)
                    
                    # SSA join for if statement (line 96)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'A' (line 103)
        A_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'A', False)
        # Processing the call keyword arguments (line 103)
        kwargs_412 = {}
        # Getting the type of 'len' (line 103)
        len_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'len', False)
        # Calling len(args, kwargs) (line 103)
        len_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), len_410, *[A_411], **kwargs_412)
        
        # Getting the type of 'maxA' (line 103)
        maxA_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'maxA')
        # Applying the binary operator '>' (line 103)
        result_gt_415 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), '>', len_call_result_413, maxA_414)
        
        # Testing if the type of an if condition is none (line 103)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 103, 8), result_gt_415):
            pass
        else:
            
            # Testing the type of an if condition (line 103)
            if_condition_416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_gt_415)
            # Assigning a type to the variable 'if_condition_416' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_416', if_condition_416)
            # SSA begins for if statement (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 103):
            
            # Assigning a Call to a Name (line 103):
            
            # Call to len(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'A' (line 103)
            A_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'A', False)
            # Processing the call keyword arguments (line 103)
            kwargs_419 = {}
            # Getting the type of 'len' (line 103)
            len_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'len', False)
            # Calling len(args, kwargs) (line 103)
            len_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 103, 33), len_417, *[A_418], **kwargs_419)
            
            # Assigning a type to the variable 'maxA' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'maxA', len_call_result_420)
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for while statement (line 68)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'goal' (line 110)
    goal_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'goal')
    # Getting the type of 'C' (line 110)
    C_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'C')
    # Applying the binary operator 'notin' (line 110)
    result_contains_423 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), 'notin', goal_421, C_422)
    
    # Testing if the type of an if condition is none (line 110)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 110, 4), result_contains_423):
        pass
    else:
        
        # Testing the type of an if condition (line 110)
        if_condition_424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_contains_423)
        # Assigning a type to the variable 'if_condition_424' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_424', if_condition_424)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 110):
        
        # Assigning a Name to a Name (line 110):
        # Getting the type of 'None' (line 110)
        None_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'None')
        # Assigning a type to the variable 'goal' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'goal', None_425)
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 111)
    tuple_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 111)
    # Adding element type (line 111)
    # Getting the type of 'C' (line 111)
    C_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 12), tuple_426, C_427)
    # Adding element type (line 111)
    # Getting the type of 'goal' (line 111)
    goal_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'goal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 12), tuple_426, goal_428)
    
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', tuple_426)
    
    # ################# End of 'parse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_429)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse'
    return stypy_return_type_429

# Assigning a type to the variable 'parse' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'parse', parse)

@norecursion
def process_edge(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'process_edge'
    module_type_store = module_type_store.open_function_context('process_edge', 114, 0, False)
    
    # Passed parameters checking function
    process_edge.stypy_localization = localization
    process_edge.stypy_type_of_self = None
    process_edge.stypy_type_store = module_type_store
    process_edge.stypy_function_name = 'process_edge'
    process_edge.stypy_param_names_list = ['newitem', 'newedge', 'A', 'C', 'exhaustive']
    process_edge.stypy_varargs_param_name = None
    process_edge.stypy_kwargs_param_name = None
    process_edge.stypy_call_defaults = defaults
    process_edge.stypy_call_varargs = varargs
    process_edge.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process_edge', ['newitem', 'newedge', 'A', 'C', 'exhaustive'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process_edge', localization, ['newitem', 'newedge', 'A', 'C', 'exhaustive'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process_edge(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Getting the type of 'newitem' (line 115)
    newitem_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 7), 'newitem')
    # Getting the type of 'C' (line 115)
    C_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'C')
    # Applying the binary operator 'notin' (line 115)
    result_contains_432 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 7), 'notin', newitem_430, C_431)
    
    
    # Getting the type of 'newitem' (line 115)
    newitem_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'newitem')
    # Getting the type of 'A' (line 115)
    A_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 43), 'A')
    # Applying the binary operator 'notin' (line 115)
    result_contains_435 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 28), 'notin', newitem_433, A_434)
    
    # Applying the binary operator 'and' (line 115)
    result_and_keyword_436 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 7), 'and', result_contains_432, result_contains_435)
    
    # Testing if the type of an if condition is none (line 115)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 115, 4), result_and_keyword_436):
        
        # Evaluating a boolean operation
        
        # Getting the type of 'newitem' (line 121)
        newitem_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 9), 'newitem')
        # Getting the type of 'A' (line 121)
        A_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'A')
        # Applying the binary operator 'in' (line 121)
        result_contains_452 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 9), 'in', newitem_450, A_451)
        
        
        # Getting the type of 'newedge' (line 121)
        newedge_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'newedge')
        # Obtaining the member 'inside' of a type (line 121)
        inside_454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 26), newedge_453, 'inside')
        
        # Obtaining the type of the subscript
        # Getting the type of 'newitem' (line 121)
        newitem_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 45), 'newitem')
        # Getting the type of 'A' (line 121)
        A_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 43), 'A')
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 43), A_456, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 121, 43), getitem___457, newitem_455)
        
        # Obtaining the member 'inside' of a type (line 121)
        inside_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 43), subscript_call_result_458, 'inside')
        # Applying the binary operator '<' (line 121)
        result_lt_460 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 26), '<', inside_454, inside_459)
        
        # Applying the binary operator 'and' (line 121)
        result_and_keyword_461 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 9), 'and', result_contains_452, result_lt_460)
        
        # Testing if the type of an if condition is none (line 121)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 9), result_and_keyword_461):
            # Getting the type of 'exhaustive' (line 125)
            exhaustive_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'exhaustive')
            # Testing if the type of an if condition is none (line 125)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477):
                pass
            else:
                
                # Testing the type of an if condition (line 125)
                if_condition_478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477)
                # Assigning a type to the variable 'if_condition_478' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'if_condition_478', if_condition_478)
                # SSA begins for if statement (line 125)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 127)
                # Processing the call arguments (line 127)
                # Getting the type of 'newedge' (line 127)
                newedge_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'newedge', False)
                # Processing the call keyword arguments (line 127)
                kwargs_485 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'newitem' (line 127)
                newitem_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 10), 'newitem', False)
                # Getting the type of 'C' (line 127)
                C_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'C', False)
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), C_480, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), getitem___481, newitem_479)
                
                # Obtaining the member 'append' of a type (line 127)
                append_483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), subscript_call_result_482, 'append')
                # Calling append(args, kwargs) (line 127)
                append_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), append_483, *[newedge_484], **kwargs_485)
                
                # SSA join for if statement (line 125)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 121)
            if_condition_462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 9), result_and_keyword_461)
            # Assigning a type to the variable 'if_condition_462' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 9), 'if_condition_462', if_condition_462)
            # SSA begins for if statement (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 123)
            # Processing the call arguments (line 123)
            
            # Obtaining the type of the subscript
            # Getting the type of 'newitem' (line 123)
            newitem_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'newitem', False)
            # Getting the type of 'A' (line 123)
            A_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'A', False)
            # Obtaining the member '__getitem__' of a type (line 123)
            getitem___470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 26), A_469, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
            subscript_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 123, 26), getitem___470, newitem_468)
            
            # Processing the call keyword arguments (line 123)
            kwargs_472 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'newitem' (line 123)
            newitem_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), 'newitem', False)
            # Getting the type of 'C' (line 123)
            C_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'C', False)
            # Obtaining the member '__getitem__' of a type (line 123)
            getitem___465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), C_464, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
            subscript_call_result_466 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___465, newitem_463)
            
            # Obtaining the member 'append' of a type (line 123)
            append_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), subscript_call_result_466, 'append')
            # Calling append(args, kwargs) (line 123)
            append_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), append_467, *[subscript_call_result_471], **kwargs_472)
            
            
            # Assigning a Name to a Subscript (line 124):
            
            # Assigning a Name to a Subscript (line 124):
            # Getting the type of 'newedge' (line 124)
            newedge_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'newedge')
            # Getting the type of 'A' (line 124)
            A_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'A')
            # Getting the type of 'newitem' (line 124)
            newitem_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'newitem')
            # Storing an element on a container (line 124)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), A_475, (newitem_476, newedge_474))
            # SSA branch for the else part of an if statement (line 121)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'exhaustive' (line 125)
            exhaustive_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'exhaustive')
            # Testing if the type of an if condition is none (line 125)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477):
                pass
            else:
                
                # Testing the type of an if condition (line 125)
                if_condition_478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477)
                # Assigning a type to the variable 'if_condition_478' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'if_condition_478', if_condition_478)
                # SSA begins for if statement (line 125)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 127)
                # Processing the call arguments (line 127)
                # Getting the type of 'newedge' (line 127)
                newedge_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'newedge', False)
                # Processing the call keyword arguments (line 127)
                kwargs_485 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'newitem' (line 127)
                newitem_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 10), 'newitem', False)
                # Getting the type of 'C' (line 127)
                C_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'C', False)
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), C_480, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), getitem___481, newitem_479)
                
                # Obtaining the member 'append' of a type (line 127)
                append_483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), subscript_call_result_482, 'append')
                # Calling append(args, kwargs) (line 127)
                append_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), append_483, *[newedge_484], **kwargs_485)
                
                # SSA join for if statement (line 125)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 121)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 115)
        if_condition_437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 4), result_and_keyword_436)
        # Assigning a type to the variable 'if_condition_437' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'if_condition_437', if_condition_437)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'newedge' (line 117)
        newedge_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'newedge')
        # Obtaining the member 'score' of a type (line 117)
        score_439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), newedge_438, 'score')
        float_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'float')
        # Applying the binary operator '>' (line 117)
        result_gt_441 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 11), '>', score_439, float_440)
        
        # Testing if the type of an if condition is none (line 117)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 117, 8), result_gt_441):
            pass
        else:
            
            # Testing the type of an if condition (line 117)
            if_condition_442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), result_gt_441)
            # Assigning a type to the variable 'if_condition_442' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'if_condition_442', if_condition_442)
            # SSA begins for if statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 41), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 34), 'stypy_return_type', int_443)
            # SSA join for if statement (line 117)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Subscript (line 119):
        
        # Assigning a Name to a Subscript (line 119):
        # Getting the type of 'newedge' (line 119)
        newedge_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'newedge')
        # Getting the type of 'A' (line 119)
        A_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'A')
        # Getting the type of 'newitem' (line 119)
        newitem_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 10), 'newitem')
        # Storing an element on a container (line 119)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 8), A_445, (newitem_446, newedge_444))
        
        # Assigning a List to a Subscript (line 120):
        
        # Assigning a List to a Subscript (line 120):
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        
        # Getting the type of 'C' (line 120)
        C_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'C')
        # Getting the type of 'newitem' (line 120)
        newitem_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 10), 'newitem')
        # Storing an element on a container (line 120)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), C_448, (newitem_449, list_447))
        # SSA branch for the else part of an if statement (line 115)
        module_type_store.open_ssa_branch('else')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'newitem' (line 121)
        newitem_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 9), 'newitem')
        # Getting the type of 'A' (line 121)
        A_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'A')
        # Applying the binary operator 'in' (line 121)
        result_contains_452 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 9), 'in', newitem_450, A_451)
        
        
        # Getting the type of 'newedge' (line 121)
        newedge_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'newedge')
        # Obtaining the member 'inside' of a type (line 121)
        inside_454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 26), newedge_453, 'inside')
        
        # Obtaining the type of the subscript
        # Getting the type of 'newitem' (line 121)
        newitem_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 45), 'newitem')
        # Getting the type of 'A' (line 121)
        A_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 43), 'A')
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 43), A_456, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 121, 43), getitem___457, newitem_455)
        
        # Obtaining the member 'inside' of a type (line 121)
        inside_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 43), subscript_call_result_458, 'inside')
        # Applying the binary operator '<' (line 121)
        result_lt_460 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 26), '<', inside_454, inside_459)
        
        # Applying the binary operator 'and' (line 121)
        result_and_keyword_461 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 9), 'and', result_contains_452, result_lt_460)
        
        # Testing if the type of an if condition is none (line 121)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 9), result_and_keyword_461):
            # Getting the type of 'exhaustive' (line 125)
            exhaustive_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'exhaustive')
            # Testing if the type of an if condition is none (line 125)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477):
                pass
            else:
                
                # Testing the type of an if condition (line 125)
                if_condition_478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477)
                # Assigning a type to the variable 'if_condition_478' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'if_condition_478', if_condition_478)
                # SSA begins for if statement (line 125)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 127)
                # Processing the call arguments (line 127)
                # Getting the type of 'newedge' (line 127)
                newedge_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'newedge', False)
                # Processing the call keyword arguments (line 127)
                kwargs_485 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'newitem' (line 127)
                newitem_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 10), 'newitem', False)
                # Getting the type of 'C' (line 127)
                C_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'C', False)
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), C_480, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), getitem___481, newitem_479)
                
                # Obtaining the member 'append' of a type (line 127)
                append_483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), subscript_call_result_482, 'append')
                # Calling append(args, kwargs) (line 127)
                append_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), append_483, *[newedge_484], **kwargs_485)
                
                # SSA join for if statement (line 125)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 121)
            if_condition_462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 9), result_and_keyword_461)
            # Assigning a type to the variable 'if_condition_462' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 9), 'if_condition_462', if_condition_462)
            # SSA begins for if statement (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 123)
            # Processing the call arguments (line 123)
            
            # Obtaining the type of the subscript
            # Getting the type of 'newitem' (line 123)
            newitem_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'newitem', False)
            # Getting the type of 'A' (line 123)
            A_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'A', False)
            # Obtaining the member '__getitem__' of a type (line 123)
            getitem___470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 26), A_469, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
            subscript_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 123, 26), getitem___470, newitem_468)
            
            # Processing the call keyword arguments (line 123)
            kwargs_472 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'newitem' (line 123)
            newitem_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), 'newitem', False)
            # Getting the type of 'C' (line 123)
            C_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'C', False)
            # Obtaining the member '__getitem__' of a type (line 123)
            getitem___465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), C_464, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
            subscript_call_result_466 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___465, newitem_463)
            
            # Obtaining the member 'append' of a type (line 123)
            append_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), subscript_call_result_466, 'append')
            # Calling append(args, kwargs) (line 123)
            append_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), append_467, *[subscript_call_result_471], **kwargs_472)
            
            
            # Assigning a Name to a Subscript (line 124):
            
            # Assigning a Name to a Subscript (line 124):
            # Getting the type of 'newedge' (line 124)
            newedge_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'newedge')
            # Getting the type of 'A' (line 124)
            A_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'A')
            # Getting the type of 'newitem' (line 124)
            newitem_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'newitem')
            # Storing an element on a container (line 124)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), A_475, (newitem_476, newedge_474))
            # SSA branch for the else part of an if statement (line 121)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'exhaustive' (line 125)
            exhaustive_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'exhaustive')
            # Testing if the type of an if condition is none (line 125)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477):
                pass
            else:
                
                # Testing the type of an if condition (line 125)
                if_condition_478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 9), exhaustive_477)
                # Assigning a type to the variable 'if_condition_478' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'if_condition_478', if_condition_478)
                # SSA begins for if statement (line 125)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 127)
                # Processing the call arguments (line 127)
                # Getting the type of 'newedge' (line 127)
                newedge_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'newedge', False)
                # Processing the call keyword arguments (line 127)
                kwargs_485 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'newitem' (line 127)
                newitem_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 10), 'newitem', False)
                # Getting the type of 'C' (line 127)
                C_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'C', False)
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), C_480, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), getitem___481, newitem_479)
                
                # Obtaining the member 'append' of a type (line 127)
                append_483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), subscript_call_result_482, 'append')
                # Calling append(args, kwargs) (line 127)
                append_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), append_483, *[newedge_484], **kwargs_485)
                
                # SSA join for if statement (line 125)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 121)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        

    int_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', int_487)
    
    # ################# End of 'process_edge(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process_edge' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_488)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process_edge'
    return stypy_return_type_488

# Assigning a type to the variable 'process_edge' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'process_edge', process_edge)

@norecursion
def concat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'concat'
    module_type_store = module_type_store.open_function_context('concat', 131, 0, False)
    
    # Passed parameters checking function
    concat.stypy_localization = localization
    concat.stypy_type_of_self = None
    concat.stypy_type_store = module_type_store
    concat.stypy_function_name = 'concat'
    concat.stypy_param_names_list = ['rule', 'lvec', 'rvec']
    concat.stypy_varargs_param_name = None
    concat.stypy_kwargs_param_name = None
    concat.stypy_call_defaults = defaults
    concat.stypy_call_varargs = varargs
    concat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'concat', ['rule', 'lvec', 'rvec'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'concat', localization, ['rule', 'lvec', 'rvec'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'concat(...)' code ##################

    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to nextset(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'lvec' (line 132)
    lvec_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'lvec', False)
    int_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'int')
    # Processing the call keyword arguments (line 132)
    kwargs_492 = {}
    # Getting the type of 'nextset' (line 132)
    nextset_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'nextset', False)
    # Calling nextset(args, kwargs) (line 132)
    nextset_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 132, 11), nextset_489, *[lvec_490, int_491], **kwargs_492)
    
    # Assigning a type to the variable 'lpos' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'lpos', nextset_call_result_493)
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to nextset(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'rvec' (line 133)
    rvec_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'rvec', False)
    int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
    # Processing the call keyword arguments (line 133)
    kwargs_497 = {}
    # Getting the type of 'nextset' (line 133)
    nextset_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'nextset', False)
    # Calling nextset(args, kwargs) (line 133)
    nextset_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), nextset_494, *[rvec_495, int_496], **kwargs_497)
    
    # Assigning a type to the variable 'rpos' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'rpos', nextset_call_result_498)
    
    
    # Call to range(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'rule' (line 135)
    rule_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'rule', False)
    # Obtaining the member 'args' of a type (line 135)
    args_502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 23), rule_501, 'args')
    # Processing the call keyword arguments (line 135)
    kwargs_503 = {}
    # Getting the type of 'len' (line 135)
    len_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 135, 19), len_500, *[args_502], **kwargs_503)
    
    # Processing the call keyword arguments (line 135)
    kwargs_505 = {}
    # Getting the type of 'range' (line 135)
    range_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'range', False)
    # Calling range(args, kwargs) (line 135)
    range_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 135, 13), range_499, *[len_call_result_504], **kwargs_505)
    
    # Assigning a type to the variable 'range_call_result_506' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'range_call_result_506', range_call_result_506)
    # Testing if the for loop is going to be iterated (line 135)
    # Testing the type of a for loop iterable (line 135)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 4), range_call_result_506)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 135, 4), range_call_result_506):
        # Getting the type of the for loop variable (line 135)
        for_loop_var_507 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 4), range_call_result_506)
        # Assigning a type to the variable 'x' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'x', for_loop_var_507)
        # SSA begins for a for statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 136):
        
        # Assigning a BinOp to a Name (line 136):
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 136)
        x_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'x')
        # Getting the type of 'rule' (line 136)
        rule_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'rule')
        # Obtaining the member 'lengths' of a type (line 136)
        lengths_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), rule_509, 'lengths')
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), lengths_510, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), getitem___511, x_508)
        
        int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 30), 'int')
        # Applying the binary operator '-' (line 136)
        result_sub_514 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 12), '-', subscript_call_result_512, int_513)
        
        # Assigning a type to the variable 'm' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'm', result_sub_514)
        
        
        # Call to range(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'm' (line 137)
        m_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'm', False)
        int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 27), 'int')
        # Applying the binary operator '+' (line 137)
        result_add_518 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 23), '+', m_516, int_517)
        
        # Processing the call keyword arguments (line 137)
        kwargs_519 = {}
        # Getting the type of 'range' (line 137)
        range_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'range', False)
        # Calling range(args, kwargs) (line 137)
        range_call_result_520 = invoke(stypy.reporting.localization.Localization(__file__, 137, 17), range_515, *[result_add_518], **kwargs_519)
        
        # Assigning a type to the variable 'range_call_result_520' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'range_call_result_520', range_call_result_520)
        # Testing if the for loop is going to be iterated (line 137)
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_520)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_520):
            # Getting the type of the for loop variable (line 137)
            for_loop_var_521 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_520)
            # Assigning a type to the variable 'n' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'n', for_loop_var_521)
            # SSA begins for a for statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to testbit(...): (line 138)
            # Processing the call arguments (line 138)
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 138)
            x_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 33), 'x', False)
            # Getting the type of 'rule' (line 138)
            rule_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'rule', False)
            # Obtaining the member 'args' of a type (line 138)
            args_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 23), rule_524, 'args')
            # Obtaining the member '__getitem__' of a type (line 138)
            getitem___526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 23), args_525, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 138)
            subscript_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), getitem___526, x_523)
            
            # Getting the type of 'n' (line 138)
            n_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'n', False)
            # Processing the call keyword arguments (line 138)
            kwargs_529 = {}
            # Getting the type of 'testbit' (line 138)
            testbit_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'testbit', False)
            # Calling testbit(args, kwargs) (line 138)
            testbit_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), testbit_522, *[subscript_call_result_527, n_528], **kwargs_529)
            
            # Testing if the type of an if condition is none (line 138)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 138, 12), testbit_call_result_530):
                
                # Evaluating a boolean operation
                
                # Getting the type of 'lpos' (line 159)
                lpos_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'lpos')
                int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'int')
                # Applying the binary operator '==' (line 159)
                result_eq_585 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), '==', lpos_583, int_584)
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'rpos' (line 159)
                rpos_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'rpos')
                int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 42), 'int')
                # Applying the binary operator '!=' (line 159)
                result_ne_588 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 34), '!=', rpos_586, int_587)
                
                
                # Getting the type of 'rpos' (line 159)
                rpos_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 49), 'rpos')
                # Getting the type of 'lpos' (line 159)
                lpos_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 57), 'lpos')
                # Applying the binary operator '<=' (line 159)
                result_le_591 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 49), '<=', rpos_589, lpos_590)
                
                # Applying the binary operator 'and' (line 159)
                result_and_keyword_592 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 34), 'and', result_ne_588, result_le_591)
                
                # Applying the binary operator 'or' (line 159)
                result_or_keyword_593 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), 'or', result_eq_585, result_and_keyword_592)
                
                # Testing if the type of an if condition is none (line 159)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 16), result_or_keyword_593):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 159)
                    if_condition_594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 16), result_or_keyword_593)
                    # Assigning a type to the variable 'if_condition_594' (line 159)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'if_condition_594', if_condition_594)
                    # SSA begins for if statement (line 159)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 160)
                    False_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 160)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'stypy_return_type', False_595)
                    # SSA join for if statement (line 159)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 161):
                
                # Assigning a Call to a Name (line 161):
                
                # Call to nextunset(...): (line 161)
                # Processing the call arguments (line 161)
                # Getting the type of 'lvec' (line 161)
                lvec_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'lvec', False)
                # Getting the type of 'lpos' (line 161)
                lpos_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'lpos', False)
                # Processing the call keyword arguments (line 161)
                kwargs_599 = {}
                # Getting the type of 'nextunset' (line 161)
                nextunset_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'nextunset', False)
                # Calling nextunset(args, kwargs) (line 161)
                nextunset_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 161, 23), nextunset_596, *[lvec_597, lpos_598], **kwargs_599)
                
                # Assigning a type to the variable 'lpos' (line 161)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'lpos', nextunset_call_result_600)
                
                # Evaluating a boolean operation
                
                # Getting the type of 'rpos' (line 162)
                rpos_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'rpos')
                int_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 27), 'int')
                # Applying the binary operator '!=' (line 162)
                result_ne_603 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 19), '!=', rpos_601, int_602)
                
                
                # Getting the type of 'rpos' (line 162)
                rpos_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 34), 'rpos')
                # Getting the type of 'lpos' (line 162)
                lpos_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'lpos')
                # Applying the binary operator '<' (line 162)
                result_lt_606 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 34), '<', rpos_604, lpos_605)
                
                # Applying the binary operator 'and' (line 162)
                result_and_keyword_607 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 19), 'and', result_ne_603, result_lt_606)
                
                # Testing if the type of an if condition is none (line 162)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 162, 16), result_and_keyword_607):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 162)
                    if_condition_608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 16), result_and_keyword_607)
                    # Assigning a type to the variable 'if_condition_608' (line 162)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'if_condition_608', if_condition_608)
                    # SSA begins for if statement (line 162)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 163)
                    False_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 163)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'stypy_return_type', False_609)
                    # SSA join for if statement (line 162)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'n' (line 164)
                n_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'n')
                # Getting the type of 'm' (line 164)
                m_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'm')
                # Applying the binary operator '==' (line 164)
                result_eq_612 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 19), '==', n_610, m_611)
                
                # Testing if the type of an if condition is none (line 164)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 164, 16), result_eq_612):
                    
                    
                    # Call to testbit(...): (line 167)
                    # Processing the call arguments (line 167)
                    # Getting the type of 'rvec' (line 167)
                    rvec_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'rvec', False)
                    # Getting the type of 'lpos' (line 167)
                    lpos_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'lpos', False)
                    # Processing the call keyword arguments (line 167)
                    kwargs_624 = {}
                    # Getting the type of 'testbit' (line 167)
                    testbit_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 167)
                    testbit_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 167, 25), testbit_621, *[rvec_622, lpos_623], **kwargs_624)
                    
                    # Applying the 'not' unary operator (line 167)
                    result_not__626 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 21), 'not', testbit_call_result_625)
                    
                    # Testing if the type of an if condition is none (line 167)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 167)
                        if_condition_627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626)
                        # Assigning a type to the variable 'if_condition_627' (line 167)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'if_condition_627', if_condition_627)
                        # SSA begins for if statement (line 167)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 168)
                        False_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 168)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'stypy_return_type', False_628)
                        # SSA join for if statement (line 167)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 164)
                    if_condition_613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 16), result_eq_612)
                    # Assigning a type to the variable 'if_condition_613' (line 164)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'if_condition_613', if_condition_613)
                    # SSA begins for if statement (line 164)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to testbit(...): (line 165)
                    # Processing the call arguments (line 165)
                    # Getting the type of 'rvec' (line 165)
                    rvec_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'rvec', False)
                    # Getting the type of 'lpos' (line 165)
                    lpos_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'lpos', False)
                    # Processing the call keyword arguments (line 165)
                    kwargs_617 = {}
                    # Getting the type of 'testbit' (line 165)
                    testbit_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 165)
                    testbit_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), testbit_614, *[rvec_615, lpos_616], **kwargs_617)
                    
                    # Testing if the type of an if condition is none (line 165)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 165, 20), testbit_call_result_618):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 165)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 20), testbit_call_result_618)
                        # Assigning a type to the variable 'if_condition_619' (line 165)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 165)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 166)
                        False_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 166)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'stypy_return_type', False_620)
                        # SSA join for if statement (line 165)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 164)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    # Call to testbit(...): (line 167)
                    # Processing the call arguments (line 167)
                    # Getting the type of 'rvec' (line 167)
                    rvec_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'rvec', False)
                    # Getting the type of 'lpos' (line 167)
                    lpos_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'lpos', False)
                    # Processing the call keyword arguments (line 167)
                    kwargs_624 = {}
                    # Getting the type of 'testbit' (line 167)
                    testbit_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 167)
                    testbit_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 167, 25), testbit_621, *[rvec_622, lpos_623], **kwargs_624)
                    
                    # Applying the 'not' unary operator (line 167)
                    result_not__626 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 21), 'not', testbit_call_result_625)
                    
                    # Testing if the type of an if condition is none (line 167)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 167)
                        if_condition_627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626)
                        # Assigning a type to the variable 'if_condition_627' (line 167)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'if_condition_627', if_condition_627)
                        # SSA begins for if statement (line 167)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 168)
                        False_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 168)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'stypy_return_type', False_628)
                        # SSA join for if statement (line 167)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 164)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 169):
                
                # Assigning a Call to a Name (line 169):
                
                # Call to nextset(...): (line 169)
                # Processing the call arguments (line 169)
                # Getting the type of 'lvec' (line 169)
                lvec_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'lvec', False)
                # Getting the type of 'lpos' (line 169)
                lpos_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'lpos', False)
                # Processing the call keyword arguments (line 169)
                kwargs_632 = {}
                # Getting the type of 'nextset' (line 169)
                nextset_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'nextset', False)
                # Calling nextset(args, kwargs) (line 169)
                nextset_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 169, 23), nextset_629, *[lvec_630, lpos_631], **kwargs_632)
                
                # Assigning a type to the variable 'lpos' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'lpos', nextset_call_result_633)
            else:
                
                # Testing the type of an if condition (line 138)
                if_condition_531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 12), testbit_call_result_530)
                # Assigning a type to the variable 'if_condition_531' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'if_condition_531', if_condition_531)
                # SSA begins for if statement (line 138)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'rpos' (line 142)
                rpos_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'rpos')
                int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'int')
                # Applying the binary operator '==' (line 142)
                result_eq_534 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 19), '==', rpos_532, int_533)
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'lpos' (line 142)
                lpos_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'lpos')
                int_536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 42), 'int')
                # Applying the binary operator '!=' (line 142)
                result_ne_537 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 34), '!=', lpos_535, int_536)
                
                
                # Getting the type of 'lpos' (line 142)
                lpos_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 49), 'lpos')
                # Getting the type of 'rpos' (line 142)
                rpos_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 57), 'rpos')
                # Applying the binary operator '<=' (line 142)
                result_le_540 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 49), '<=', lpos_538, rpos_539)
                
                # Applying the binary operator 'and' (line 142)
                result_and_keyword_541 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 34), 'and', result_ne_537, result_le_540)
                
                # Applying the binary operator 'or' (line 142)
                result_or_keyword_542 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 19), 'or', result_eq_534, result_and_keyword_541)
                
                # Testing if the type of an if condition is none (line 142)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 142, 16), result_or_keyword_542):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 142)
                    if_condition_543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 16), result_or_keyword_542)
                    # Assigning a type to the variable 'if_condition_543' (line 142)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'if_condition_543', if_condition_543)
                    # SSA begins for if statement (line 142)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 143)
                    False_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 143)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'stypy_return_type', False_544)
                    # SSA join for if statement (line 142)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 145):
                
                # Assigning a Call to a Name (line 145):
                
                # Call to nextunset(...): (line 145)
                # Processing the call arguments (line 145)
                # Getting the type of 'rvec' (line 145)
                rvec_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), 'rvec', False)
                # Getting the type of 'rpos' (line 145)
                rpos_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'rpos', False)
                # Processing the call keyword arguments (line 145)
                kwargs_548 = {}
                # Getting the type of 'nextunset' (line 145)
                nextunset_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 23), 'nextunset', False)
                # Calling nextunset(args, kwargs) (line 145)
                nextunset_call_result_549 = invoke(stypy.reporting.localization.Localization(__file__, 145, 23), nextunset_545, *[rvec_546, rpos_547], **kwargs_548)
                
                # Assigning a type to the variable 'rpos' (line 145)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'rpos', nextunset_call_result_549)
                
                # Evaluating a boolean operation
                
                # Getting the type of 'lpos' (line 146)
                lpos_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'lpos')
                int_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 27), 'int')
                # Applying the binary operator '!=' (line 146)
                result_ne_552 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 19), '!=', lpos_550, int_551)
                
                
                # Getting the type of 'lpos' (line 146)
                lpos_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'lpos')
                # Getting the type of 'rpos' (line 146)
                rpos_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'rpos')
                # Applying the binary operator '<' (line 146)
                result_lt_555 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 34), '<', lpos_553, rpos_554)
                
                # Applying the binary operator 'and' (line 146)
                result_and_keyword_556 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 19), 'and', result_ne_552, result_lt_555)
                
                # Testing if the type of an if condition is none (line 146)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 16), result_and_keyword_556):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 146)
                    if_condition_557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 16), result_and_keyword_556)
                    # Assigning a type to the variable 'if_condition_557' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'if_condition_557', if_condition_557)
                    # SSA begins for if statement (line 146)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 147)
                    False_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 147)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'stypy_return_type', False_558)
                    # SSA join for if statement (line 146)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'n' (line 150)
                n_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'n')
                # Getting the type of 'm' (line 150)
                m_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'm')
                # Applying the binary operator '==' (line 150)
                result_eq_561 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 19), '==', n_559, m_560)
                
                # Testing if the type of an if condition is none (line 150)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 16), result_eq_561):
                    
                    
                    # Call to testbit(...): (line 153)
                    # Processing the call arguments (line 153)
                    # Getting the type of 'lvec' (line 153)
                    lvec_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'lvec', False)
                    # Getting the type of 'rpos' (line 153)
                    rpos_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'rpos', False)
                    # Processing the call keyword arguments (line 153)
                    kwargs_573 = {}
                    # Getting the type of 'testbit' (line 153)
                    testbit_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 153)
                    testbit_call_result_574 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), testbit_570, *[lvec_571, rpos_572], **kwargs_573)
                    
                    # Applying the 'not' unary operator (line 153)
                    result_not__575 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 21), 'not', testbit_call_result_574)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 21), result_not__575):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 21), result_not__575)
                        # Assigning a type to the variable 'if_condition_576' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'if_condition_576', if_condition_576)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 154)
                        False_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 154)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'stypy_return_type', False_577)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 150)
                    if_condition_562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 16), result_eq_561)
                    # Assigning a type to the variable 'if_condition_562' (line 150)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'if_condition_562', if_condition_562)
                    # SSA begins for if statement (line 150)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to testbit(...): (line 151)
                    # Processing the call arguments (line 151)
                    # Getting the type of 'lvec' (line 151)
                    lvec_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'lvec', False)
                    # Getting the type of 'rpos' (line 151)
                    rpos_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'rpos', False)
                    # Processing the call keyword arguments (line 151)
                    kwargs_566 = {}
                    # Getting the type of 'testbit' (line 151)
                    testbit_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 151)
                    testbit_call_result_567 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), testbit_563, *[lvec_564, rpos_565], **kwargs_566)
                    
                    # Testing if the type of an if condition is none (line 151)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 20), testbit_call_result_567):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 151)
                        if_condition_568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 20), testbit_call_result_567)
                        # Assigning a type to the variable 'if_condition_568' (line 151)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'if_condition_568', if_condition_568)
                        # SSA begins for if statement (line 151)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 152)
                        False_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 152)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'stypy_return_type', False_569)
                        # SSA join for if statement (line 151)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 150)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    # Call to testbit(...): (line 153)
                    # Processing the call arguments (line 153)
                    # Getting the type of 'lvec' (line 153)
                    lvec_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'lvec', False)
                    # Getting the type of 'rpos' (line 153)
                    rpos_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'rpos', False)
                    # Processing the call keyword arguments (line 153)
                    kwargs_573 = {}
                    # Getting the type of 'testbit' (line 153)
                    testbit_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 153)
                    testbit_call_result_574 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), testbit_570, *[lvec_571, rpos_572], **kwargs_573)
                    
                    # Applying the 'not' unary operator (line 153)
                    result_not__575 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 21), 'not', testbit_call_result_574)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 21), result_not__575):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 21), result_not__575)
                        # Assigning a type to the variable 'if_condition_576' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'if_condition_576', if_condition_576)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 154)
                        False_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 154)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'stypy_return_type', False_577)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 150)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 156):
                
                # Assigning a Call to a Name (line 156):
                
                # Call to nextset(...): (line 156)
                # Processing the call arguments (line 156)
                # Getting the type of 'rvec' (line 156)
                rvec_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'rvec', False)
                # Getting the type of 'rpos' (line 156)
                rpos_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'rpos', False)
                # Processing the call keyword arguments (line 156)
                kwargs_581 = {}
                # Getting the type of 'nextset' (line 156)
                nextset_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'nextset', False)
                # Calling nextset(args, kwargs) (line 156)
                nextset_call_result_582 = invoke(stypy.reporting.localization.Localization(__file__, 156, 23), nextset_578, *[rvec_579, rpos_580], **kwargs_581)
                
                # Assigning a type to the variable 'rpos' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'rpos', nextset_call_result_582)
                # SSA branch for the else part of an if statement (line 138)
                module_type_store.open_ssa_branch('else')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'lpos' (line 159)
                lpos_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'lpos')
                int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'int')
                # Applying the binary operator '==' (line 159)
                result_eq_585 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), '==', lpos_583, int_584)
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'rpos' (line 159)
                rpos_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'rpos')
                int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 42), 'int')
                # Applying the binary operator '!=' (line 159)
                result_ne_588 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 34), '!=', rpos_586, int_587)
                
                
                # Getting the type of 'rpos' (line 159)
                rpos_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 49), 'rpos')
                # Getting the type of 'lpos' (line 159)
                lpos_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 57), 'lpos')
                # Applying the binary operator '<=' (line 159)
                result_le_591 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 49), '<=', rpos_589, lpos_590)
                
                # Applying the binary operator 'and' (line 159)
                result_and_keyword_592 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 34), 'and', result_ne_588, result_le_591)
                
                # Applying the binary operator 'or' (line 159)
                result_or_keyword_593 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), 'or', result_eq_585, result_and_keyword_592)
                
                # Testing if the type of an if condition is none (line 159)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 16), result_or_keyword_593):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 159)
                    if_condition_594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 16), result_or_keyword_593)
                    # Assigning a type to the variable 'if_condition_594' (line 159)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'if_condition_594', if_condition_594)
                    # SSA begins for if statement (line 159)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 160)
                    False_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 160)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'stypy_return_type', False_595)
                    # SSA join for if statement (line 159)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 161):
                
                # Assigning a Call to a Name (line 161):
                
                # Call to nextunset(...): (line 161)
                # Processing the call arguments (line 161)
                # Getting the type of 'lvec' (line 161)
                lvec_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'lvec', False)
                # Getting the type of 'lpos' (line 161)
                lpos_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'lpos', False)
                # Processing the call keyword arguments (line 161)
                kwargs_599 = {}
                # Getting the type of 'nextunset' (line 161)
                nextunset_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'nextunset', False)
                # Calling nextunset(args, kwargs) (line 161)
                nextunset_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 161, 23), nextunset_596, *[lvec_597, lpos_598], **kwargs_599)
                
                # Assigning a type to the variable 'lpos' (line 161)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'lpos', nextunset_call_result_600)
                
                # Evaluating a boolean operation
                
                # Getting the type of 'rpos' (line 162)
                rpos_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'rpos')
                int_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 27), 'int')
                # Applying the binary operator '!=' (line 162)
                result_ne_603 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 19), '!=', rpos_601, int_602)
                
                
                # Getting the type of 'rpos' (line 162)
                rpos_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 34), 'rpos')
                # Getting the type of 'lpos' (line 162)
                lpos_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'lpos')
                # Applying the binary operator '<' (line 162)
                result_lt_606 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 34), '<', rpos_604, lpos_605)
                
                # Applying the binary operator 'and' (line 162)
                result_and_keyword_607 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 19), 'and', result_ne_603, result_lt_606)
                
                # Testing if the type of an if condition is none (line 162)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 162, 16), result_and_keyword_607):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 162)
                    if_condition_608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 16), result_and_keyword_607)
                    # Assigning a type to the variable 'if_condition_608' (line 162)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'if_condition_608', if_condition_608)
                    # SSA begins for if statement (line 162)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 163)
                    False_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 163)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'stypy_return_type', False_609)
                    # SSA join for if statement (line 162)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'n' (line 164)
                n_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'n')
                # Getting the type of 'm' (line 164)
                m_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'm')
                # Applying the binary operator '==' (line 164)
                result_eq_612 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 19), '==', n_610, m_611)
                
                # Testing if the type of an if condition is none (line 164)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 164, 16), result_eq_612):
                    
                    
                    # Call to testbit(...): (line 167)
                    # Processing the call arguments (line 167)
                    # Getting the type of 'rvec' (line 167)
                    rvec_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'rvec', False)
                    # Getting the type of 'lpos' (line 167)
                    lpos_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'lpos', False)
                    # Processing the call keyword arguments (line 167)
                    kwargs_624 = {}
                    # Getting the type of 'testbit' (line 167)
                    testbit_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 167)
                    testbit_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 167, 25), testbit_621, *[rvec_622, lpos_623], **kwargs_624)
                    
                    # Applying the 'not' unary operator (line 167)
                    result_not__626 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 21), 'not', testbit_call_result_625)
                    
                    # Testing if the type of an if condition is none (line 167)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 167)
                        if_condition_627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626)
                        # Assigning a type to the variable 'if_condition_627' (line 167)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'if_condition_627', if_condition_627)
                        # SSA begins for if statement (line 167)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 168)
                        False_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 168)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'stypy_return_type', False_628)
                        # SSA join for if statement (line 167)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 164)
                    if_condition_613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 16), result_eq_612)
                    # Assigning a type to the variable 'if_condition_613' (line 164)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'if_condition_613', if_condition_613)
                    # SSA begins for if statement (line 164)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to testbit(...): (line 165)
                    # Processing the call arguments (line 165)
                    # Getting the type of 'rvec' (line 165)
                    rvec_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'rvec', False)
                    # Getting the type of 'lpos' (line 165)
                    lpos_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'lpos', False)
                    # Processing the call keyword arguments (line 165)
                    kwargs_617 = {}
                    # Getting the type of 'testbit' (line 165)
                    testbit_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 165)
                    testbit_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), testbit_614, *[rvec_615, lpos_616], **kwargs_617)
                    
                    # Testing if the type of an if condition is none (line 165)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 165, 20), testbit_call_result_618):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 165)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 20), testbit_call_result_618)
                        # Assigning a type to the variable 'if_condition_619' (line 165)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 165)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 166)
                        False_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 166)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'stypy_return_type', False_620)
                        # SSA join for if statement (line 165)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 164)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    # Call to testbit(...): (line 167)
                    # Processing the call arguments (line 167)
                    # Getting the type of 'rvec' (line 167)
                    rvec_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'rvec', False)
                    # Getting the type of 'lpos' (line 167)
                    lpos_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'lpos', False)
                    # Processing the call keyword arguments (line 167)
                    kwargs_624 = {}
                    # Getting the type of 'testbit' (line 167)
                    testbit_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'testbit', False)
                    # Calling testbit(args, kwargs) (line 167)
                    testbit_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 167, 25), testbit_621, *[rvec_622, lpos_623], **kwargs_624)
                    
                    # Applying the 'not' unary operator (line 167)
                    result_not__626 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 21), 'not', testbit_call_result_625)
                    
                    # Testing if the type of an if condition is none (line 167)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 167)
                        if_condition_627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 21), result_not__626)
                        # Assigning a type to the variable 'if_condition_627' (line 167)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'if_condition_627', if_condition_627)
                        # SSA begins for if statement (line 167)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 168)
                        False_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 168)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'stypy_return_type', False_628)
                        # SSA join for if statement (line 167)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 164)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 169):
                
                # Assigning a Call to a Name (line 169):
                
                # Call to nextset(...): (line 169)
                # Processing the call arguments (line 169)
                # Getting the type of 'lvec' (line 169)
                lvec_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'lvec', False)
                # Getting the type of 'lpos' (line 169)
                lpos_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'lpos', False)
                # Processing the call keyword arguments (line 169)
                kwargs_632 = {}
                # Getting the type of 'nextset' (line 169)
                nextset_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'nextset', False)
                # Calling nextset(args, kwargs) (line 169)
                nextset_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 169, 23), nextset_629, *[lvec_630, lpos_631], **kwargs_632)
                
                # Assigning a type to the variable 'lpos' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'lpos', nextset_call_result_633)
                # SSA join for if statement (line 138)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lpos' (line 171)
    lpos_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'lpos')
    int_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 15), 'int')
    # Applying the binary operator '!=' (line 171)
    result_ne_636 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '!=', lpos_634, int_635)
    
    
    # Getting the type of 'rpos' (line 171)
    rpos_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 21), 'rpos')
    int_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'int')
    # Applying the binary operator '!=' (line 171)
    result_ne_639 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 21), '!=', rpos_637, int_638)
    
    # Applying the binary operator 'or' (line 171)
    result_or_keyword_640 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), 'or', result_ne_636, result_ne_639)
    
    # Testing if the type of an if condition is none (line 171)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 171, 4), result_or_keyword_640):
        pass
    else:
        
        # Testing the type of an if condition (line 171)
        if_condition_641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_or_keyword_640)
        # Assigning a type to the variable 'if_condition_641' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_641', if_condition_641)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 172)
        False_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', False_642)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'True' (line 174)
    True_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', True_643)
    
    # ################# End of 'concat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'concat' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_644)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'concat'
    return stypy_return_type_644

# Assigning a type to the variable 'concat' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'concat', concat)

@norecursion
def mostprobablederivation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mostprobablederivation'
    module_type_store = module_type_store.open_function_context('mostprobablederivation', 177, 0, False)
    
    # Passed parameters checking function
    mostprobablederivation.stypy_localization = localization
    mostprobablederivation.stypy_type_of_self = None
    mostprobablederivation.stypy_type_store = module_type_store
    mostprobablederivation.stypy_function_name = 'mostprobablederivation'
    mostprobablederivation.stypy_param_names_list = ['chart', 'start', 'tolabel']
    mostprobablederivation.stypy_varargs_param_name = None
    mostprobablederivation.stypy_kwargs_param_name = None
    mostprobablederivation.stypy_call_defaults = defaults
    mostprobablederivation.stypy_call_varargs = varargs
    mostprobablederivation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mostprobablederivation', ['chart', 'start', 'tolabel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mostprobablederivation', localization, ['chart', 'start', 'tolabel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mostprobablederivation(...)' code ##################

    str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, (-1)), 'str', ' produce a string representation of the viterbi parse in bracket\n    notation')
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to min(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Obtaining the type of the subscript
    # Getting the type of 'start' (line 180)
    start_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'start', False)
    # Getting the type of 'chart' (line 180)
    chart_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'chart', False)
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 15), chart_648, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_650 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), getitem___649, start_647)
    
    # Processing the call keyword arguments (line 180)
    kwargs_651 = {}
    # Getting the type of 'min' (line 180)
    min_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'min', False)
    # Calling min(args, kwargs) (line 180)
    min_call_result_652 = invoke(stypy.reporting.localization.Localization(__file__, 180, 11), min_646, *[subscript_call_result_650], **kwargs_651)
    
    # Assigning a type to the variable 'edge' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'edge', min_call_result_652)
    
    # Obtaining an instance of the builtin type 'tuple' (line 181)
    tuple_653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 181)
    # Adding element type (line 181)
    
    # Call to getmpd(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'chart' (line 181)
    chart_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'chart', False)
    # Getting the type of 'start' (line 181)
    start_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'start', False)
    # Getting the type of 'tolabel' (line 181)
    tolabel_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 32), 'tolabel', False)
    # Processing the call keyword arguments (line 181)
    kwargs_658 = {}
    # Getting the type of 'getmpd' (line 181)
    getmpd_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'getmpd', False)
    # Calling getmpd(args, kwargs) (line 181)
    getmpd_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 181, 11), getmpd_654, *[chart_655, start_656, tolabel_657], **kwargs_658)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 11), tuple_653, getmpd_call_result_659)
    # Adding element type (line 181)
    # Getting the type of 'edge' (line 181)
    edge_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'edge')
    # Obtaining the member 'inside' of a type (line 181)
    inside_661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 42), edge_660, 'inside')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 11), tuple_653, inside_661)
    
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type', tuple_653)
    
    # ################# End of 'mostprobablederivation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mostprobablederivation' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_662)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mostprobablederivation'
    return stypy_return_type_662

# Assigning a type to the variable 'mostprobablederivation' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'mostprobablederivation', mostprobablederivation)

@norecursion
def getmpd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getmpd'
    module_type_store = module_type_store.open_function_context('getmpd', 184, 0, False)
    
    # Passed parameters checking function
    getmpd.stypy_localization = localization
    getmpd.stypy_type_of_self = None
    getmpd.stypy_type_store = module_type_store
    getmpd.stypy_function_name = 'getmpd'
    getmpd.stypy_param_names_list = ['chart', 'start', 'tolabel']
    getmpd.stypy_varargs_param_name = None
    getmpd.stypy_kwargs_param_name = None
    getmpd.stypy_call_defaults = defaults
    getmpd.stypy_call_varargs = varargs
    getmpd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getmpd', ['chart', 'start', 'tolabel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getmpd', localization, ['chart', 'start', 'tolabel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getmpd(...)' code ##################

    
    # Assigning a Call to a Name (line 185):
    
    # Assigning a Call to a Name (line 185):
    
    # Call to min(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Obtaining the type of the subscript
    # Getting the type of 'start' (line 185)
    start_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'start', False)
    # Getting the type of 'chart' (line 185)
    chart_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'chart', False)
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 15), chart_665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 185, 15), getitem___666, start_664)
    
    # Processing the call keyword arguments (line 185)
    kwargs_668 = {}
    # Getting the type of 'min' (line 185)
    min_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'min', False)
    # Calling min(args, kwargs) (line 185)
    min_call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 185, 11), min_663, *[subscript_call_result_667], **kwargs_668)
    
    # Assigning a type to the variable 'edge' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'edge', min_call_result_669)
    
    # Evaluating a boolean operation
    # Getting the type of 'edge' (line 186)
    edge_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 'edge')
    # Obtaining the member 'right' of a type (line 186)
    right_671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 7), edge_670, 'right')
    # Getting the type of 'edge' (line 186)
    edge_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'edge')
    # Obtaining the member 'right' of a type (line 186)
    right_673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 22), edge_672, 'right')
    # Obtaining the member 'label' of a type (line 186)
    label_674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 22), right_673, 'label')
    # Applying the binary operator 'and' (line 186)
    result_and_keyword_675 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 7), 'and', right_671, label_674)
    
    # Testing if the type of an if condition is none (line 186)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 186, 4), result_and_keyword_675):
        str_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 15), 'str', '(%s %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 191)
        tuple_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 191)
        # Adding element type (line 191)
        
        # Obtaining the type of the subscript
        # Getting the type of 'start' (line 191)
        start_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 36), 'start')
        # Obtaining the member 'label' of a type (line 191)
        label_702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 36), start_701, 'label')
        # Getting the type of 'tolabel' (line 191)
        tolabel_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'tolabel')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 28), tolabel_703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 191, 28), getitem___704, label_702)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 28), tuple_700, subscript_call_result_705)
        # Adding element type (line 191)
        
        # Getting the type of 'edge' (line 193)
        edge_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'edge')
        # Obtaining the member 'left' of a type (line 193)
        left_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 31), edge_706, 'left')
        # Obtaining the member 'label' of a type (line 193)
        label_708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 31), left_707, 'label')
        # Testing the type of an if expression (line 192)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 28), label_708)
        # SSA begins for if expression (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to getmpd(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'chart' (line 192)
        chart_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 35), 'chart', False)
        # Getting the type of 'edge' (line 192)
        edge_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'edge', False)
        # Obtaining the member 'left' of a type (line 192)
        left_712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 42), edge_711, 'left')
        # Getting the type of 'tolabel' (line 192)
        tolabel_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 53), 'tolabel', False)
        # Processing the call keyword arguments (line 192)
        kwargs_714 = {}
        # Getting the type of 'getmpd' (line 192)
        getmpd_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'getmpd', False)
        # Calling getmpd(args, kwargs) (line 192)
        getmpd_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 192, 28), getmpd_709, *[chart_710, left_712, tolabel_713], **kwargs_714)
        
        # SSA branch for the else part of an if expression (line 192)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to str(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'edge' (line 193)
        edge_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 56), 'edge', False)
        # Obtaining the member 'left' of a type (line 193)
        left_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 56), edge_717, 'left')
        # Obtaining the member 'vec' of a type (line 193)
        vec_719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 56), left_718, 'vec')
        # Processing the call keyword arguments (line 193)
        kwargs_720 = {}
        # Getting the type of 'str' (line 193)
        str_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 52), 'str', False)
        # Calling str(args, kwargs) (line 193)
        str_call_result_721 = invoke(stypy.reporting.localization.Localization(__file__, 193, 52), str_716, *[vec_719], **kwargs_720)
        
        # SSA join for if expression (line 192)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_722 = union_type.UnionType.add(getmpd_call_result_715, str_call_result_721)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 28), tuple_700, if_exp_722)
        
        # Applying the binary operator '%' (line 191)
        result_mod_723 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '%', str_699, tuple_700)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', result_mod_723)
    else:
        
        # Testing the type of an if condition (line 186)
        if_condition_676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 4), result_and_keyword_675)
        # Assigning a type to the variable 'if_condition_676' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'if_condition_676', if_condition_676)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 15), 'str', '(%s %s %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        
        # Obtaining the type of the subscript
        # Getting the type of 'start' (line 187)
        start_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 39), 'start')
        # Obtaining the member 'label' of a type (line 187)
        label_680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 39), start_679, 'label')
        # Getting the type of 'tolabel' (line 187)
        tolabel_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'tolabel')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 31), tolabel_681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 187, 31), getitem___682, label_680)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 31), tuple_678, subscript_call_result_683)
        # Adding element type (line 187)
        
        # Call to getmpd(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'chart' (line 188)
        chart_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'chart', False)
        # Getting the type of 'edge' (line 188)
        edge_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 45), 'edge', False)
        # Obtaining the member 'left' of a type (line 188)
        left_687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 45), edge_686, 'left')
        # Getting the type of 'tolabel' (line 188)
        tolabel_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 56), 'tolabel', False)
        # Processing the call keyword arguments (line 188)
        kwargs_689 = {}
        # Getting the type of 'getmpd' (line 188)
        getmpd_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'getmpd', False)
        # Calling getmpd(args, kwargs) (line 188)
        getmpd_call_result_690 = invoke(stypy.reporting.localization.Localization(__file__, 188, 31), getmpd_684, *[chart_685, left_687, tolabel_688], **kwargs_689)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 31), tuple_678, getmpd_call_result_690)
        # Adding element type (line 187)
        
        # Call to getmpd(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'chart' (line 189)
        chart_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 38), 'chart', False)
        # Getting the type of 'edge' (line 189)
        edge_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 45), 'edge', False)
        # Obtaining the member 'right' of a type (line 189)
        right_694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 45), edge_693, 'right')
        # Getting the type of 'tolabel' (line 189)
        tolabel_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 57), 'tolabel', False)
        # Processing the call keyword arguments (line 189)
        kwargs_696 = {}
        # Getting the type of 'getmpd' (line 189)
        getmpd_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 31), 'getmpd', False)
        # Calling getmpd(args, kwargs) (line 189)
        getmpd_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 189, 31), getmpd_691, *[chart_692, right_694, tolabel_695], **kwargs_696)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 31), tuple_678, getmpd_call_result_697)
        
        # Applying the binary operator '%' (line 187)
        result_mod_698 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 15), '%', str_677, tuple_678)
        
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', result_mod_698)
        # SSA branch for the else part of an if statement (line 186)
        module_type_store.open_ssa_branch('else')
        str_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 15), 'str', '(%s %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 191)
        tuple_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 191)
        # Adding element type (line 191)
        
        # Obtaining the type of the subscript
        # Getting the type of 'start' (line 191)
        start_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 36), 'start')
        # Obtaining the member 'label' of a type (line 191)
        label_702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 36), start_701, 'label')
        # Getting the type of 'tolabel' (line 191)
        tolabel_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'tolabel')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 28), tolabel_703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 191, 28), getitem___704, label_702)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 28), tuple_700, subscript_call_result_705)
        # Adding element type (line 191)
        
        # Getting the type of 'edge' (line 193)
        edge_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'edge')
        # Obtaining the member 'left' of a type (line 193)
        left_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 31), edge_706, 'left')
        # Obtaining the member 'label' of a type (line 193)
        label_708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 31), left_707, 'label')
        # Testing the type of an if expression (line 192)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 28), label_708)
        # SSA begins for if expression (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to getmpd(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'chart' (line 192)
        chart_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 35), 'chart', False)
        # Getting the type of 'edge' (line 192)
        edge_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'edge', False)
        # Obtaining the member 'left' of a type (line 192)
        left_712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 42), edge_711, 'left')
        # Getting the type of 'tolabel' (line 192)
        tolabel_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 53), 'tolabel', False)
        # Processing the call keyword arguments (line 192)
        kwargs_714 = {}
        # Getting the type of 'getmpd' (line 192)
        getmpd_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'getmpd', False)
        # Calling getmpd(args, kwargs) (line 192)
        getmpd_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 192, 28), getmpd_709, *[chart_710, left_712, tolabel_713], **kwargs_714)
        
        # SSA branch for the else part of an if expression (line 192)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to str(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'edge' (line 193)
        edge_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 56), 'edge', False)
        # Obtaining the member 'left' of a type (line 193)
        left_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 56), edge_717, 'left')
        # Obtaining the member 'vec' of a type (line 193)
        vec_719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 56), left_718, 'vec')
        # Processing the call keyword arguments (line 193)
        kwargs_720 = {}
        # Getting the type of 'str' (line 193)
        str_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 52), 'str', False)
        # Calling str(args, kwargs) (line 193)
        str_call_result_721 = invoke(stypy.reporting.localization.Localization(__file__, 193, 52), str_716, *[vec_719], **kwargs_720)
        
        # SSA join for if expression (line 192)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_722 = union_type.UnionType.add(getmpd_call_result_715, str_call_result_721)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 28), tuple_700, if_exp_722)
        
        # Applying the binary operator '%' (line 191)
        result_mod_723 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '%', str_699, tuple_700)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', result_mod_723)
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'getmpd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getmpd' in the type store
    # Getting the type of 'stypy_return_type' (line 184)
    stypy_return_type_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_724)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getmpd'
    return stypy_return_type_724

# Assigning a type to the variable 'getmpd' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'getmpd', getmpd)

@norecursion
def binrepr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'binrepr'
    module_type_store = module_type_store.open_function_context('binrepr', 196, 0, False)
    
    # Passed parameters checking function
    binrepr.stypy_localization = localization
    binrepr.stypy_type_of_self = None
    binrepr.stypy_type_store = module_type_store
    binrepr.stypy_function_name = 'binrepr'
    binrepr.stypy_param_names_list = ['a', 'sent']
    binrepr.stypy_varargs_param_name = None
    binrepr.stypy_kwargs_param_name = None
    binrepr.stypy_call_defaults = defaults
    binrepr.stypy_call_varargs = varargs
    binrepr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binrepr', ['a', 'sent'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binrepr', localization, ['a', 'sent'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binrepr(...)' code ##################

    
    # Call to join(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Call to reversed(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Call to rjust(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Call to len(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'sent' (line 197)
    sent_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 53), 'sent', False)
    # Processing the call keyword arguments (line 197)
    kwargs_740 = {}
    # Getting the type of 'len' (line 197)
    len_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 49), 'len', False)
    # Calling len(args, kwargs) (line 197)
    len_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 197, 49), len_738, *[sent_739], **kwargs_740)
    
    str_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 60), 'str', '0')
    # Processing the call keyword arguments (line 197)
    kwargs_743 = {}
    
    # Obtaining the type of the subscript
    int_728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 39), 'int')
    slice_729 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 197, 28), int_728, None, None)
    
    # Call to bin(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'a' (line 197)
    a_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 32), 'a', False)
    # Obtaining the member 'vec' of a type (line 197)
    vec_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 32), a_731, 'vec')
    # Processing the call keyword arguments (line 197)
    kwargs_733 = {}
    # Getting the type of 'bin' (line 197)
    bin_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 28), 'bin', False)
    # Calling bin(args, kwargs) (line 197)
    bin_call_result_734 = invoke(stypy.reporting.localization.Localization(__file__, 197, 28), bin_730, *[vec_732], **kwargs_733)
    
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 28), bin_call_result_734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 197, 28), getitem___735, slice_729)
    
    # Obtaining the member 'rjust' of a type (line 197)
    rjust_737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 28), subscript_call_result_736, 'rjust')
    # Calling rjust(args, kwargs) (line 197)
    rjust_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 197, 28), rjust_737, *[len_call_result_741, str_742], **kwargs_743)
    
    # Processing the call keyword arguments (line 197)
    kwargs_745 = {}
    # Getting the type of 'reversed' (line 197)
    reversed_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'reversed', False)
    # Calling reversed(args, kwargs) (line 197)
    reversed_call_result_746 = invoke(stypy.reporting.localization.Localization(__file__, 197, 19), reversed_727, *[rjust_call_result_744], **kwargs_745)
    
    # Processing the call keyword arguments (line 197)
    kwargs_747 = {}
    str_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 11), 'str', '')
    # Obtaining the member 'join' of a type (line 197)
    join_726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 11), str_725, 'join')
    # Calling join(args, kwargs) (line 197)
    join_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 197, 11), join_726, *[reversed_call_result_746], **kwargs_747)
    
    # Assigning a type to the variable 'stypy_return_type' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type', join_call_result_748)
    
    # ################# End of 'binrepr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binrepr' in the type store
    # Getting the type of 'stypy_return_type' (line 196)
    stypy_return_type_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binrepr'
    return stypy_return_type_749

# Assigning a type to the variable 'binrepr' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'binrepr', binrepr)

@norecursion
def pprint_chart(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pprint_chart'
    module_type_store = module_type_store.open_function_context('pprint_chart', 200, 0, False)
    
    # Passed parameters checking function
    pprint_chart.stypy_localization = localization
    pprint_chart.stypy_type_of_self = None
    pprint_chart.stypy_type_store = module_type_store
    pprint_chart.stypy_function_name = 'pprint_chart'
    pprint_chart.stypy_param_names_list = ['chart', 'sent', 'tolabel']
    pprint_chart.stypy_varargs_param_name = None
    pprint_chart.stypy_kwargs_param_name = None
    pprint_chart.stypy_call_defaults = defaults
    pprint_chart.stypy_call_varargs = varargs
    pprint_chart.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pprint_chart', ['chart', 'sent', 'tolabel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pprint_chart', localization, ['chart', 'sent', 'tolabel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pprint_chart(...)' code ##################

    
    
    # Call to sorted(...): (line 202)
    # Processing the call arguments (line 202)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 202, 23, True)
    # Calculating comprehension expression
    # Getting the type of 'chart' (line 202)
    chart_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 53), 'chart', False)
    comprehension_759 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 23), chart_758)
    # Assigning a type to the variable 'a' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'a', comprehension_759)
    
    # Obtaining an instance of the builtin type 'tuple' (line 202)
    tuple_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 202)
    # Adding element type (line 202)
    
    # Call to bitcount(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'a' (line 202)
    a_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 33), 'a', False)
    # Obtaining the member 'vec' of a type (line 202)
    vec_754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 33), a_753, 'vec')
    # Processing the call keyword arguments (line 202)
    kwargs_755 = {}
    # Getting the type of 'bitcount' (line 202)
    bitcount_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'bitcount', False)
    # Calling bitcount(args, kwargs) (line 202)
    bitcount_call_result_756 = invoke(stypy.reporting.localization.Localization(__file__, 202, 24), bitcount_752, *[vec_754], **kwargs_755)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 24), tuple_751, bitcount_call_result_756)
    # Adding element type (line 202)
    # Getting the type of 'a' (line 202)
    a_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 41), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 24), tuple_751, a_757)
    
    list_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 23), list_760, tuple_751)
    # Processing the call keyword arguments (line 202)
    kwargs_761 = {}
    # Getting the type of 'sorted' (line 202)
    sorted_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'sorted', False)
    # Calling sorted(args, kwargs) (line 202)
    sorted_call_result_762 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), sorted_750, *[list_760], **kwargs_761)
    
    # Assigning a type to the variable 'sorted_call_result_762' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'sorted_call_result_762', sorted_call_result_762)
    # Testing if the for loop is going to be iterated (line 202)
    # Testing the type of a for loop iterable (line 202)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 202, 4), sorted_call_result_762)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 202, 4), sorted_call_result_762):
        # Getting the type of the for loop variable (line 202)
        for_loop_var_763 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 202, 4), sorted_call_result_762)
        # Assigning a type to the variable 'n' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 4), for_loop_var_763))
        # Assigning a type to the variable 'a' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 4), for_loop_var_763))
        # SSA begins for a for statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 203)
        a_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'a')
        # Getting the type of 'chart' (line 203)
        chart_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'chart')
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), chart_765, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_767 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), getitem___766, a_764)
        
        # Applying the 'not' unary operator (line 203)
        result_not__768 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), 'not', subscript_call_result_767)
        
        # Testing if the type of an if condition is none (line 203)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 8), result_not__768):
            pass
        else:
            
            # Testing the type of an if condition (line 203)
            if_condition_769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_not__768)
            # Assigning a type to the variable 'if_condition_769' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_769', if_condition_769)
            # SSA begins for if statement (line 203)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 203)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to binrepr(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'a' (line 205)
        a_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'a', False)
        # Getting the type of 'sent' (line 205)
        sent_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'sent', False)
        # Processing the call keyword arguments (line 205)
        kwargs_773 = {}
        # Getting the type of 'binrepr' (line 205)
        binrepr_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'binrepr', False)
        # Calling binrepr(args, kwargs) (line 205)
        binrepr_call_result_774 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), binrepr_770, *[a_771, sent_772], **kwargs_773)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 206)
        a_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'a')
        # Getting the type of 'chart' (line 206)
        chart_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'chart')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), chart_776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_778 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), getitem___777, a_775)
        
        # Assigning a type to the variable 'subscript_call_result_778' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'subscript_call_result_778', subscript_call_result_778)
        # Testing if the for loop is going to be iterated (line 206)
        # Testing the type of a for loop iterable (line 206)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 8), subscript_call_result_778)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 206, 8), subscript_call_result_778):
            # Getting the type of the for loop variable (line 206)
            for_loop_var_779 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 8), subscript_call_result_778)
            # Assigning a type to the variable 'edge' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'edge', for_loop_var_779)
            # SSA begins for a for statement (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Obtaining an instance of the builtin type 'tuple' (line 208)
            tuple_780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 13), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 208)
            # Adding element type (line 208)
            
            # Call to exp(...): (line 208)
            # Processing the call arguments (line 208)
            
            # Getting the type of 'edge' (line 208)
            edge_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'edge', False)
            # Obtaining the member 'inside' of a type (line 208)
            inside_783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 18), edge_782, 'inside')
            # Applying the 'usub' unary operator (line 208)
            result___neg___784 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 17), 'usub', inside_783)
            
            # Processing the call keyword arguments (line 208)
            kwargs_785 = {}
            # Getting the type of 'exp' (line 208)
            exp_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 13), 'exp', False)
            # Calling exp(args, kwargs) (line 208)
            exp_call_result_786 = invoke(stypy.reporting.localization.Localization(__file__, 208, 13), exp_781, *[result___neg___784], **kwargs_785)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), tuple_780, exp_call_result_786)
            # Adding element type (line 208)
            
            # Call to exp(...): (line 208)
            # Processing the call arguments (line 208)
            
            # Getting the type of 'edge' (line 208)
            edge_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 37), 'edge', False)
            # Obtaining the member 'prob' of a type (line 208)
            prob_789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 37), edge_788, 'prob')
            # Applying the 'usub' unary operator (line 208)
            result___neg___790 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 36), 'usub', prob_789)
            
            # Processing the call keyword arguments (line 208)
            kwargs_791 = {}
            # Getting the type of 'exp' (line 208)
            exp_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'exp', False)
            # Calling exp(args, kwargs) (line 208)
            exp_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 208, 32), exp_787, *[result___neg___790], **kwargs_791)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), tuple_780, exp_call_result_792)
            
            # Getting the type of 'edge' (line 209)
            edge_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'edge')
            # Obtaining the member 'left' of a type (line 209)
            left_794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), edge_793, 'left')
            # Obtaining the member 'label' of a type (line 209)
            label_795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), left_794, 'label')
            # Testing if the type of an if condition is none (line 209)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 209, 12), label_795):
                pass
            else:
                
                # Testing the type of an if condition (line 209)
                if_condition_796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 12), label_795)
                # Assigning a type to the variable 'if_condition_796' (line 209)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'if_condition_796', if_condition_796)
                # SSA begins for if statement (line 209)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 209)
                module_type_store.open_ssa_branch('else')
                pass
                # SSA join for if statement (line 209)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'edge' (line 216)
            edge_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'edge')
            # Obtaining the member 'right' of a type (line 216)
            right_798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), edge_797, 'right')
            # Testing if the type of an if condition is none (line 216)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 216, 12), right_798):
                pass
            else:
                
                # Testing the type of an if condition (line 216)
                if_condition_799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 12), right_798)
                # Assigning a type to the variable 'if_condition_799' (line 216)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'if_condition_799', if_condition_799)
                # SSA begins for if statement (line 216)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA join for if statement (line 216)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'pprint_chart(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pprint_chart' in the type store
    # Getting the type of 'stypy_return_type' (line 200)
    stypy_return_type_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_800)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pprint_chart'
    return stypy_return_type_800

# Assigning a type to the variable 'pprint_chart' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'pprint_chart', pprint_chart)

@norecursion
def do(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'do'
    module_type_store = module_type_store.open_function_context('do', 225, 0, False)
    
    # Passed parameters checking function
    do.stypy_localization = localization
    do.stypy_type_of_self = None
    do.stypy_type_store = module_type_store
    do.stypy_function_name = 'do'
    do.stypy_param_names_list = ['sent', 'grammar']
    do.stypy_varargs_param_name = None
    do.stypy_kwargs_param_name = None
    do.stypy_call_defaults = defaults
    do.stypy_call_varargs = varargs
    do.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'do', ['sent', 'grammar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'do', localization, ['sent', 'grammar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'do(...)' code ##################

    
    # Assigning a Call to a Tuple (line 227):
    
    # Assigning a Call to a Name:
    
    # Call to parse(...): (line 227)
    # Processing the call arguments (line 227)
    
    # Call to split(...): (line 227)
    # Processing the call keyword arguments (line 227)
    kwargs_804 = {}
    # Getting the type of 'sent' (line 227)
    sent_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'sent', False)
    # Obtaining the member 'split' of a type (line 227)
    split_803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 25), sent_802, 'split')
    # Calling split(args, kwargs) (line 227)
    split_call_result_805 = invoke(stypy.reporting.localization.Localization(__file__, 227, 25), split_803, *[], **kwargs_804)
    
    # Getting the type of 'grammar' (line 227)
    grammar_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 39), 'grammar', False)
    # Getting the type of 'None' (line 227)
    None_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 48), 'None', False)
    
    # Obtaining the type of the subscript
    str_808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 67), 'str', 'S')
    # Getting the type of 'grammar' (line 227)
    grammar_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 54), 'grammar', False)
    # Obtaining the member 'toid' of a type (line 227)
    toid_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 54), grammar_809, 'toid')
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 54), toid_810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 227, 54), getitem___811, str_808)
    
    # Getting the type of 'False' (line 227)
    False_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 73), 'False', False)
    # Processing the call keyword arguments (line 227)
    kwargs_814 = {}
    # Getting the type of 'parse' (line 227)
    parse_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'parse', False)
    # Calling parse(args, kwargs) (line 227)
    parse_call_result_815 = invoke(stypy.reporting.localization.Localization(__file__, 227, 19), parse_801, *[split_call_result_805, grammar_806, None_807, subscript_call_result_812, False_813], **kwargs_814)
    
    # Assigning a type to the variable 'call_assignment_4' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'call_assignment_4', parse_call_result_815)
    
    # Assigning a Call to a Name (line 227):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    # Processing the call keyword arguments
    kwargs_819 = {}
    # Getting the type of 'call_assignment_4' (line 227)
    call_assignment_4_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'call_assignment_4', False)
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), call_assignment_4_816, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___817, *[int_818], **kwargs_819)
    
    # Assigning a type to the variable 'call_assignment_5' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'call_assignment_5', getitem___call_result_820)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'call_assignment_5' (line 227)
    call_assignment_5_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'call_assignment_5')
    # Assigning a type to the variable 'chart' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'chart', call_assignment_5_821)
    
    # Assigning a Call to a Name (line 227):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    # Processing the call keyword arguments
    kwargs_825 = {}
    # Getting the type of 'call_assignment_4' (line 227)
    call_assignment_4_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'call_assignment_4', False)
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), call_assignment_4_822, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_826 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___823, *[int_824], **kwargs_825)
    
    # Assigning a type to the variable 'call_assignment_6' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'call_assignment_6', getitem___call_result_826)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'call_assignment_6' (line 227)
    call_assignment_6_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'call_assignment_6')
    # Assigning a type to the variable 'start' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'start', call_assignment_6_827)
    
    # Call to pprint_chart(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'chart' (line 228)
    chart_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'chart', False)
    
    # Call to split(...): (line 228)
    # Processing the call keyword arguments (line 228)
    kwargs_832 = {}
    # Getting the type of 'sent' (line 228)
    sent_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'sent', False)
    # Obtaining the member 'split' of a type (line 228)
    split_831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 24), sent_830, 'split')
    # Calling split(args, kwargs) (line 228)
    split_call_result_833 = invoke(stypy.reporting.localization.Localization(__file__, 228, 24), split_831, *[], **kwargs_832)
    
    # Getting the type of 'grammar' (line 228)
    grammar_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'grammar', False)
    # Obtaining the member 'tolabel' of a type (line 228)
    tolabel_835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 38), grammar_834, 'tolabel')
    # Processing the call keyword arguments (line 228)
    kwargs_836 = {}
    # Getting the type of 'pprint_chart' (line 228)
    pprint_chart_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'pprint_chart', False)
    # Calling pprint_chart(args, kwargs) (line 228)
    pprint_chart_call_result_837 = invoke(stypy.reporting.localization.Localization(__file__, 228, 4), pprint_chart_828, *[chart_829, split_call_result_833, tolabel_835], **kwargs_836)
    
    # Getting the type of 'start' (line 229)
    start_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 7), 'start')
    # Testing if the type of an if condition is none (line 229)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 229, 4), start_838):
        pass
    else:
        
        # Testing the type of an if condition (line 229)
        if_condition_839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 4), start_838)
        # Assigning a type to the variable 'if_condition_839' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'if_condition_839', if_condition_839)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 230):
        
        # Assigning a Call to a Name:
        
        # Call to mostprobablederivation(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'chart' (line 230)
        chart_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'chart', False)
        # Getting the type of 'start' (line 230)
        start_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'start', False)
        # Getting the type of 'grammar' (line 230)
        grammar_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 52), 'grammar', False)
        # Obtaining the member 'tolabel' of a type (line 230)
        tolabel_844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 52), grammar_843, 'tolabel')
        # Processing the call keyword arguments (line 230)
        kwargs_845 = {}
        # Getting the type of 'mostprobablederivation' (line 230)
        mostprobablederivation_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'mostprobablederivation', False)
        # Calling mostprobablederivation(args, kwargs) (line 230)
        mostprobablederivation_call_result_846 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), mostprobablederivation_840, *[chart_841, start_842, tolabel_844], **kwargs_845)
        
        # Assigning a type to the variable 'call_assignment_7' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_7', mostprobablederivation_call_result_846)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
        # Processing the call keyword arguments
        kwargs_850 = {}
        # Getting the type of 'call_assignment_7' (line 230)
        call_assignment_7_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_7', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), call_assignment_7_847, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_851 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___848, *[int_849], **kwargs_850)
        
        # Assigning a type to the variable 'call_assignment_8' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_8', getitem___call_result_851)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_8' (line 230)
        call_assignment_8_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_8')
        # Assigning a type to the variable 't' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 't', call_assignment_8_852)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
        # Processing the call keyword arguments
        kwargs_856 = {}
        # Getting the type of 'call_assignment_7' (line 230)
        call_assignment_7_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_7', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), call_assignment_7_853, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_857 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___854, *[int_855], **kwargs_856)
        
        # Assigning a type to the variable 'call_assignment_9' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_9', getitem___call_result_857)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_9' (line 230)
        call_assignment_9_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_9')
        # Assigning a type to the variable 'p' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'p', call_assignment_9_858)
        
        # Call to exp(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Getting the type of 'p' (line 232)
        p_860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 13), 'p', False)
        # Applying the 'usub' unary operator (line 232)
        result___neg___861 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 12), 'usub', p_860)
        
        # Processing the call keyword arguments (line 232)
        kwargs_862 = {}
        # Getting the type of 'exp' (line 232)
        exp_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'exp', False)
        # Calling exp(args, kwargs) (line 232)
        exp_call_result_863 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), exp_859, *[result___neg___861], **kwargs_862)
        
        # SSA branch for the else part of an if statement (line 229)
        module_type_store.open_ssa_branch('else')
        pass
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'start' (line 235)
    start_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'start')
    # Getting the type of 'None' (line 235)
    None_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'None')
    # Applying the binary operator 'isnot' (line 235)
    result_is_not_866 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), 'isnot', start_864, None_865)
    
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type', result_is_not_866)
    
    # ################# End of 'do(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'do' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_867)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'do'
    return stypy_return_type_867

# Assigning a type to the variable 'do' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'do', do)

@norecursion
def read_srcg_grammar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_srcg_grammar'
    module_type_store = module_type_store.open_function_context('read_srcg_grammar', 238, 0, False)
    
    # Passed parameters checking function
    read_srcg_grammar.stypy_localization = localization
    read_srcg_grammar.stypy_type_of_self = None
    read_srcg_grammar.stypy_type_store = module_type_store
    read_srcg_grammar.stypy_function_name = 'read_srcg_grammar'
    read_srcg_grammar.stypy_param_names_list = ['rulefile', 'lexiconfile']
    read_srcg_grammar.stypy_varargs_param_name = None
    read_srcg_grammar.stypy_kwargs_param_name = None
    read_srcg_grammar.stypy_call_defaults = defaults
    read_srcg_grammar.stypy_call_varargs = varargs
    read_srcg_grammar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_srcg_grammar', ['rulefile', 'lexiconfile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_srcg_grammar', localization, ['rulefile', 'lexiconfile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_srcg_grammar(...)' code ##################

    str_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'str', ' Reads a grammar as produced by write_srcg_grammar. ')
    
    # Assigning a ListComp to a Name (line 240):
    
    # Assigning a ListComp to a Name (line 240):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to open(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'rulefile' (line 240)
    rulefile_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 64), 'rulefile', False)
    # Processing the call keyword arguments (line 240)
    kwargs_885 = {}
    # Getting the type of 'open' (line 240)
    open_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 59), 'open', False)
    # Calling open(args, kwargs) (line 240)
    open_call_result_886 = invoke(stypy.reporting.localization.Localization(__file__, 240, 59), open_883, *[rulefile_884], **kwargs_885)
    
    comprehension_887 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 14), open_call_result_886)
    # Assigning a type to the variable 'line' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'line', comprehension_887)
    
    # Call to split(...): (line 240)
    # Processing the call arguments (line 240)
    str_880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 41), 'str', '\t')
    # Processing the call keyword arguments (line 240)
    kwargs_881 = {}
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'line' (line 240)
    line_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'line', False)
    # Processing the call keyword arguments (line 240)
    kwargs_871 = {}
    # Getting the type of 'len' (line 240)
    len_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'len', False)
    # Calling len(args, kwargs) (line 240)
    len_call_result_872 = invoke(stypy.reporting.localization.Localization(__file__, 240, 20), len_869, *[line_870], **kwargs_871)
    
    int_873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 32), 'int')
    # Applying the binary operator '-' (line 240)
    result_sub_874 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 20), '-', len_call_result_872, int_873)
    
    slice_875 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 240, 14), None, result_sub_874, None)
    # Getting the type of 'line' (line 240)
    line_876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 14), line_876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_878 = invoke(stypy.reporting.localization.Localization(__file__, 240, 14), getitem___877, slice_875)
    
    # Obtaining the member 'split' of a type (line 240)
    split_879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 14), subscript_call_result_878, 'split')
    # Calling split(args, kwargs) (line 240)
    split_call_result_882 = invoke(stypy.reporting.localization.Localization(__file__, 240, 14), split_879, *[str_880], **kwargs_881)
    
    list_888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 14), list_888, split_call_result_882)
    # Assigning a type to the variable 'srules' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'srules', list_888)
    
    # Assigning a ListComp to a Name (line 241):
    
    # Assigning a ListComp to a Name (line 241):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to open(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'lexiconfile' (line 241)
    lexiconfile_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 66), 'lexiconfile', False)
    # Processing the call keyword arguments (line 241)
    kwargs_905 = {}
    # Getting the type of 'open' (line 241)
    open_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 61), 'open', False)
    # Calling open(args, kwargs) (line 241)
    open_call_result_906 = invoke(stypy.reporting.localization.Localization(__file__, 241, 61), open_903, *[lexiconfile_904], **kwargs_905)
    
    comprehension_907 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 16), open_call_result_906)
    # Assigning a type to the variable 'line' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'line', comprehension_907)
    
    # Call to split(...): (line 241)
    # Processing the call arguments (line 241)
    str_900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 43), 'str', '\t')
    # Processing the call keyword arguments (line 241)
    kwargs_901 = {}
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'line' (line 241)
    line_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'line', False)
    # Processing the call keyword arguments (line 241)
    kwargs_891 = {}
    # Getting the type of 'len' (line 241)
    len_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'len', False)
    # Calling len(args, kwargs) (line 241)
    len_call_result_892 = invoke(stypy.reporting.localization.Localization(__file__, 241, 22), len_889, *[line_890], **kwargs_891)
    
    int_893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 34), 'int')
    # Applying the binary operator '-' (line 241)
    result_sub_894 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 22), '-', len_call_result_892, int_893)
    
    slice_895 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 16), None, result_sub_894, None)
    # Getting the type of 'line' (line 241)
    line_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), line_896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_898 = invoke(stypy.reporting.localization.Localization(__file__, 241, 16), getitem___897, slice_895)
    
    # Obtaining the member 'split' of a type (line 241)
    split_899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), subscript_call_result_898, 'split')
    # Calling split(args, kwargs) (line 241)
    split_call_result_902 = invoke(stypy.reporting.localization.Localization(__file__, 241, 16), split_899, *[str_900], **kwargs_901)
    
    list_908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 16), list_908, split_call_result_902)
    # Assigning a type to the variable 'slexicon' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'slexicon', list_908)
    
    # Assigning a ListComp to a Name (line 242):
    
    # Assigning a ListComp to a Name (line 242):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'srules' (line 244)
    srules_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 45), 'srules')
    comprehension_963 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 13), srules_962)
    # Assigning a type to the variable 'a' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 13), 'a', comprehension_963)
    
    # Obtaining an instance of the builtin type 'tuple' (line 242)
    tuple_909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 242)
    # Adding element type (line 242)
    
    # Obtaining an instance of the builtin type 'tuple' (line 242)
    tuple_910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 242)
    # Adding element type (line 242)
    
    # Call to tuple(...): (line 242)
    # Processing the call arguments (line 242)
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'a' (line 242)
    a_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'a', False)
    # Processing the call keyword arguments (line 242)
    kwargs_914 = {}
    # Getting the type of 'len' (line 242)
    len_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'len', False)
    # Calling len(args, kwargs) (line 242)
    len_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 242, 24), len_912, *[a_913], **kwargs_914)
    
    int_916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 33), 'int')
    # Applying the binary operator '-' (line 242)
    result_sub_917 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 24), '-', len_call_result_915, int_916)
    
    slice_918 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 242, 21), None, result_sub_917, None)
    # Getting the type of 'a' (line 242)
    a_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 242)
    getitem___920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), a_919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 242)
    subscript_call_result_921 = invoke(stypy.reporting.localization.Localization(__file__, 242, 21), getitem___920, slice_918)
    
    # Processing the call keyword arguments (line 242)
    kwargs_922 = {}
    # Getting the type of 'tuple' (line 242)
    tuple_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 242)
    tuple_call_result_923 = invoke(stypy.reporting.localization.Localization(__file__, 242, 15), tuple_911, *[subscript_call_result_921], **kwargs_922)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 15), tuple_910, tuple_call_result_923)
    # Adding element type (line 242)
    
    # Call to tuple(...): (line 242)
    # Processing the call arguments (line 242)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 242, 44, True)
    # Calculating comprehension expression
    
    # Call to split(...): (line 243)
    # Processing the call arguments (line 243)
    str_943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 73), 'str', ',')
    # Processing the call keyword arguments (line 243)
    kwargs_944 = {}
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'a' (line 243)
    a_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 59), 'a', False)
    # Processing the call keyword arguments (line 243)
    kwargs_935 = {}
    # Getting the type of 'len' (line 243)
    len_933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 55), 'len', False)
    # Calling len(args, kwargs) (line 243)
    len_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 243, 55), len_933, *[a_934], **kwargs_935)
    
    int_937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 64), 'int')
    # Applying the binary operator '-' (line 243)
    result_sub_938 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 55), '-', len_call_result_936, int_937)
    
    # Getting the type of 'a' (line 243)
    a_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 53), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 53), a_939, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_941 = invoke(stypy.reporting.localization.Localization(__file__, 243, 53), getitem___940, result_sub_938)
    
    # Obtaining the member 'split' of a type (line 243)
    split_942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 53), subscript_call_result_941, 'split')
    # Calling split(args, kwargs) (line 243)
    split_call_result_945 = invoke(stypy.reporting.localization.Localization(__file__, 243, 53), split_942, *[str_943], **kwargs_944)
    
    comprehension_946 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 44), split_call_result_945)
    # Assigning a type to the variable 'b' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 44), 'b', comprehension_946)
    
    # Call to tuple(...): (line 242)
    # Processing the call arguments (line 242)
    
    # Call to map(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'int' (line 242)
    int_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 54), 'int', False)
    # Getting the type of 'b' (line 242)
    b_928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 59), 'b', False)
    # Processing the call keyword arguments (line 242)
    kwargs_929 = {}
    # Getting the type of 'map' (line 242)
    map_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 50), 'map', False)
    # Calling map(args, kwargs) (line 242)
    map_call_result_930 = invoke(stypy.reporting.localization.Localization(__file__, 242, 50), map_926, *[int_927, b_928], **kwargs_929)
    
    # Processing the call keyword arguments (line 242)
    kwargs_931 = {}
    # Getting the type of 'tuple' (line 242)
    tuple_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 44), 'tuple', False)
    # Calling tuple(args, kwargs) (line 242)
    tuple_call_result_932 = invoke(stypy.reporting.localization.Localization(__file__, 242, 44), tuple_925, *[map_call_result_930], **kwargs_931)
    
    list_947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 44), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 44), list_947, tuple_call_result_932)
    # Processing the call keyword arguments (line 242)
    kwargs_948 = {}
    # Getting the type of 'tuple' (line 242)
    tuple_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 38), 'tuple', False)
    # Calling tuple(args, kwargs) (line 242)
    tuple_call_result_949 = invoke(stypy.reporting.localization.Localization(__file__, 242, 38), tuple_924, *[list_947], **kwargs_948)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 15), tuple_910, tuple_call_result_949)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 14), tuple_909, tuple_910)
    # Adding element type (line 242)
    
    # Call to float(...): (line 244)
    # Processing the call arguments (line 244)
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'a' (line 244)
    a_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'a', False)
    # Processing the call keyword arguments (line 244)
    kwargs_953 = {}
    # Getting the type of 'len' (line 244)
    len_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 22), 'len', False)
    # Calling len(args, kwargs) (line 244)
    len_call_result_954 = invoke(stypy.reporting.localization.Localization(__file__, 244, 22), len_951, *[a_952], **kwargs_953)
    
    int_955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 31), 'int')
    # Applying the binary operator '-' (line 244)
    result_sub_956 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 22), '-', len_call_result_954, int_955)
    
    # Getting the type of 'a' (line 244)
    a_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 20), a_957, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_959 = invoke(stypy.reporting.localization.Localization(__file__, 244, 20), getitem___958, result_sub_956)
    
    # Processing the call keyword arguments (line 244)
    kwargs_960 = {}
    # Getting the type of 'float' (line 244)
    float_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 14), 'float', False)
    # Calling float(args, kwargs) (line 244)
    float_call_result_961 = invoke(stypy.reporting.localization.Localization(__file__, 244, 14), float_950, *[subscript_call_result_959], **kwargs_960)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 14), tuple_909, float_call_result_961)
    
    list_964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 13), list_964, tuple_909)
    # Assigning a type to the variable 'rules' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'rules', list_964)
    
    # Assigning a ListComp to a Name (line 245):
    
    # Assigning a ListComp to a Name (line 245):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'slexicon' (line 246)
    slexicon_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'slexicon')
    comprehension_1002 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), slexicon_1001)
    # Assigning a type to the variable 'a' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'a', comprehension_1002)
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    
    # Call to tuple(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'a' (line 245)
    a_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'a', False)
    # Processing the call keyword arguments (line 245)
    kwargs_970 = {}
    # Getting the type of 'len' (line 245)
    len_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'len', False)
    # Calling len(args, kwargs) (line 245)
    len_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 245, 26), len_968, *[a_969], **kwargs_970)
    
    int_972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 35), 'int')
    # Applying the binary operator '-' (line 245)
    result_sub_973 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 26), '-', len_call_result_971, int_972)
    
    slice_974 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 23), None, result_sub_973, None)
    # Getting the type of 'a' (line 245)
    a_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 23), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 23), a_975, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 245, 23), getitem___976, slice_974)
    
    # Processing the call keyword arguments (line 245)
    kwargs_978 = {}
    # Getting the type of 'tuple' (line 245)
    tuple_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 245)
    tuple_call_result_979 = invoke(stypy.reporting.localization.Localization(__file__, 245, 17), tuple_967, *[subscript_call_result_977], **kwargs_978)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), tuple_966, tuple_call_result_979)
    # Adding element type (line 245)
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'a' (line 245)
    a_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 46), 'a', False)
    # Processing the call keyword arguments (line 245)
    kwargs_982 = {}
    # Getting the type of 'len' (line 245)
    len_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 42), 'len', False)
    # Calling len(args, kwargs) (line 245)
    len_call_result_983 = invoke(stypy.reporting.localization.Localization(__file__, 245, 42), len_980, *[a_981], **kwargs_982)
    
    int_984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 51), 'int')
    # Applying the binary operator '-' (line 245)
    result_sub_985 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 42), '-', len_call_result_983, int_984)
    
    # Getting the type of 'a' (line 245)
    a_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 40), 'a')
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 40), a_986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_988 = invoke(stypy.reporting.localization.Localization(__file__, 245, 40), getitem___987, result_sub_985)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), tuple_966, subscript_call_result_988)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 16), tuple_965, tuple_966)
    # Adding element type (line 245)
    
    # Call to float(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'a' (line 245)
    a_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 68), 'a', False)
    # Processing the call keyword arguments (line 245)
    kwargs_992 = {}
    # Getting the type of 'len' (line 245)
    len_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 64), 'len', False)
    # Calling len(args, kwargs) (line 245)
    len_call_result_993 = invoke(stypy.reporting.localization.Localization(__file__, 245, 64), len_990, *[a_991], **kwargs_992)
    
    int_994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 73), 'int')
    # Applying the binary operator '-' (line 245)
    result_sub_995 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 64), '-', len_call_result_993, int_994)
    
    # Getting the type of 'a' (line 245)
    a_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 62), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 62), a_996, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_998 = invoke(stypy.reporting.localization.Localization(__file__, 245, 62), getitem___997, result_sub_995)
    
    # Processing the call keyword arguments (line 245)
    kwargs_999 = {}
    # Getting the type of 'float' (line 245)
    float_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 56), 'float', False)
    # Calling float(args, kwargs) (line 245)
    float_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 245, 56), float_989, *[subscript_call_result_998], **kwargs_999)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 16), tuple_965, float_call_result_1000)
    
    list_1003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1003, tuple_965)
    # Assigning a type to the variable 'lexicon' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'lexicon', list_1003)
    
    # Obtaining an instance of the builtin type 'tuple' (line 247)
    tuple_1004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 247)
    # Adding element type (line 247)
    # Getting the type of 'rules' (line 247)
    rules_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'rules')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 11), tuple_1004, rules_1005)
    # Adding element type (line 247)
    # Getting the type of 'lexicon' (line 247)
    lexicon_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'lexicon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 11), tuple_1004, lexicon_1006)
    
    # Assigning a type to the variable 'stypy_return_type' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type', tuple_1004)
    
    # ################# End of 'read_srcg_grammar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_srcg_grammar' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1007)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_srcg_grammar'
    return stypy_return_type_1007

# Assigning a type to the variable 'read_srcg_grammar' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'read_srcg_grammar', read_srcg_grammar)

@norecursion
def splitgrammar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'splitgrammar'
    module_type_store = module_type_store.open_function_context('splitgrammar', 250, 0, False)
    
    # Passed parameters checking function
    splitgrammar.stypy_localization = localization
    splitgrammar.stypy_type_of_self = None
    splitgrammar.stypy_type_store = module_type_store
    splitgrammar.stypy_function_name = 'splitgrammar'
    splitgrammar.stypy_param_names_list = ['grammar', 'lexicon']
    splitgrammar.stypy_varargs_param_name = None
    splitgrammar.stypy_kwargs_param_name = None
    splitgrammar.stypy_call_defaults = defaults
    splitgrammar.stypy_call_varargs = varargs
    splitgrammar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splitgrammar', ['grammar', 'lexicon'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splitgrammar', localization, ['grammar', 'lexicon'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splitgrammar(...)' code ##################

    str_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, (-1)), 'str', ' split the grammar into various lookup tables, mapping nonterminal\n    labels to numeric identifiers. Also negates log-probabilities to\n    accommodate min-heaps.\n    Can only represent ordered SRCG rules (monotone LCFRS). ')
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to list(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Call to enumerate(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    str_1012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 35), 'str', 'Epsilon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 34), list_1011, str_1012)
    # Adding element type (line 257)
    str_1013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 46), 'str', 'ROOT')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 34), list_1011, str_1013)
    
    
    # Call to sorted(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Call to set(...): (line 258)
    # Processing the call arguments (line 258)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 258, 47, True)
    # Calculating comprehension expression
    # Getting the type of 'grammar' (line 258)
    grammar_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 76), 'grammar', False)
    comprehension_1018 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 47), grammar_1017)
    # Assigning a type to the variable 'rule' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'rule', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 47), comprehension_1018))
    # Assigning a type to the variable 'yf' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'yf', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 47), comprehension_1018))
    # Assigning a type to the variable 'weight' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'weight', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 47), comprehension_1018))
    # Calculating comprehension expression
    # Getting the type of 'rule' (line 258)
    rule_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 94), 'rule', False)
    comprehension_1021 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 47), rule_1020)
    # Assigning a type to the variable 'nt' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'nt', comprehension_1021)
    # Getting the type of 'nt' (line 258)
    nt_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'nt', False)
    list_1022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 47), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 47), list_1022, nt_1016)
    # Processing the call keyword arguments (line 258)
    kwargs_1023 = {}
    # Getting the type of 'set' (line 258)
    set_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 43), 'set', False)
    # Calling set(args, kwargs) (line 258)
    set_call_result_1024 = invoke(stypy.reporting.localization.Localization(__file__, 258, 43), set_1015, *[list_1022], **kwargs_1023)
    
    
    # Call to set(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Obtaining an instance of the builtin type 'list' (line 259)
    list_1026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 259)
    # Adding element type (line 259)
    str_1027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 50), 'str', 'Epsilon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 49), list_1026, str_1027)
    # Adding element type (line 259)
    str_1028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 61), 'str', 'ROOT')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 49), list_1026, str_1028)
    
    # Processing the call keyword arguments (line 259)
    kwargs_1029 = {}
    # Getting the type of 'set' (line 259)
    set_1025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 45), 'set', False)
    # Calling set(args, kwargs) (line 259)
    set_call_result_1030 = invoke(stypy.reporting.localization.Localization(__file__, 259, 45), set_1025, *[list_1026], **kwargs_1029)
    
    # Applying the binary operator '-' (line 258)
    result_sub_1031 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 43), '-', set_call_result_1024, set_call_result_1030)
    
    # Processing the call keyword arguments (line 258)
    kwargs_1032 = {}
    # Getting the type of 'sorted' (line 258)
    sorted_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 36), 'sorted', False)
    # Calling sorted(args, kwargs) (line 258)
    sorted_call_result_1033 = invoke(stypy.reporting.localization.Localization(__file__, 258, 36), sorted_1014, *[result_sub_1031], **kwargs_1032)
    
    # Applying the binary operator '+' (line 257)
    result_add_1034 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 34), '+', list_1011, sorted_call_result_1033)
    
    # Processing the call keyword arguments (line 257)
    kwargs_1035 = {}
    # Getting the type of 'enumerate' (line 257)
    enumerate_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 24), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 257)
    enumerate_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 257, 24), enumerate_1010, *[result_add_1034], **kwargs_1035)
    
    # Processing the call keyword arguments (line 257)
    kwargs_1037 = {}
    # Getting the type of 'list' (line 257)
    list_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 'list', False)
    # Calling list(args, kwargs) (line 257)
    list_call_result_1038 = invoke(stypy.reporting.localization.Localization(__file__, 257, 19), list_1009, *[enumerate_call_result_1036], **kwargs_1037)
    
    # Assigning a type to the variable 'nonterminals' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'nonterminals', list_call_result_1038)
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 260):
    
    # Call to dict(...): (line 260)
    # Processing the call arguments (line 260)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 260, 16, True)
    # Calculating comprehension expression
    # Getting the type of 'nonterminals' (line 260)
    nonterminals_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 39), 'nonterminals', False)
    comprehension_1044 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 16), nonterminals_1043)
    # Assigning a type to the variable 'n' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 16), comprehension_1044))
    # Assigning a type to the variable 'lhs' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'lhs', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 16), comprehension_1044))
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_1040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    # Getting the type of 'lhs' (line 260)
    lhs_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'lhs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 17), tuple_1040, lhs_1041)
    # Adding element type (line 260)
    # Getting the type of 'n' (line 260)
    n_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 17), tuple_1040, n_1042)
    
    list_1045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 16), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 16), list_1045, tuple_1040)
    # Processing the call keyword arguments (line 260)
    kwargs_1046 = {}
    # Getting the type of 'dict' (line 260)
    dict_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'dict', False)
    # Calling dict(args, kwargs) (line 260)
    dict_call_result_1047 = invoke(stypy.reporting.localization.Localization(__file__, 260, 11), dict_1039, *[list_1045], **kwargs_1046)
    
    # Assigning a type to the variable 'toid' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'toid', dict_call_result_1047)
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to dict(...): (line 261)
    # Processing the call arguments (line 261)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 261, 19, True)
    # Calculating comprehension expression
    # Getting the type of 'nonterminals' (line 261)
    nonterminals_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 42), 'nonterminals', False)
    comprehension_1053 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 19), nonterminals_1052)
    # Assigning a type to the variable 'n' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 19), comprehension_1053))
    # Assigning a type to the variable 'lhs' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'lhs', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 19), comprehension_1053))
    
    # Obtaining an instance of the builtin type 'tuple' (line 261)
    tuple_1049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 261)
    # Adding element type (line 261)
    # Getting the type of 'n' (line 261)
    n_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 20), tuple_1049, n_1050)
    # Adding element type (line 261)
    # Getting the type of 'lhs' (line 261)
    lhs_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'lhs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 20), tuple_1049, lhs_1051)
    
    list_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 19), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 19), list_1054, tuple_1049)
    # Processing the call keyword arguments (line 261)
    kwargs_1055 = {}
    # Getting the type of 'dict' (line 261)
    dict_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 14), 'dict', False)
    # Calling dict(args, kwargs) (line 261)
    dict_call_result_1056 = invoke(stypy.reporting.localization.Localization(__file__, 261, 14), dict_1048, *[list_1054], **kwargs_1055)
    
    # Assigning a type to the variable 'tolabel' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'tolabel', dict_call_result_1056)
    
    # Assigning a ListComp to a Name (line 262):
    
    # Assigning a ListComp to a Name (line 262):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'nonterminals' (line 262)
    nonterminals_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'nonterminals')
    comprehension_1059 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 13), nonterminals_1058)
    # Assigning a type to the variable '_' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), '_', comprehension_1059)
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    
    list_1060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 13), list_1060, list_1057)
    # Assigning a type to the variable 'bylhs' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'bylhs', list_1060)
    
    # Assigning a ListComp to a Name (line 263):
    
    # Assigning a ListComp to a Name (line 263):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'nonterminals' (line 263)
    nonterminals_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'nonterminals')
    comprehension_1063 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 13), nonterminals_1062)
    # Assigning a type to the variable '_' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), '_', comprehension_1063)
    
    # Obtaining an instance of the builtin type 'list' (line 263)
    list_1061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 263)
    
    list_1064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 13), list_1064, list_1061)
    # Assigning a type to the variable 'unary' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'unary', list_1064)
    
    # Assigning a ListComp to a Name (line 264):
    
    # Assigning a ListComp to a Name (line 264):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'nonterminals' (line 264)
    nonterminals_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 27), 'nonterminals')
    comprehension_1067 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 15), nonterminals_1066)
    # Assigning a type to the variable '_' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), '_', comprehension_1067)
    
    # Obtaining an instance of the builtin type 'list' (line 264)
    list_1065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 264)
    
    list_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 15), list_1068, list_1065)
    # Assigning a type to the variable 'lbinary' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'lbinary', list_1068)
    
    # Assigning a ListComp to a Name (line 265):
    
    # Assigning a ListComp to a Name (line 265):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'nonterminals' (line 265)
    nonterminals_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 27), 'nonterminals')
    comprehension_1071 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 15), nonterminals_1070)
    # Assigning a type to the variable '_' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), '_', comprehension_1071)
    
    # Obtaining an instance of the builtin type 'list' (line 265)
    list_1069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 265)
    
    list_1072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 15), list_1072, list_1069)
    # Assigning a type to the variable 'rbinary' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'rbinary', list_1072)
    
    # Assigning a Dict to a Name (line 266):
    
    # Assigning a Dict to a Name (line 266):
    
    # Obtaining an instance of the builtin type 'dict' (line 266)
    dict_1073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 266)
    
    # Assigning a type to the variable 'lexical' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'lexical', dict_1073)
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to array(...): (line 267)
    # Processing the call arguments (line 267)
    str_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 18), 'str', 'B')
    
    # Obtaining an instance of the builtin type 'list' (line 267)
    list_1076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 267)
    # Adding element type (line 267)
    int_1077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 23), list_1076, int_1077)
    
    
    # Call to len(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'nonterminals' (line 267)
    nonterminals_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'nonterminals', False)
    # Processing the call keyword arguments (line 267)
    kwargs_1080 = {}
    # Getting the type of 'len' (line 267)
    len_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 29), 'len', False)
    # Calling len(args, kwargs) (line 267)
    len_call_result_1081 = invoke(stypy.reporting.localization.Localization(__file__, 267, 29), len_1078, *[nonterminals_1079], **kwargs_1080)
    
    # Applying the binary operator '*' (line 267)
    result_mul_1082 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 23), '*', list_1076, len_call_result_1081)
    
    # Processing the call keyword arguments (line 267)
    kwargs_1083 = {}
    # Getting the type of 'array' (line 267)
    array_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'array', False)
    # Calling array(args, kwargs) (line 267)
    array_call_result_1084 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), array_1074, *[str_1075, result_mul_1082], **kwargs_1083)
    
    # Assigning a type to the variable 'arity' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'arity', array_call_result_1084)
    
    # Getting the type of 'lexicon' (line 268)
    lexicon_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 26), 'lexicon')
    # Assigning a type to the variable 'lexicon_1085' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'lexicon_1085', lexicon_1085)
    # Testing if the for loop is going to be iterated (line 268)
    # Testing the type of a for loop iterable (line 268)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 268, 4), lexicon_1085)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 268, 4), lexicon_1085):
        # Getting the type of the for loop variable (line 268)
        for_loop_var_1086 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 268, 4), lexicon_1085)
        # Assigning a type to the variable 'tag' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tag', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 4), for_loop_var_1086))
        # Assigning a type to the variable 'word' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'word', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 4), for_loop_var_1086))
        # Assigning a type to the variable 'w' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 4), for_loop_var_1086))
        # SSA begins for a for statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to Terminal(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_1088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 30), 'int')
        # Getting the type of 'tag' (line 269)
        tag_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 26), 'tag', False)
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___1090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 26), tag_1089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_1091 = invoke(stypy.reporting.localization.Localization(__file__, 269, 26), getitem___1090, int_1088)
        
        # Getting the type of 'toid' (line 269)
        toid_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'toid', False)
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___1093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 21), toid_1092, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_1094 = invoke(stypy.reporting.localization.Localization(__file__, 269, 21), getitem___1093, subscript_call_result_1091)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_1095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 44), 'int')
        # Getting the type of 'tag' (line 269)
        tag_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 40), 'tag', False)
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___1097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 40), tag_1096, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_1098 = invoke(stypy.reporting.localization.Localization(__file__, 269, 40), getitem___1097, int_1095)
        
        # Getting the type of 'toid' (line 269)
        toid_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 35), 'toid', False)
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___1100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 35), toid_1099, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_1101 = invoke(stypy.reporting.localization.Localization(__file__, 269, 35), getitem___1100, subscript_call_result_1098)
        
        int_1102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 49), 'int')
        # Getting the type of 'word' (line 269)
        word_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 52), 'word', False)
        
        # Call to abs(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'w' (line 269)
        w_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 62), 'w', False)
        # Processing the call keyword arguments (line 269)
        kwargs_1106 = {}
        # Getting the type of 'abs' (line 269)
        abs_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 58), 'abs', False)
        # Calling abs(args, kwargs) (line 269)
        abs_call_result_1107 = invoke(stypy.reporting.localization.Localization(__file__, 269, 58), abs_1104, *[w_1105], **kwargs_1106)
        
        # Processing the call keyword arguments (line 269)
        kwargs_1108 = {}
        # Getting the type of 'Terminal' (line 269)
        Terminal_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'Terminal', False)
        # Calling Terminal(args, kwargs) (line 269)
        Terminal_call_result_1109 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), Terminal_1087, *[subscript_call_result_1094, subscript_call_result_1101, int_1102, word_1103, abs_call_result_1107], **kwargs_1108)
        
        # Assigning a type to the variable 't' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 't', Terminal_call_result_1109)
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 270)
        t_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 21), 't')
        # Obtaining the member 'lhs' of a type (line 270)
        lhs_1111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 21), t_1110, 'lhs')
        # Getting the type of 'arity' (line 270)
        arity_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'arity')
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___1113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 15), arity_1112, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_1114 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), getitem___1113, lhs_1111)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_1115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        int_1116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 32), tuple_1115, int_1116)
        # Adding element type (line 270)
        int_1117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 32), tuple_1115, int_1117)
        
        # Applying the binary operator 'in' (line 270)
        result_contains_1118 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 15), 'in', subscript_call_result_1114, tuple_1115)
        
        assert_1119 = result_contains_1118
        # Assigning a type to the variable 'assert_1119' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'assert_1119', result_contains_1118)
        
        # Assigning a Num to a Subscript (line 271):
        
        # Assigning a Num to a Subscript (line 271):
        int_1120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 23), 'int')
        # Getting the type of 'arity' (line 271)
        arity_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'arity')
        # Getting the type of 't' (line 271)
        t_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 't')
        # Obtaining the member 'lhs' of a type (line 271)
        lhs_1123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 14), t_1122, 'lhs')
        # Storing an element on a container (line 271)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 8), arity_1121, (lhs_1123, int_1120))
        
        # Call to append(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 't' (line 272)
        t_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 44), 't', False)
        # Processing the call keyword arguments (line 272)
        kwargs_1132 = {}
        
        # Call to setdefault(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'word' (line 272)
        word_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 27), 'word', False)
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        
        # Processing the call keyword arguments (line 272)
        kwargs_1128 = {}
        # Getting the type of 'lexical' (line 272)
        lexical_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'lexical', False)
        # Obtaining the member 'setdefault' of a type (line 272)
        setdefault_1125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), lexical_1124, 'setdefault')
        # Calling setdefault(args, kwargs) (line 272)
        setdefault_call_result_1129 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), setdefault_1125, *[word_1126, list_1127], **kwargs_1128)
        
        # Obtaining the member 'append' of a type (line 272)
        append_1130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), setdefault_call_result_1129, 'append')
        # Calling append(args, kwargs) (line 272)
        append_call_result_1133 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), append_1130, *[t_1131], **kwargs_1132)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'grammar' (line 273)
    grammar_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'grammar')
    # Assigning a type to the variable 'grammar_1134' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'grammar_1134', grammar_1134)
    # Testing if the for loop is going to be iterated (line 273)
    # Testing the type of a for loop iterable (line 273)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 273, 4), grammar_1134)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 273, 4), grammar_1134):
        # Getting the type of the for loop variable (line 273)
        for_loop_var_1135 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 273, 4), grammar_1134)
        # Assigning a type to the variable 'rule' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'rule', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), for_loop_var_1135))
        # Assigning a type to the variable 'yf' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'yf', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), for_loop_var_1135))
        # Assigning a type to the variable 'w' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 4), for_loop_var_1135))
        # SSA begins for a for statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 274):
        
        # Assigning a Call to a Name:
        
        # Call to yfarray(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'yf' (line 274)
        yf_1137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 32), 'yf', False)
        # Processing the call keyword arguments (line 274)
        kwargs_1138 = {}
        # Getting the type of 'yfarray' (line 274)
        yfarray_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'yfarray', False)
        # Calling yfarray(args, kwargs) (line 274)
        yfarray_call_result_1139 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), yfarray_1136, *[yf_1137], **kwargs_1138)
        
        # Assigning a type to the variable 'call_assignment_10' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_10', yfarray_call_result_1139)
        
        # Assigning a Call to a Name (line 274):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        # Processing the call keyword arguments
        kwargs_1143 = {}
        # Getting the type of 'call_assignment_10' (line 274)
        call_assignment_10_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_10', False)
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___1141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), call_assignment_10_1140, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1144 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1141, *[int_1142], **kwargs_1143)
        
        # Assigning a type to the variable 'call_assignment_11' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_11', getitem___call_result_1144)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'call_assignment_11' (line 274)
        call_assignment_11_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_11')
        # Assigning a type to the variable 'args' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'args', call_assignment_11_1145)
        
        # Assigning a Call to a Name (line 274):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        # Processing the call keyword arguments
        kwargs_1149 = {}
        # Getting the type of 'call_assignment_10' (line 274)
        call_assignment_10_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_10', False)
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___1147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), call_assignment_10_1146, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1150 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1147, *[int_1148], **kwargs_1149)
        
        # Assigning a type to the variable 'call_assignment_12' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_12', getitem___call_result_1150)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'call_assignment_12' (line 274)
        call_assignment_12_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_12')
        # Assigning a type to the variable 'lengths' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 14), 'lengths', call_assignment_12_1151)
        # Evaluating assert statement condition
        
        # Getting the type of 'yf' (line 275)
        yf_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'yf')
        
        # Call to arraytoyf(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'args' (line 275)
        args_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 31), 'args', False)
        # Getting the type of 'lengths' (line 275)
        lengths_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 37), 'lengths', False)
        # Processing the call keyword arguments (line 275)
        kwargs_1156 = {}
        # Getting the type of 'arraytoyf' (line 275)
        arraytoyf_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'arraytoyf', False)
        # Calling arraytoyf(args, kwargs) (line 275)
        arraytoyf_call_result_1157 = invoke(stypy.reporting.localization.Localization(__file__, 275, 21), arraytoyf_1153, *[args_1154, lengths_1155], **kwargs_1156)
        
        # Applying the binary operator '==' (line 275)
        result_eq_1158 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 15), '==', yf_1152, arraytoyf_call_result_1157)
        
        assert_1159 = result_eq_1158
        # Assigning a type to the variable 'assert_1159' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'assert_1159', result_eq_1158)
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'rule' (line 277)
        rule_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'rule', False)
        # Processing the call keyword arguments (line 277)
        kwargs_1162 = {}
        # Getting the type of 'len' (line 277)
        len_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 11), 'len', False)
        # Calling len(args, kwargs) (line 277)
        len_call_result_1163 = invoke(stypy.reporting.localization.Localization(__file__, 277, 11), len_1160, *[rule_1161], **kwargs_1162)
        
        int_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'int')
        # Applying the binary operator '==' (line 277)
        result_eq_1165 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 11), '==', len_call_result_1163, int_1164)
        
        
        # Getting the type of 'w' (line 277)
        w_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), 'w')
        float_1167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 35), 'float')
        # Applying the binary operator '==' (line 277)
        result_eq_1168 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 30), '==', w_1166, float_1167)
        
        # Applying the binary operator 'and' (line 277)
        result_and_keyword_1169 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 11), 'and', result_eq_1165, result_eq_1168)
        
        # Testing if the type of an if condition is none (line 277)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 277, 8), result_and_keyword_1169):
            pass
        else:
            
            # Testing the type of an if condition (line 277)
            if_condition_1170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 8), result_and_keyword_1169)
            # Assigning a type to the variable 'if_condition_1170' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'if_condition_1170', if_condition_1170)
            # SSA begins for if statement (line 277)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'w' (line 277)
            w_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 40), 'w')
            float_1172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 45), 'float')
            # Applying the binary operator '+=' (line 277)
            result_iadd_1173 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 40), '+=', w_1171, float_1172)
            # Assigning a type to the variable 'w' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 40), 'w', result_iadd_1173)
            
            # SSA join for if statement (line 277)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to Rule(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 27), 'int')
        # Getting the type of 'rule' (line 278)
        rule_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'rule', False)
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___1177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 22), rule_1176, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_1178 = invoke(stypy.reporting.localization.Localization(__file__, 278, 22), getitem___1177, int_1175)
        
        # Getting the type of 'toid' (line 278)
        toid_1179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'toid', False)
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___1180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 17), toid_1179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_1181 = invoke(stypy.reporting.localization.Localization(__file__, 278, 17), getitem___1180, subscript_call_result_1178)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_1182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 42), 'int')
        # Getting the type of 'rule' (line 278)
        rule_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 37), 'rule', False)
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___1184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 37), rule_1183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 278, 37), getitem___1184, int_1182)
        
        # Getting the type of 'toid' (line 278)
        toid_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'toid', False)
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___1187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 32), toid_1186, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_1188 = invoke(stypy.reporting.localization.Localization(__file__, 278, 32), getitem___1187, subscript_call_result_1185)
        
        
        
        
        # Call to len(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'rule' (line 279)
        rule_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 38), 'rule', False)
        # Processing the call keyword arguments (line 279)
        kwargs_1191 = {}
        # Getting the type of 'len' (line 279)
        len_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 34), 'len', False)
        # Calling len(args, kwargs) (line 279)
        len_call_result_1192 = invoke(stypy.reporting.localization.Localization(__file__, 279, 34), len_1189, *[rule_1190], **kwargs_1191)
        
        int_1193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 47), 'int')
        # Applying the binary operator '==' (line 279)
        result_eq_1194 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 34), '==', len_call_result_1192, int_1193)
        
        # Testing the type of an if expression (line 279)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 17), result_eq_1194)
        # SSA begins for if expression (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_1195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 27), 'int')
        # Getting the type of 'rule' (line 279)
        rule_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 22), 'rule', False)
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___1197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 22), rule_1196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_1198 = invoke(stypy.reporting.localization.Localization(__file__, 279, 22), getitem___1197, int_1195)
        
        # Getting the type of 'toid' (line 279)
        toid_1199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'toid', False)
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___1200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 17), toid_1199, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_1201 = invoke(stypy.reporting.localization.Localization(__file__, 279, 17), getitem___1200, subscript_call_result_1198)
        
        # SSA branch for the else part of an if expression (line 279)
        module_type_store.open_ssa_branch('if expression else')
        int_1202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 54), 'int')
        # SSA join for if expression (line 279)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1203 = union_type.UnionType.add(subscript_call_result_1201, int_1202)
        
        # Getting the type of 'args' (line 279)
        args_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 57), 'args', False)
        # Getting the type of 'lengths' (line 279)
        lengths_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 63), 'lengths', False)
        
        # Call to abs(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'w' (line 279)
        w_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 76), 'w', False)
        # Processing the call keyword arguments (line 279)
        kwargs_1208 = {}
        # Getting the type of 'abs' (line 279)
        abs_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 72), 'abs', False)
        # Calling abs(args, kwargs) (line 279)
        abs_call_result_1209 = invoke(stypy.reporting.localization.Localization(__file__, 279, 72), abs_1206, *[w_1207], **kwargs_1208)
        
        # Processing the call keyword arguments (line 278)
        kwargs_1210 = {}
        # Getting the type of 'Rule' (line 278)
        Rule_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'Rule', False)
        # Calling Rule(args, kwargs) (line 278)
        Rule_call_result_1211 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), Rule_1174, *[subscript_call_result_1181, subscript_call_result_1188, if_exp_1203, args_1204, lengths_1205, abs_call_result_1209], **kwargs_1210)
        
        # Assigning a type to the variable 'r' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'r', Rule_call_result_1211)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r' (line 280)
        r_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 17), 'r')
        # Obtaining the member 'lhs' of a type (line 280)
        lhs_1213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 17), r_1212, 'lhs')
        # Getting the type of 'arity' (line 280)
        arity_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'arity')
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___1215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), arity_1214, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_1216 = invoke(stypy.reporting.localization.Localization(__file__, 280, 11), getitem___1215, lhs_1213)
        
        int_1217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 27), 'int')
        # Applying the binary operator '==' (line 280)
        result_eq_1218 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 11), '==', subscript_call_result_1216, int_1217)
        
        # Testing if the type of an if condition is none (line 280)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 280, 8), result_eq_1218):
            pass
        else:
            
            # Testing the type of an if condition (line 280)
            if_condition_1219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), result_eq_1218)
            # Assigning a type to the variable 'if_condition_1219' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_1219', if_condition_1219)
            # SSA begins for if statement (line 280)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 281):
            
            # Assigning a Call to a Subscript (line 281):
            
            # Call to len(...): (line 281)
            # Processing the call arguments (line 281)
            # Getting the type of 'args' (line 281)
            args_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 31), 'args', False)
            # Processing the call keyword arguments (line 281)
            kwargs_1222 = {}
            # Getting the type of 'len' (line 281)
            len_1220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'len', False)
            # Calling len(args, kwargs) (line 281)
            len_call_result_1223 = invoke(stypy.reporting.localization.Localization(__file__, 281, 27), len_1220, *[args_1221], **kwargs_1222)
            
            # Getting the type of 'arity' (line 281)
            arity_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'arity')
            # Getting the type of 'r' (line 281)
            r_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'r')
            # Obtaining the member 'lhs' of a type (line 281)
            lhs_1226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 18), r_1225, 'lhs')
            # Storing an element on a container (line 281)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 12), arity_1224, (lhs_1226, len_call_result_1223))
            # SSA join for if statement (line 280)
            module_type_store = module_type_store.join_ssa_context()
            

        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r' (line 282)
        r_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'r')
        # Obtaining the member 'lhs' of a type (line 282)
        lhs_1228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 21), r_1227, 'lhs')
        # Getting the type of 'arity' (line 282)
        arity_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'arity')
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___1230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 15), arity_1229, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_1231 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), getitem___1230, lhs_1228)
        
        
        # Call to len(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'args' (line 282)
        args_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 35), 'args', False)
        # Processing the call keyword arguments (line 282)
        kwargs_1234 = {}
        # Getting the type of 'len' (line 282)
        len_1232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 31), 'len', False)
        # Calling len(args, kwargs) (line 282)
        len_call_result_1235 = invoke(stypy.reporting.localization.Localization(__file__, 282, 31), len_1232, *[args_1233], **kwargs_1234)
        
        # Applying the binary operator '==' (line 282)
        result_eq_1236 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), '==', subscript_call_result_1231, len_call_result_1235)
        
        assert_1237 = result_eq_1236
        # Assigning a type to the variable 'assert_1237' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'assert_1237', result_eq_1236)
        
        
        # Call to len(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'rule' (line 283)
        rule_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'rule', False)
        # Processing the call keyword arguments (line 283)
        kwargs_1240 = {}
        # Getting the type of 'len' (line 283)
        len_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'len', False)
        # Calling len(args, kwargs) (line 283)
        len_call_result_1241 = invoke(stypy.reporting.localization.Localization(__file__, 283, 11), len_1238, *[rule_1239], **kwargs_1240)
        
        int_1242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 24), 'int')
        # Applying the binary operator '==' (line 283)
        result_eq_1243 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 11), '==', len_call_result_1241, int_1242)
        
        # Testing if the type of an if condition is none (line 283)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 283, 8), result_eq_1243):
            
            
            # Call to len(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 'rule' (line 286)
            rule_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 17), 'rule', False)
            # Processing the call keyword arguments (line 286)
            kwargs_1265 = {}
            # Getting the type of 'len' (line 286)
            len_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'len', False)
            # Calling len(args, kwargs) (line 286)
            len_call_result_1266 = invoke(stypy.reporting.localization.Localization(__file__, 286, 13), len_1263, *[rule_1264], **kwargs_1265)
            
            int_1267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 26), 'int')
            # Applying the binary operator '==' (line 286)
            result_eq_1268 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 13), '==', len_call_result_1266, int_1267)
            
            # Testing if the type of an if condition is none (line 286)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 286, 13), result_eq_1268):
                
                # Call to ValueError(...): (line 291)
                # Processing the call arguments (line 291)
                str_1298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'str', 'grammar not binarized: %r')
                # Getting the type of 'r' (line 291)
                r_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 59), 'r', False)
                # Applying the binary operator '%' (line 291)
                result_mod_1300 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 29), '%', str_1298, r_1299)
                
                # Processing the call keyword arguments (line 291)
                kwargs_1301 = {}
                # Getting the type of 'ValueError' (line 291)
                ValueError_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 291)
                ValueError_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), ValueError_1297, *[result_mod_1300], **kwargs_1301)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 291, 12), ValueError_call_result_1302, 'raise parameter', BaseException)
            else:
                
                # Testing the type of an if condition (line 286)
                if_condition_1269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 13), result_eq_1268)
                # Assigning a type to the variable 'if_condition_1269' (line 286)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'if_condition_1269', if_condition_1269)
                # SSA begins for if statement (line 286)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 287)
                # Processing the call arguments (line 287)
                # Getting the type of 'r' (line 287)
                r_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'r', False)
                # Processing the call keyword arguments (line 287)
                kwargs_1277 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'r' (line 287)
                r_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'r', False)
                # Obtaining the member 'rhs1' of a type (line 287)
                rhs1_1271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 20), r_1270, 'rhs1')
                # Getting the type of 'lbinary' (line 287)
                lbinary_1272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'lbinary', False)
                # Obtaining the member '__getitem__' of a type (line 287)
                getitem___1273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), lbinary_1272, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 287)
                subscript_call_result_1274 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), getitem___1273, rhs1_1271)
                
                # Obtaining the member 'append' of a type (line 287)
                append_1275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), subscript_call_result_1274, 'append')
                # Calling append(args, kwargs) (line 287)
                append_call_result_1278 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), append_1275, *[r_1276], **kwargs_1277)
                
                
                # Call to append(...): (line 288)
                # Processing the call arguments (line 288)
                # Getting the type of 'r' (line 288)
                r_1285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 35), 'r', False)
                # Processing the call keyword arguments (line 288)
                kwargs_1286 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'r' (line 288)
                r_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 20), 'r', False)
                # Obtaining the member 'rhs2' of a type (line 288)
                rhs2_1280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 20), r_1279, 'rhs2')
                # Getting the type of 'rbinary' (line 288)
                rbinary_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'rbinary', False)
                # Obtaining the member '__getitem__' of a type (line 288)
                getitem___1282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), rbinary_1281, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 288)
                subscript_call_result_1283 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), getitem___1282, rhs2_1280)
                
                # Obtaining the member 'append' of a type (line 288)
                append_1284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), subscript_call_result_1283, 'append')
                # Calling append(args, kwargs) (line 288)
                append_call_result_1287 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), append_1284, *[r_1285], **kwargs_1286)
                
                
                # Call to append(...): (line 289)
                # Processing the call arguments (line 289)
                # Getting the type of 'r' (line 289)
                r_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'r', False)
                # Processing the call keyword arguments (line 289)
                kwargs_1295 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'r' (line 289)
                r_1288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 'r', False)
                # Obtaining the member 'lhs' of a type (line 289)
                lhs_1289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 18), r_1288, 'lhs')
                # Getting the type of 'bylhs' (line 289)
                bylhs_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'bylhs', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___1291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), bylhs_1290, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_1292 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), getitem___1291, lhs_1289)
                
                # Obtaining the member 'append' of a type (line 289)
                append_1293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), subscript_call_result_1292, 'append')
                # Calling append(args, kwargs) (line 289)
                append_call_result_1296 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), append_1293, *[r_1294], **kwargs_1295)
                
                # SSA branch for the else part of an if statement (line 286)
                module_type_store.open_ssa_branch('else')
                
                # Call to ValueError(...): (line 291)
                # Processing the call arguments (line 291)
                str_1298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'str', 'grammar not binarized: %r')
                # Getting the type of 'r' (line 291)
                r_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 59), 'r', False)
                # Applying the binary operator '%' (line 291)
                result_mod_1300 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 29), '%', str_1298, r_1299)
                
                # Processing the call keyword arguments (line 291)
                kwargs_1301 = {}
                # Getting the type of 'ValueError' (line 291)
                ValueError_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 291)
                ValueError_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), ValueError_1297, *[result_mod_1300], **kwargs_1301)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 291, 12), ValueError_call_result_1302, 'raise parameter', BaseException)
                # SSA join for if statement (line 286)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 283)
            if_condition_1244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 8), result_eq_1243)
            # Assigning a type to the variable 'if_condition_1244' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'if_condition_1244', if_condition_1244)
            # SSA begins for if statement (line 283)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 284)
            # Processing the call arguments (line 284)
            # Getting the type of 'r' (line 284)
            r_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 33), 'r', False)
            # Processing the call keyword arguments (line 284)
            kwargs_1252 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'r' (line 284)
            r_1245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'r', False)
            # Obtaining the member 'rhs1' of a type (line 284)
            rhs1_1246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 18), r_1245, 'rhs1')
            # Getting the type of 'unary' (line 284)
            unary_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'unary', False)
            # Obtaining the member '__getitem__' of a type (line 284)
            getitem___1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), unary_1247, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 284)
            subscript_call_result_1249 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), getitem___1248, rhs1_1246)
            
            # Obtaining the member 'append' of a type (line 284)
            append_1250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), subscript_call_result_1249, 'append')
            # Calling append(args, kwargs) (line 284)
            append_call_result_1253 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), append_1250, *[r_1251], **kwargs_1252)
            
            
            # Call to append(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 'r' (line 285)
            r_1260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 32), 'r', False)
            # Processing the call keyword arguments (line 285)
            kwargs_1261 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'r' (line 285)
            r_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 18), 'r', False)
            # Obtaining the member 'lhs' of a type (line 285)
            lhs_1255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 18), r_1254, 'lhs')
            # Getting the type of 'bylhs' (line 285)
            bylhs_1256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'bylhs', False)
            # Obtaining the member '__getitem__' of a type (line 285)
            getitem___1257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), bylhs_1256, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 285)
            subscript_call_result_1258 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), getitem___1257, lhs_1255)
            
            # Obtaining the member 'append' of a type (line 285)
            append_1259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), subscript_call_result_1258, 'append')
            # Calling append(args, kwargs) (line 285)
            append_call_result_1262 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), append_1259, *[r_1260], **kwargs_1261)
            
            # SSA branch for the else part of an if statement (line 283)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to len(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 'rule' (line 286)
            rule_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 17), 'rule', False)
            # Processing the call keyword arguments (line 286)
            kwargs_1265 = {}
            # Getting the type of 'len' (line 286)
            len_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'len', False)
            # Calling len(args, kwargs) (line 286)
            len_call_result_1266 = invoke(stypy.reporting.localization.Localization(__file__, 286, 13), len_1263, *[rule_1264], **kwargs_1265)
            
            int_1267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 26), 'int')
            # Applying the binary operator '==' (line 286)
            result_eq_1268 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 13), '==', len_call_result_1266, int_1267)
            
            # Testing if the type of an if condition is none (line 286)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 286, 13), result_eq_1268):
                
                # Call to ValueError(...): (line 291)
                # Processing the call arguments (line 291)
                str_1298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'str', 'grammar not binarized: %r')
                # Getting the type of 'r' (line 291)
                r_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 59), 'r', False)
                # Applying the binary operator '%' (line 291)
                result_mod_1300 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 29), '%', str_1298, r_1299)
                
                # Processing the call keyword arguments (line 291)
                kwargs_1301 = {}
                # Getting the type of 'ValueError' (line 291)
                ValueError_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 291)
                ValueError_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), ValueError_1297, *[result_mod_1300], **kwargs_1301)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 291, 12), ValueError_call_result_1302, 'raise parameter', BaseException)
            else:
                
                # Testing the type of an if condition (line 286)
                if_condition_1269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 13), result_eq_1268)
                # Assigning a type to the variable 'if_condition_1269' (line 286)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'if_condition_1269', if_condition_1269)
                # SSA begins for if statement (line 286)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 287)
                # Processing the call arguments (line 287)
                # Getting the type of 'r' (line 287)
                r_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'r', False)
                # Processing the call keyword arguments (line 287)
                kwargs_1277 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'r' (line 287)
                r_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'r', False)
                # Obtaining the member 'rhs1' of a type (line 287)
                rhs1_1271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 20), r_1270, 'rhs1')
                # Getting the type of 'lbinary' (line 287)
                lbinary_1272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'lbinary', False)
                # Obtaining the member '__getitem__' of a type (line 287)
                getitem___1273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), lbinary_1272, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 287)
                subscript_call_result_1274 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), getitem___1273, rhs1_1271)
                
                # Obtaining the member 'append' of a type (line 287)
                append_1275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), subscript_call_result_1274, 'append')
                # Calling append(args, kwargs) (line 287)
                append_call_result_1278 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), append_1275, *[r_1276], **kwargs_1277)
                
                
                # Call to append(...): (line 288)
                # Processing the call arguments (line 288)
                # Getting the type of 'r' (line 288)
                r_1285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 35), 'r', False)
                # Processing the call keyword arguments (line 288)
                kwargs_1286 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'r' (line 288)
                r_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 20), 'r', False)
                # Obtaining the member 'rhs2' of a type (line 288)
                rhs2_1280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 20), r_1279, 'rhs2')
                # Getting the type of 'rbinary' (line 288)
                rbinary_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'rbinary', False)
                # Obtaining the member '__getitem__' of a type (line 288)
                getitem___1282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), rbinary_1281, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 288)
                subscript_call_result_1283 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), getitem___1282, rhs2_1280)
                
                # Obtaining the member 'append' of a type (line 288)
                append_1284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), subscript_call_result_1283, 'append')
                # Calling append(args, kwargs) (line 288)
                append_call_result_1287 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), append_1284, *[r_1285], **kwargs_1286)
                
                
                # Call to append(...): (line 289)
                # Processing the call arguments (line 289)
                # Getting the type of 'r' (line 289)
                r_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'r', False)
                # Processing the call keyword arguments (line 289)
                kwargs_1295 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'r' (line 289)
                r_1288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 'r', False)
                # Obtaining the member 'lhs' of a type (line 289)
                lhs_1289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 18), r_1288, 'lhs')
                # Getting the type of 'bylhs' (line 289)
                bylhs_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'bylhs', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___1291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), bylhs_1290, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_1292 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), getitem___1291, lhs_1289)
                
                # Obtaining the member 'append' of a type (line 289)
                append_1293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), subscript_call_result_1292, 'append')
                # Calling append(args, kwargs) (line 289)
                append_call_result_1296 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), append_1293, *[r_1294], **kwargs_1295)
                
                # SSA branch for the else part of an if statement (line 286)
                module_type_store.open_ssa_branch('else')
                
                # Call to ValueError(...): (line 291)
                # Processing the call arguments (line 291)
                str_1298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'str', 'grammar not binarized: %r')
                # Getting the type of 'r' (line 291)
                r_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 59), 'r', False)
                # Applying the binary operator '%' (line 291)
                result_mod_1300 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 29), '%', str_1298, r_1299)
                
                # Processing the call keyword arguments (line 291)
                kwargs_1301 = {}
                # Getting the type of 'ValueError' (line 291)
                ValueError_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 291)
                ValueError_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), ValueError_1297, *[result_mod_1300], **kwargs_1301)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 291, 12), ValueError_call_result_1302, 'raise parameter', BaseException)
                # SSA join for if statement (line 286)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 283)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to Grammar(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'unary' (line 293)
    unary_1304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'unary', False)
    # Getting the type of 'lbinary' (line 293)
    lbinary_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'lbinary', False)
    # Getting the type of 'rbinary' (line 293)
    rbinary_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 35), 'rbinary', False)
    # Getting the type of 'lexical' (line 293)
    lexical_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 44), 'lexical', False)
    # Getting the type of 'bylhs' (line 293)
    bylhs_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 53), 'bylhs', False)
    # Getting the type of 'toid' (line 293)
    toid_1309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 60), 'toid', False)
    # Getting the type of 'tolabel' (line 293)
    tolabel_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 66), 'tolabel', False)
    # Processing the call keyword arguments (line 293)
    kwargs_1311 = {}
    # Getting the type of 'Grammar' (line 293)
    Grammar_1303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'Grammar', False)
    # Calling Grammar(args, kwargs) (line 293)
    Grammar_call_result_1312 = invoke(stypy.reporting.localization.Localization(__file__, 293, 11), Grammar_1303, *[unary_1304, lbinary_1305, rbinary_1306, lexical_1307, bylhs_1308, toid_1309, tolabel_1310], **kwargs_1311)
    
    # Assigning a type to the variable 'stypy_return_type' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type', Grammar_call_result_1312)
    
    # ################# End of 'splitgrammar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splitgrammar' in the type store
    # Getting the type of 'stypy_return_type' (line 250)
    stypy_return_type_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1313)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splitgrammar'
    return stypy_return_type_1313

# Assigning a type to the variable 'splitgrammar' (line 250)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'splitgrammar', splitgrammar)

@norecursion
def yfarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'yfarray'
    module_type_store = module_type_store.open_function_context('yfarray', 296, 0, False)
    
    # Passed parameters checking function
    yfarray.stypy_localization = localization
    yfarray.stypy_type_of_self = None
    yfarray.stypy_type_store = module_type_store
    yfarray.stypy_function_name = 'yfarray'
    yfarray.stypy_param_names_list = ['yf']
    yfarray.stypy_varargs_param_name = None
    yfarray.stypy_kwargs_param_name = None
    yfarray.stypy_call_defaults = defaults
    yfarray.stypy_call_varargs = varargs
    yfarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'yfarray', ['yf'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'yfarray', localization, ['yf'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'yfarray(...)' code ##################

    str_1314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, (-1)), 'str', ' convert a yield function represented as a 2D sequence to an array\n    object. ')
    
    # Assigning a Str to a Name (line 300):
    
    # Assigning a Str to a Name (line 300):
    str_1315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 14), 'str', 'I')
    # Assigning a type to the variable 'vectype' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'vectype', str_1315)
    
    # Assigning a Num to a Name (line 301):
    
    # Assigning a Num to a Name (line 301):
    int_1316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 14), 'int')
    # Assigning a type to the variable 'vecsize' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'vecsize', int_1316)
    
    # Assigning a Str to a Name (line 302):
    
    # Assigning a Str to a Name (line 302):
    str_1317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 14), 'str', 'H')
    # Assigning a type to the variable 'lentype' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'lentype', str_1317)
    
    # Assigning a Num to a Name (line 303):
    
    # Assigning a Num to a Name (line 303):
    int_1318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 14), 'int')
    # Assigning a type to the variable 'lensize' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'lensize', int_1318)
    # Evaluating assert statement condition
    
    
    # Call to len(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'yf' (line 304)
    yf_1320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'yf', False)
    # Processing the call keyword arguments (line 304)
    kwargs_1321 = {}
    # Getting the type of 'len' (line 304)
    len_1319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'len', False)
    # Calling len(args, kwargs) (line 304)
    len_call_result_1322 = invoke(stypy.reporting.localization.Localization(__file__, 304, 11), len_1319, *[yf_1320], **kwargs_1321)
    
    # Getting the type of 'lensize' (line 304)
    lensize_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'lensize')
    # Applying the binary operator '<=' (line 304)
    result_le_1324 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 11), '<=', len_call_result_1322, lensize_1323)
    
    assert_1325 = result_le_1324
    # Assigning a type to the variable 'assert_1325' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'assert_1325', result_le_1324)
    # Evaluating assert statement condition
    
    # Call to all(...): (line 305)
    # Processing the call arguments (line 305)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 305, 15, True)
    # Calculating comprehension expression
    # Getting the type of 'yf' (line 305)
    yf_1333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 42), 'yf', False)
    comprehension_1334 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 15), yf_1333)
    # Assigning a type to the variable 'a' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'a', comprehension_1334)
    
    
    # Call to len(...): (line 305)
    # Processing the call arguments (line 305)
    # Getting the type of 'a' (line 305)
    a_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'a', False)
    # Processing the call keyword arguments (line 305)
    kwargs_1329 = {}
    # Getting the type of 'len' (line 305)
    len_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'len', False)
    # Calling len(args, kwargs) (line 305)
    len_call_result_1330 = invoke(stypy.reporting.localization.Localization(__file__, 305, 15), len_1327, *[a_1328], **kwargs_1329)
    
    # Getting the type of 'vecsize' (line 305)
    vecsize_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 25), 'vecsize', False)
    # Applying the binary operator '<=' (line 305)
    result_le_1332 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 15), '<=', len_call_result_1330, vecsize_1331)
    
    list_1335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 15), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 15), list_1335, result_le_1332)
    # Processing the call keyword arguments (line 305)
    kwargs_1336 = {}
    # Getting the type of 'all' (line 305)
    all_1326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'all', False)
    # Calling all(args, kwargs) (line 305)
    all_call_result_1337 = invoke(stypy.reporting.localization.Localization(__file__, 305, 11), all_1326, *[list_1335], **kwargs_1336)
    
    assert_1338 = all_call_result_1337
    # Assigning a type to the variable 'assert_1338' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'assert_1338', all_call_result_1337)
    
    # Assigning a ListComp to a Name (line 306):
    
    # Assigning a ListComp to a Name (line 306):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'yf' (line 306)
    yf_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 70), 'yf')
    comprehension_1353 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 19), yf_1352)
    # Assigning a type to the variable 'a' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'a', comprehension_1353)
    
    # Call to sum(...): (line 306)
    # Processing the call arguments (line 306)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 306, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'a' (line 306)
    a_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 52), 'a', False)
    # Processing the call keyword arguments (line 306)
    kwargs_1346 = {}
    # Getting the type of 'enumerate' (line 306)
    enumerate_1344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 42), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 306)
    enumerate_call_result_1347 = invoke(stypy.reporting.localization.Localization(__file__, 306, 42), enumerate_1344, *[a_1345], **kwargs_1346)
    
    comprehension_1348 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 23), enumerate_call_result_1347)
    # Assigning a type to the variable 'n' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 23), comprehension_1348))
    # Assigning a type to the variable 'b' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 23), comprehension_1348))
    # Getting the type of 'b' (line 306)
    b_1343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 58), 'b', False)
    int_1340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 23), 'int')
    # Getting the type of 'n' (line 306)
    n_1341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 28), 'n', False)
    # Applying the binary operator '<<' (line 306)
    result_lshift_1342 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 23), '<<', int_1340, n_1341)
    
    list_1349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 23), list_1349, result_lshift_1342)
    # Processing the call keyword arguments (line 306)
    kwargs_1350 = {}
    # Getting the type of 'sum' (line 306)
    sum_1339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'sum', False)
    # Calling sum(args, kwargs) (line 306)
    sum_call_result_1351 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), sum_1339, *[list_1349], **kwargs_1350)
    
    list_1354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 19), list_1354, sum_call_result_1351)
    # Assigning a type to the variable 'initializer' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'initializer', list_1354)
    
    # Assigning a Call to a Name (line 307):
    
    # Assigning a Call to a Name (line 307):
    
    # Call to array(...): (line 307)
    # Processing the call arguments (line 307)
    str_1356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 17), 'str', 'I')
    # Getting the type of 'initializer' (line 307)
    initializer_1357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'initializer', False)
    # Processing the call keyword arguments (line 307)
    kwargs_1358 = {}
    # Getting the type of 'array' (line 307)
    array_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'array', False)
    # Calling array(args, kwargs) (line 307)
    array_call_result_1359 = invoke(stypy.reporting.localization.Localization(__file__, 307, 11), array_1355, *[str_1356, initializer_1357], **kwargs_1358)
    
    # Assigning a type to the variable 'args' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'args', array_call_result_1359)
    
    # Assigning a Call to a Name (line 308):
    
    # Assigning a Call to a Name (line 308):
    
    # Call to array(...): (line 308)
    # Processing the call arguments (line 308)
    str_1361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 20), 'str', 'H')
    
    # Call to map(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'len' (line 308)
    len_1363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'len', False)
    # Getting the type of 'yf' (line 308)
    yf_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 34), 'yf', False)
    # Processing the call keyword arguments (line 308)
    kwargs_1365 = {}
    # Getting the type of 'map' (line 308)
    map_1362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'map', False)
    # Calling map(args, kwargs) (line 308)
    map_call_result_1366 = invoke(stypy.reporting.localization.Localization(__file__, 308, 25), map_1362, *[len_1363, yf_1364], **kwargs_1365)
    
    # Processing the call keyword arguments (line 308)
    kwargs_1367 = {}
    # Getting the type of 'array' (line 308)
    array_1360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 14), 'array', False)
    # Calling array(args, kwargs) (line 308)
    array_call_result_1368 = invoke(stypy.reporting.localization.Localization(__file__, 308, 14), array_1360, *[str_1361, map_call_result_1366], **kwargs_1367)
    
    # Assigning a type to the variable 'lengths' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'lengths', array_call_result_1368)
    
    # Obtaining an instance of the builtin type 'tuple' (line 309)
    tuple_1369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 309)
    # Adding element type (line 309)
    # Getting the type of 'args' (line 309)
    args_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 11), 'args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 11), tuple_1369, args_1370)
    # Adding element type (line 309)
    # Getting the type of 'lengths' (line 309)
    lengths_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 17), 'lengths')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 11), tuple_1369, lengths_1371)
    
    # Assigning a type to the variable 'stypy_return_type' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type', tuple_1369)
    
    # ################# End of 'yfarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'yfarray' in the type store
    # Getting the type of 'stypy_return_type' (line 296)
    stypy_return_type_1372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'yfarray'
    return stypy_return_type_1372

# Assigning a type to the variable 'yfarray' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'yfarray', yfarray)

@norecursion
def arraytoyf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arraytoyf'
    module_type_store = module_type_store.open_function_context('arraytoyf', 312, 0, False)
    
    # Passed parameters checking function
    arraytoyf.stypy_localization = localization
    arraytoyf.stypy_type_of_self = None
    arraytoyf.stypy_type_store = module_type_store
    arraytoyf.stypy_function_name = 'arraytoyf'
    arraytoyf.stypy_param_names_list = ['args', 'lengths']
    arraytoyf.stypy_varargs_param_name = None
    arraytoyf.stypy_kwargs_param_name = None
    arraytoyf.stypy_call_defaults = defaults
    arraytoyf.stypy_call_varargs = varargs
    arraytoyf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arraytoyf', ['args', 'lengths'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arraytoyf', localization, ['args', 'lengths'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arraytoyf(...)' code ##################

    
    # Call to tuple(...): (line 313)
    # Processing the call arguments (line 313)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 313, 17, True)
    # Calculating comprehension expression
    
    # Call to zip(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'lengths' (line 314)
    lengths_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 33), 'lengths', False)
    # Getting the type of 'args' (line 314)
    args_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 42), 'args', False)
    # Processing the call keyword arguments (line 314)
    kwargs_1394 = {}
    # Getting the type of 'zip' (line 314)
    zip_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 314)
    zip_call_result_1395 = invoke(stypy.reporting.localization.Localization(__file__, 314, 29), zip_1391, *[lengths_1392, args_1393], **kwargs_1394)
    
    comprehension_1396 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 17), zip_call_result_1395)
    # Assigning a type to the variable 'n' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 17), comprehension_1396))
    # Assigning a type to the variable 'a' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 17), comprehension_1396))
    
    # Call to tuple(...): (line 313)
    # Processing the call arguments (line 313)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 313, 23, True)
    # Calculating comprehension expression
    
    # Call to range(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'n' (line 313)
    n_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 63), 'n', False)
    # Processing the call keyword arguments (line 313)
    kwargs_1385 = {}
    # Getting the type of 'range' (line 313)
    range_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 57), 'range', False)
    # Calling range(args, kwargs) (line 313)
    range_call_result_1386 = invoke(stypy.reporting.localization.Localization(__file__, 313, 57), range_1383, *[n_1384], **kwargs_1385)
    
    comprehension_1387 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), range_call_result_1386)
    # Assigning a type to the variable 'm' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'm', comprehension_1387)
    
    # Getting the type of 'a' (line 313)
    a_1375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'a', False)
    int_1376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 33), 'int')
    # Getting the type of 'm' (line 313)
    m_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 38), 'm', False)
    # Applying the binary operator '<<' (line 313)
    result_lshift_1378 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 33), '<<', int_1376, m_1377)
    
    # Applying the binary operator '&' (line 313)
    result_and__1379 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 28), '&', a_1375, result_lshift_1378)
    
    # Testing the type of an if expression (line 313)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 23), result_and__1379)
    # SSA begins for if expression (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    int_1380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'int')
    # SSA branch for the else part of an if expression (line 313)
    module_type_store.open_ssa_branch('if expression else')
    int_1381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 46), 'int')
    # SSA join for if expression (line 313)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_1382 = union_type.UnionType.add(int_1380, int_1381)
    
    list_1388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_1388, if_exp_1382)
    # Processing the call keyword arguments (line 313)
    kwargs_1389 = {}
    # Getting the type of 'tuple' (line 313)
    tuple_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 313)
    tuple_call_result_1390 = invoke(stypy.reporting.localization.Localization(__file__, 313, 17), tuple_1374, *[list_1388], **kwargs_1389)
    
    list_1397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 17), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 17), list_1397, tuple_call_result_1390)
    # Processing the call keyword arguments (line 313)
    kwargs_1398 = {}
    # Getting the type of 'tuple' (line 313)
    tuple_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 313)
    tuple_call_result_1399 = invoke(stypy.reporting.localization.Localization(__file__, 313, 11), tuple_1373, *[list_1397], **kwargs_1398)
    
    # Assigning a type to the variable 'stypy_return_type' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'stypy_return_type', tuple_call_result_1399)
    
    # ################# End of 'arraytoyf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arraytoyf' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1400)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arraytoyf'
    return stypy_return_type_1400

# Assigning a type to the variable 'arraytoyf' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'arraytoyf', arraytoyf)

@norecursion
def nextset(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nextset'
    module_type_store = module_type_store.open_function_context('nextset', 318, 0, False)
    
    # Passed parameters checking function
    nextset.stypy_localization = localization
    nextset.stypy_type_of_self = None
    nextset.stypy_type_store = module_type_store
    nextset.stypy_function_name = 'nextset'
    nextset.stypy_param_names_list = ['a', 'pos']
    nextset.stypy_varargs_param_name = None
    nextset.stypy_kwargs_param_name = None
    nextset.stypy_call_defaults = defaults
    nextset.stypy_call_varargs = varargs
    nextset.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nextset', ['a', 'pos'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nextset', localization, ['a', 'pos'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nextset(...)' code ##################

    str_1401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 4), 'str', ' First set bit, starting from pos ')
    
    # Assigning a Name to a Name (line 320):
    
    # Assigning a Name to a Name (line 320):
    # Getting the type of 'pos' (line 320)
    pos_1402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'pos')
    # Assigning a type to the variable 'result' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'result', pos_1402)
    # Getting the type of 'a' (line 321)
    a_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 7), 'a')
    # Getting the type of 'result' (line 321)
    result_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'result')
    # Applying the binary operator '>>' (line 321)
    result_rshift_1405 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 7), '>>', a_1403, result_1404)
    
    # Testing if the type of an if condition is none (line 321)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 321, 4), result_rshift_1405):
        pass
    else:
        
        # Testing the type of an if condition (line 321)
        if_condition_1406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 4), result_rshift_1405)
        # Assigning a type to the variable 'if_condition_1406' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'if_condition_1406', if_condition_1406)
        # SSA begins for if statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'a' (line 322)
        a_1407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'a')
        # Getting the type of 'result' (line 322)
        result_1408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'result')
        # Applying the binary operator '>>' (line 322)
        result_rshift_1409 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), '>>', a_1407, result_1408)
        
        int_1410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 30), 'int')
        # Applying the binary operator '&' (line 322)
        result_and__1411 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 14), '&', result_rshift_1409, int_1410)
        
        int_1412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 35), 'int')
        # Applying the binary operator '==' (line 322)
        result_eq_1413 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 14), '==', result_and__1411, int_1412)
        
        # Assigning a type to the variable 'result_eq_1413' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'result_eq_1413', result_eq_1413)
        # Testing if the while is going to be iterated (line 322)
        # Testing the type of an if condition (line 322)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 8), result_eq_1413)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 322, 8), result_eq_1413):
            # SSA begins for while statement (line 322)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 'result' (line 323)
            result_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'result')
            int_1415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 22), 'int')
            # Applying the binary operator '+=' (line 323)
            result_iadd_1416 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 12), '+=', result_1414, int_1415)
            # Assigning a type to the variable 'result' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'result', result_iadd_1416)
            
            # SSA join for while statement (line 322)
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'result' (line 324)
        result_1417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'stypy_return_type', result_1417)
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        

    int_1418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type', int_1418)
    
    # ################# End of 'nextset(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nextset' in the type store
    # Getting the type of 'stypy_return_type' (line 318)
    stypy_return_type_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1419)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nextset'
    return stypy_return_type_1419

# Assigning a type to the variable 'nextset' (line 318)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'nextset', nextset)

@norecursion
def nextunset(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nextunset'
    module_type_store = module_type_store.open_function_context('nextunset', 328, 0, False)
    
    # Passed parameters checking function
    nextunset.stypy_localization = localization
    nextunset.stypy_type_of_self = None
    nextunset.stypy_type_store = module_type_store
    nextunset.stypy_function_name = 'nextunset'
    nextunset.stypy_param_names_list = ['a', 'pos']
    nextunset.stypy_varargs_param_name = None
    nextunset.stypy_kwargs_param_name = None
    nextunset.stypy_call_defaults = defaults
    nextunset.stypy_call_varargs = varargs
    nextunset.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nextunset', ['a', 'pos'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nextunset', localization, ['a', 'pos'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nextunset(...)' code ##################

    str_1420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 4), 'str', ' First unset bit, starting from pos ')
    
    # Assigning a Name to a Name (line 330):
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'pos' (line 330)
    pos_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'pos')
    # Assigning a type to the variable 'result' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'result', pos_1421)
    
    # Getting the type of 'a' (line 331)
    a_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'a')
    # Getting the type of 'result' (line 331)
    result_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'result')
    # Applying the binary operator '>>' (line 331)
    result_rshift_1424 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), '>>', a_1422, result_1423)
    
    int_1425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 26), 'int')
    # Applying the binary operator '&' (line 331)
    result_and__1426 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 10), '&', result_rshift_1424, int_1425)
    
    # Assigning a type to the variable 'result_and__1426' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'result_and__1426', result_and__1426)
    # Testing if the while is going to be iterated (line 331)
    # Testing the type of an if condition (line 331)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 4), result_and__1426)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 331, 4), result_and__1426):
        # SSA begins for while statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'result' (line 332)
        result_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'result')
        int_1428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 18), 'int')
        # Applying the binary operator '+=' (line 332)
        result_iadd_1429 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 8), '+=', result_1427, int_1428)
        # Assigning a type to the variable 'result' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'result', result_iadd_1429)
        
        # SSA join for while statement (line 331)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'result' (line 333)
    result_1430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type', result_1430)
    
    # ################# End of 'nextunset(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nextunset' in the type store
    # Getting the type of 'stypy_return_type' (line 328)
    stypy_return_type_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1431)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nextunset'
    return stypy_return_type_1431

# Assigning a type to the variable 'nextunset' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'nextunset', nextunset)

@norecursion
def bitcount(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bitcount'
    module_type_store = module_type_store.open_function_context('bitcount', 336, 0, False)
    
    # Passed parameters checking function
    bitcount.stypy_localization = localization
    bitcount.stypy_type_of_self = None
    bitcount.stypy_type_store = module_type_store
    bitcount.stypy_function_name = 'bitcount'
    bitcount.stypy_param_names_list = ['a']
    bitcount.stypy_varargs_param_name = None
    bitcount.stypy_kwargs_param_name = None
    bitcount.stypy_call_defaults = defaults
    bitcount.stypy_call_varargs = varargs
    bitcount.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bitcount', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bitcount', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bitcount(...)' code ##################

    str_1432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 4), 'str', ' Number of set bits (1s) ')
    
    # Assigning a Num to a Name (line 338):
    
    # Assigning a Num to a Name (line 338):
    int_1433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 12), 'int')
    # Assigning a type to the variable 'count' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'count', int_1433)
    
    # Getting the type of 'a' (line 339)
    a_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 10), 'a')
    # Assigning a type to the variable 'a_1434' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'a_1434', a_1434)
    # Testing if the while is going to be iterated (line 339)
    # Testing the type of an if condition (line 339)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 339, 4), a_1434)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 339, 4), a_1434):
        # SSA begins for while statement (line 339)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'a' (line 340)
        a_1435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'a')
        # Getting the type of 'a' (line 340)
        a_1436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'a')
        int_1437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 17), 'int')
        # Applying the binary operator '-' (line 340)
        result_sub_1438 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 13), '-', a_1436, int_1437)
        
        # Applying the binary operator '&=' (line 340)
        result_iand_1439 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 8), '&=', a_1435, result_sub_1438)
        # Assigning a type to the variable 'a' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'a', result_iand_1439)
        
        
        # Getting the type of 'count' (line 341)
        count_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'count')
        int_1441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 17), 'int')
        # Applying the binary operator '+=' (line 341)
        result_iadd_1442 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 8), '+=', count_1440, int_1441)
        # Assigning a type to the variable 'count' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'count', result_iadd_1442)
        
        # SSA join for while statement (line 339)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'count' (line 342)
    count_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 11), 'count')
    # Assigning a type to the variable 'stypy_return_type' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type', count_1443)
    
    # ################# End of 'bitcount(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bitcount' in the type store
    # Getting the type of 'stypy_return_type' (line 336)
    stypy_return_type_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1444)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bitcount'
    return stypy_return_type_1444

# Assigning a type to the variable 'bitcount' (line 336)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 0), 'bitcount', bitcount)

@norecursion
def testbit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'testbit'
    module_type_store = module_type_store.open_function_context('testbit', 345, 0, False)
    
    # Passed parameters checking function
    testbit.stypy_localization = localization
    testbit.stypy_type_of_self = None
    testbit.stypy_type_store = module_type_store
    testbit.stypy_function_name = 'testbit'
    testbit.stypy_param_names_list = ['a', 'offset']
    testbit.stypy_varargs_param_name = None
    testbit.stypy_kwargs_param_name = None
    testbit.stypy_call_defaults = defaults
    testbit.stypy_call_varargs = varargs
    testbit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'testbit', ['a', 'offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'testbit', localization, ['a', 'offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'testbit(...)' code ##################

    str_1445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 4), 'str', ' Mask a particular bit, return nonzero if set ')
    # Getting the type of 'a' (line 347)
    a_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'a')
    int_1447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 16), 'int')
    # Getting the type of 'offset' (line 347)
    offset_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 21), 'offset')
    # Applying the binary operator '<<' (line 347)
    result_lshift_1449 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 16), '<<', int_1447, offset_1448)
    
    # Applying the binary operator '&' (line 347)
    result_and__1450 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 11), '&', a_1446, result_lshift_1449)
    
    # Assigning a type to the variable 'stypy_return_type' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type', result_and__1450)
    
    # ################# End of 'testbit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'testbit' in the type store
    # Getting the type of 'stypy_return_type' (line 345)
    stypy_return_type_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1451)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'testbit'
    return stypy_return_type_1451

# Assigning a type to the variable 'testbit' (line 345)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'testbit', testbit)
# Declaration of the 'Grammar' class

class Grammar(object, ):
    
    # Assigning a Tuple to a Name (line 352):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Grammar.__init__', ['unary', 'lbinary', 'rbinary', 'lexical', 'bylhs', 'toid', 'tolabel'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['unary', 'lbinary', 'rbinary', 'lexical', 'bylhs', 'toid', 'tolabel'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 356):
        
        # Assigning a Name to a Attribute (line 356):
        # Getting the type of 'unary' (line 356)
        unary_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'unary')
        # Getting the type of 'self' (line 356)
        self_1453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'self')
        # Setting the type of the member 'unary' of a type (line 356)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), self_1453, 'unary', unary_1452)
        
        # Assigning a Name to a Attribute (line 357):
        
        # Assigning a Name to a Attribute (line 357):
        # Getting the type of 'lbinary' (line 357)
        lbinary_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'lbinary')
        # Getting the type of 'self' (line 357)
        self_1455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'self')
        # Setting the type of the member 'lbinary' of a type (line 357)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), self_1455, 'lbinary', lbinary_1454)
        
        # Assigning a Name to a Attribute (line 358):
        
        # Assigning a Name to a Attribute (line 358):
        # Getting the type of 'rbinary' (line 358)
        rbinary_1456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'rbinary')
        # Getting the type of 'self' (line 358)
        self_1457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'self')
        # Setting the type of the member 'rbinary' of a type (line 358)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), self_1457, 'rbinary', rbinary_1456)
        
        # Assigning a Name to a Attribute (line 359):
        
        # Assigning a Name to a Attribute (line 359):
        # Getting the type of 'lexical' (line 359)
        lexical_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 23), 'lexical')
        # Getting the type of 'self' (line 359)
        self_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self')
        # Setting the type of the member 'lexical' of a type (line 359)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), self_1459, 'lexical', lexical_1458)
        
        # Assigning a Name to a Attribute (line 360):
        
        # Assigning a Name to a Attribute (line 360):
        # Getting the type of 'bylhs' (line 360)
        bylhs_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 21), 'bylhs')
        # Getting the type of 'self' (line 360)
        self_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'self')
        # Setting the type of the member 'bylhs' of a type (line 360)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), self_1461, 'bylhs', bylhs_1460)
        
        # Assigning a Name to a Attribute (line 361):
        
        # Assigning a Name to a Attribute (line 361):
        # Getting the type of 'toid' (line 361)
        toid_1462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'toid')
        # Getting the type of 'self' (line 361)
        self_1463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self')
        # Setting the type of the member 'toid' of a type (line 361)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_1463, 'toid', toid_1462)
        
        # Assigning a Name to a Attribute (line 362):
        
        # Assigning a Name to a Attribute (line 362):
        # Getting the type of 'tolabel' (line 362)
        tolabel_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'tolabel')
        # Getting the type of 'self' (line 362)
        self_1465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'self')
        # Setting the type of the member 'tolabel' of a type (line 362)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), self_1465, 'tolabel', tolabel_1464)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Grammar' (line 351)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 0), 'Grammar', Grammar)

# Assigning a Tuple to a Name (line 352):

# Obtaining an instance of the builtin type 'tuple' (line 352)
tuple_1466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 352)
# Adding element type (line 352)
str_1467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 17), 'str', 'unary')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), tuple_1466, str_1467)
# Adding element type (line 352)
str_1468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 26), 'str', 'lbinary')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), tuple_1466, str_1468)
# Adding element type (line 352)
str_1469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 37), 'str', 'rbinary')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), tuple_1466, str_1469)
# Adding element type (line 352)
str_1470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 48), 'str', 'lexical')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), tuple_1466, str_1470)
# Adding element type (line 352)
str_1471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 17), 'str', 'bylhs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), tuple_1466, str_1471)
# Adding element type (line 352)
str_1472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 26), 'str', 'toid')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), tuple_1466, str_1472)
# Adding element type (line 352)
str_1473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 34), 'str', 'tolabel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), tuple_1466, str_1473)

# Getting the type of 'Grammar'
Grammar_1474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Grammar')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Grammar_1474, '__slots__', tuple_1466)
# Declaration of the 'ChartItem' class

class ChartItem:
    
    # Assigning a Tuple to a Name (line 366):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ChartItem.__init__', ['label', 'vec'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['label', 'vec'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 369):
        
        # Assigning a Name to a Attribute (line 369):
        # Getting the type of 'label' (line 369)
        label_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'label')
        # Getting the type of 'self' (line 369)
        self_1476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self')
        # Setting the type of the member 'label' of a type (line 369)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_1476, 'label', label_1475)
        
        # Assigning a Name to a Attribute (line 370):
        
        # Assigning a Name to a Attribute (line 370):
        # Getting the type of 'vec' (line 370)
        vec_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 'vec')
        # Getting the type of 'self' (line 370)
        self_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'self')
        # Setting the type of the member 'vec' of a type (line 370)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), self_1478, 'vec', vec_1477)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__hash__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hash__'
        module_type_store = module_type_store.open_function_context('__hash__', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_localization', localization)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_function_name', 'ChartItem.stypy__hash__')
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_param_names_list', [])
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ChartItem.stypy__hash__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ChartItem.stypy__hash__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hash__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hash__(...)' code ##################

        
        # Assigning a BinOp to a Name (line 376):
        
        # Assigning a BinOp to a Name (line 376):
        int_1479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 14), 'int')
        int_1480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 22), 'int')
        # Applying the binary operator '<<' (line 376)
        result_lshift_1481 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 14), '<<', int_1479, int_1480)
        
        int_1482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 27), 'int')
        # Applying the binary operator '+' (line 376)
        result_add_1483 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 13), '+', result_lshift_1481, int_1482)
        
        int_1484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 35), 'int')
        # Applying the binary operator '*' (line 376)
        result_mul_1485 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 12), '*', result_add_1483, int_1484)
        
        # Getting the type of 'self' (line 376)
        self_1486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 40), 'self')
        # Obtaining the member 'label' of a type (line 376)
        label_1487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 40), self_1486, 'label')
        # Applying the binary operator '^' (line 376)
        result_xor_1488 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 12), '^', result_mul_1485, label_1487)
        
        # Assigning a type to the variable 'h' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'h', result_xor_1488)
        
        # Assigning a BinOp to a Name (line 377):
        
        # Assigning a BinOp to a Name (line 377):
        # Getting the type of 'h' (line 377)
        h_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 14), 'h')
        int_1490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 19), 'int')
        # Applying the binary operator '<<' (line 377)
        result_lshift_1491 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 14), '<<', h_1489, int_1490)
        
        # Getting the type of 'h' (line 377)
        h_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 24), 'h')
        # Applying the binary operator '+' (line 377)
        result_add_1493 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 13), '+', result_lshift_1491, h_1492)
        
        int_1494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 29), 'int')
        # Applying the binary operator '*' (line 377)
        result_mul_1495 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 12), '*', result_add_1493, int_1494)
        
        # Getting the type of 'self' (line 377)
        self_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 34), 'self')
        # Obtaining the member 'vec' of a type (line 377)
        vec_1497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 34), self_1496, 'vec')
        # Applying the binary operator '^' (line 377)
        result_xor_1498 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 12), '^', result_mul_1495, vec_1497)
        
        # Assigning a type to the variable 'h' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'h', result_xor_1498)
        
        
        # Getting the type of 'h' (line 378)
        h_1499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 21), 'h')
        int_1500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 26), 'int')
        # Applying the binary operator '==' (line 378)
        result_eq_1501 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 21), '==', h_1499, int_1500)
        
        # Testing the type of an if expression (line 378)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 15), result_eq_1501)
        # SSA begins for if expression (line 378)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_1502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 15), 'int')
        # SSA branch for the else part of an if expression (line 378)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'h' (line 378)
        h_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 34), 'h')
        # SSA join for if expression (line 378)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1504 = union_type.UnionType.add(int_1502, h_1503)
        
        # Assigning a type to the variable 'stypy_return_type' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'stypy_return_type', if_exp_1504)
        
        # ################# End of '__hash__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hash__' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_1505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hash__'
        return stypy_return_type_1505


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 380, 4, False)
        # Assigning a type to the variable 'self' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'ChartItem.stypy__eq__')
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ChartItem.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ChartItem.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 381)
        # Getting the type of 'other' (line 381)
        other_1506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'other')
        # Getting the type of 'None' (line 381)
        None_1507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'None')
        
        (may_be_1508, more_types_in_union_1509) = may_be_none(other_1506, None_1507)

        if may_be_1508:

            if more_types_in_union_1509:
                # Runtime conditional SSA (line 381)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'False' (line 381)
            False_1510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 33), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 381)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'stypy_return_type', False_1510)

            if more_types_in_union_1509:
                # SSA join for if statement (line 381)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 382)
        self_1511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'self')
        # Obtaining the member 'label' of a type (line 382)
        label_1512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 15), self_1511, 'label')
        # Getting the type of 'other' (line 382)
        other_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 29), 'other')
        # Obtaining the member 'label' of a type (line 382)
        label_1514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 29), other_1513, 'label')
        # Applying the binary operator '==' (line 382)
        result_eq_1515 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 15), '==', label_1512, label_1514)
        
        
        # Getting the type of 'self' (line 382)
        self_1516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 45), 'self')
        # Obtaining the member 'vec' of a type (line 382)
        vec_1517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 45), self_1516, 'vec')
        # Getting the type of 'other' (line 382)
        other_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 57), 'other')
        # Obtaining the member 'vec' of a type (line 382)
        vec_1519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 57), other_1518, 'vec')
        # Applying the binary operator '==' (line 382)
        result_eq_1520 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 45), '==', vec_1517, vec_1519)
        
        # Applying the binary operator 'and' (line 382)
        result_and_keyword_1521 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 15), 'and', result_eq_1515, result_eq_1520)
        
        # Assigning a type to the variable 'stypy_return_type' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'stypy_return_type', result_and_keyword_1521)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 380)
        stypy_return_type_1522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_1522


# Assigning a type to the variable 'ChartItem' (line 365)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'ChartItem', ChartItem)

# Assigning a Tuple to a Name (line 366):

# Obtaining an instance of the builtin type 'tuple' (line 366)
tuple_1523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 366)
# Adding element type (line 366)
str_1524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 17), 'str', 'label')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 17), tuple_1523, str_1524)
# Adding element type (line 366)
str_1525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 26), 'str', 'vec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 17), tuple_1523, str_1525)

# Getting the type of 'ChartItem'
ChartItem_1526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ChartItem')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ChartItem_1526, '__slots__', tuple_1523)
# Declaration of the 'Edge' class

class Edge:
    
    # Assigning a Tuple to a Name (line 386):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Edge.__init__', ['score', 'inside', 'prob', 'left', 'right'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['score', 'inside', 'prob', 'left', 'right'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 389):
        
        # Assigning a Name to a Attribute (line 389):
        # Getting the type of 'score' (line 389)
        score_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), 'score')
        # Getting the type of 'self' (line 389)
        self_1528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self')
        # Setting the type of the member 'score' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_1528, 'score', score_1527)
        
        # Assigning a Name to a Attribute (line 390):
        
        # Assigning a Name to a Attribute (line 390):
        # Getting the type of 'inside' (line 390)
        inside_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 22), 'inside')
        # Getting the type of 'self' (line 390)
        self_1530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        # Setting the type of the member 'inside' of a type (line 390)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_1530, 'inside', inside_1529)
        
        # Assigning a Name to a Attribute (line 391):
        
        # Assigning a Name to a Attribute (line 391):
        # Getting the type of 'prob' (line 391)
        prob_1531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'prob')
        # Getting the type of 'self' (line 391)
        self_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self')
        # Setting the type of the member 'prob' of a type (line 391)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_1532, 'prob', prob_1531)
        
        # Assigning a Name to a Attribute (line 392):
        
        # Assigning a Name to a Attribute (line 392):
        # Getting the type of 'left' (line 392)
        left_1533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 20), 'left')
        # Getting the type of 'self' (line 392)
        self_1534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self')
        # Setting the type of the member 'left' of a type (line 392)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_1534, 'left', left_1533)
        
        # Assigning a Name to a Attribute (line 393):
        
        # Assigning a Name to a Attribute (line 393):
        # Getting the type of 'right' (line 393)
        right_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 21), 'right')
        # Getting the type of 'self' (line 393)
        self_1536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'self')
        # Setting the type of the member 'right' of a type (line 393)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), self_1536, 'right', right_1535)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 395, 4, False)
        # Assigning a type to the variable 'self' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Edge.__lt__.__dict__.__setitem__('stypy_localization', localization)
        Edge.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Edge.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Edge.__lt__.__dict__.__setitem__('stypy_function_name', 'Edge.__lt__')
        Edge.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Edge.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Edge.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Edge.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Edge.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Edge.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Edge.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Edge.__lt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lt__(...)' code ##################

        
        # Getting the type of 'self' (line 398)
        self_1537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'self')
        # Obtaining the member 'score' of a type (line 398)
        score_1538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 15), self_1537, 'score')
        # Getting the type of 'other' (line 398)
        other_1539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'other')
        # Obtaining the member 'score' of a type (line 398)
        score_1540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 28), other_1539, 'score')
        # Applying the binary operator '<' (line 398)
        result_lt_1541 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 15), '<', score_1538, score_1540)
        
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'stypy_return_type', result_lt_1541)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 395)
        stypy_return_type_1542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_1542


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 400, 4, False)
        # Assigning a type to the variable 'self' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Edge.__gt__.__dict__.__setitem__('stypy_localization', localization)
        Edge.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Edge.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Edge.__gt__.__dict__.__setitem__('stypy_function_name', 'Edge.__gt__')
        Edge.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Edge.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Edge.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Edge.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Edge.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Edge.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Edge.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Edge.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        
        # Getting the type of 'self' (line 401)
        self_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'self')
        # Obtaining the member 'score' of a type (line 401)
        score_1544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 15), self_1543, 'score')
        # Getting the type of 'other' (line 401)
        other_1545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'other')
        # Obtaining the member 'score' of a type (line 401)
        score_1546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 28), other_1545, 'score')
        # Applying the binary operator '>' (line 401)
        result_gt_1547 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 15), '>', score_1544, score_1546)
        
        # Assigning a type to the variable 'stypy_return_type' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'stypy_return_type', result_gt_1547)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 400)
        stypy_return_type_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_1548


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 403, 4, False)
        # Assigning a type to the variable 'self' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Edge.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Edge.stypy__eq__')
        Edge.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Edge.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Edge.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Edge.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 404)
        self_1549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 16), 'self')
        # Obtaining the member 'inside' of a type (line 404)
        inside_1550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 16), self_1549, 'inside')
        # Getting the type of 'other' (line 404)
        other_1551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 31), 'other')
        # Obtaining the member 'inside' of a type (line 404)
        inside_1552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 31), other_1551, 'inside')
        # Applying the binary operator '==' (line 404)
        result_eq_1553 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 16), '==', inside_1550, inside_1552)
        
        
        # Getting the type of 'self' (line 405)
        self_1554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 20), 'self')
        # Obtaining the member 'prob' of a type (line 405)
        prob_1555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 20), self_1554, 'prob')
        # Getting the type of 'other' (line 405)
        other_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 33), 'other')
        # Obtaining the member 'prob' of a type (line 405)
        prob_1557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 33), other_1556, 'prob')
        # Applying the binary operator '==' (line 405)
        result_eq_1558 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 20), '==', prob_1555, prob_1557)
        
        # Applying the binary operator 'and' (line 404)
        result_and_keyword_1559 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 16), 'and', result_eq_1553, result_eq_1558)
        
        # Getting the type of 'self' (line 406)
        self_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'self')
        # Obtaining the member 'left' of a type (line 406)
        left_1561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), self_1560, 'left')
        # Getting the type of 'other' (line 406)
        other_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 33), 'other')
        # Obtaining the member 'right' of a type (line 406)
        right_1563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 33), other_1562, 'right')
        # Applying the binary operator '==' (line 406)
        result_eq_1564 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 20), '==', left_1561, right_1563)
        
        # Applying the binary operator 'and' (line 404)
        result_and_keyword_1565 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 16), 'and', result_and_keyword_1559, result_eq_1564)
        
        # Getting the type of 'self' (line 407)
        self_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 20), 'self')
        # Obtaining the member 'right' of a type (line 407)
        right_1567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 20), self_1566, 'right')
        # Getting the type of 'other' (line 407)
        other_1568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 34), 'other')
        # Obtaining the member 'right' of a type (line 407)
        right_1569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 34), other_1568, 'right')
        # Applying the binary operator '==' (line 407)
        result_eq_1570 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 20), '==', right_1567, right_1569)
        
        # Applying the binary operator 'and' (line 404)
        result_and_keyword_1571 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 16), 'and', result_and_keyword_1565, result_eq_1570)
        
        # Assigning a type to the variable 'stypy_return_type' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'stypy_return_type', result_and_keyword_1571)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 403)
        stypy_return_type_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1572)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_1572


# Assigning a type to the variable 'Edge' (line 385)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'Edge', Edge)

# Assigning a Tuple to a Name (line 386):

# Obtaining an instance of the builtin type 'tuple' (line 386)
tuple_1573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 386)
# Adding element type (line 386)
str_1574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 17), 'str', 'score')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), tuple_1573, str_1574)
# Adding element type (line 386)
str_1575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 26), 'str', 'inside')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), tuple_1573, str_1575)
# Adding element type (line 386)
str_1576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 36), 'str', 'prob')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), tuple_1573, str_1576)
# Adding element type (line 386)
str_1577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 44), 'str', 'left')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), tuple_1573, str_1577)
# Adding element type (line 386)
str_1578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 52), 'str', 'right')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), tuple_1573, str_1578)

# Getting the type of 'Edge'
Edge_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Edge')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Edge_1579, '__slots__', tuple_1573)
# Declaration of the 'Terminal' class

class Terminal:
    
    # Assigning a Tuple to a Name (line 411):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Terminal.__init__', ['lhs', 'rhs1', 'rhs2', 'word', 'prob'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['lhs', 'rhs1', 'rhs2', 'word', 'prob'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 414):
        
        # Assigning a Name to a Attribute (line 414):
        # Getting the type of 'lhs' (line 414)
        lhs_1580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 19), 'lhs')
        # Getting the type of 'self' (line 414)
        self_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'self')
        # Setting the type of the member 'lhs' of a type (line 414)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), self_1581, 'lhs', lhs_1580)
        
        # Assigning a Name to a Attribute (line 415):
        
        # Assigning a Name to a Attribute (line 415):
        # Getting the type of 'rhs1' (line 415)
        rhs1_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 20), 'rhs1')
        # Getting the type of 'self' (line 415)
        self_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'self')
        # Setting the type of the member 'rhs1' of a type (line 415)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), self_1583, 'rhs1', rhs1_1582)
        
        # Assigning a Name to a Attribute (line 416):
        
        # Assigning a Name to a Attribute (line 416):
        # Getting the type of 'rhs2' (line 416)
        rhs2_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'rhs2')
        # Getting the type of 'self' (line 416)
        self_1585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'self')
        # Setting the type of the member 'rhs2' of a type (line 416)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), self_1585, 'rhs2', rhs2_1584)
        
        # Assigning a Name to a Attribute (line 417):
        
        # Assigning a Name to a Attribute (line 417):
        # Getting the type of 'word' (line 417)
        word_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'word')
        # Getting the type of 'self' (line 417)
        self_1587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'self')
        # Setting the type of the member 'word' of a type (line 417)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), self_1587, 'word', word_1586)
        
        # Assigning a Name to a Attribute (line 418):
        
        # Assigning a Name to a Attribute (line 418):
        # Getting the type of 'prob' (line 418)
        prob_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 20), 'prob')
        # Getting the type of 'self' (line 418)
        self_1589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self')
        # Setting the type of the member 'prob' of a type (line 418)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_1589, 'prob', prob_1588)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Terminal' (line 410)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'Terminal', Terminal)

# Assigning a Tuple to a Name (line 411):

# Obtaining an instance of the builtin type 'tuple' (line 411)
tuple_1590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 411)
# Adding element type (line 411)
str_1591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 17), 'str', 'lhs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 17), tuple_1590, str_1591)
# Adding element type (line 411)
str_1592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 24), 'str', 'rhs1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 17), tuple_1590, str_1592)
# Adding element type (line 411)
str_1593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 32), 'str', 'rhs2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 17), tuple_1590, str_1593)
# Adding element type (line 411)
str_1594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 40), 'str', 'word')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 17), tuple_1590, str_1594)
# Adding element type (line 411)
str_1595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 48), 'str', 'prob')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 17), tuple_1590, str_1595)

# Getting the type of 'Terminal'
Terminal_1596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Terminal')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Terminal_1596, '__slots__', tuple_1590)
# Declaration of the 'Rule' class

class Rule:
    
    # Assigning a Tuple to a Name (line 422):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rule.__init__', ['lhs', 'rhs1', 'rhs2', 'args', 'lengths', 'prob'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['lhs', 'rhs1', 'rhs2', 'args', 'lengths', 'prob'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 426):
        
        # Assigning a Name to a Attribute (line 426):
        # Getting the type of 'lhs' (line 426)
        lhs_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 19), 'lhs')
        # Getting the type of 'self' (line 426)
        self_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'self')
        # Setting the type of the member 'lhs' of a type (line 426)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), self_1598, 'lhs', lhs_1597)
        
        # Assigning a Name to a Attribute (line 427):
        
        # Assigning a Name to a Attribute (line 427):
        # Getting the type of 'rhs1' (line 427)
        rhs1_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 20), 'rhs1')
        # Getting the type of 'self' (line 427)
        self_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self')
        # Setting the type of the member 'rhs1' of a type (line 427)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_1600, 'rhs1', rhs1_1599)
        
        # Assigning a Name to a Attribute (line 428):
        
        # Assigning a Name to a Attribute (line 428):
        # Getting the type of 'rhs2' (line 428)
        rhs2_1601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'rhs2')
        # Getting the type of 'self' (line 428)
        self_1602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'self')
        # Setting the type of the member 'rhs2' of a type (line 428)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), self_1602, 'rhs2', rhs2_1601)
        
        # Assigning a Name to a Attribute (line 429):
        
        # Assigning a Name to a Attribute (line 429):
        # Getting the type of 'args' (line 429)
        args_1603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'args')
        # Getting the type of 'self' (line 429)
        self_1604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self')
        # Setting the type of the member 'args' of a type (line 429)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_1604, 'args', args_1603)
        
        # Assigning a Name to a Attribute (line 430):
        
        # Assigning a Name to a Attribute (line 430):
        # Getting the type of 'lengths' (line 430)
        lengths_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 23), 'lengths')
        # Getting the type of 'self' (line 430)
        self_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self')
        # Setting the type of the member 'lengths' of a type (line 430)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_1606, 'lengths', lengths_1605)
        
        # Assigning a Name to a Attribute (line 431):
        
        # Assigning a Name to a Attribute (line 431):
        # Getting the type of 'prob' (line 431)
        prob_1607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 20), 'prob')
        # Getting the type of 'self' (line 431)
        self_1608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self')
        # Setting the type of the member 'prob' of a type (line 431)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_1608, 'prob', prob_1607)
        
        # Assigning a Attribute to a Attribute (line 432):
        
        # Assigning a Attribute to a Attribute (line 432):
        # Getting the type of 'self' (line 432)
        self_1609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'self')
        # Obtaining the member 'args' of a type (line 432)
        args_1610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 21), self_1609, 'args')
        # Getting the type of 'self' (line 432)
        self_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self')
        # Setting the type of the member '_args' of a type (line 432)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_1611, '_args', args_1610)
        
        # Assigning a Attribute to a Attribute (line 433):
        
        # Assigning a Attribute to a Attribute (line 433):
        # Getting the type of 'self' (line 433)
        self_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 24), 'self')
        # Obtaining the member 'lengths' of a type (line 433)
        lengths_1613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 24), self_1612, 'lengths')
        # Getting the type of 'self' (line 433)
        self_1614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self')
        # Setting the type of the member '_lengths' of a type (line 433)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_1614, '_lengths', lengths_1613)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Rule' (line 421)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 0), 'Rule', Rule)

# Assigning a Tuple to a Name (line 422):

# Obtaining an instance of the builtin type 'tuple' (line 422)
tuple_1615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 422)
# Adding element type (line 422)
str_1616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 17), 'str', 'lhs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1616)
# Adding element type (line 422)
str_1617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 24), 'str', 'rhs1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1617)
# Adding element type (line 422)
str_1618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 32), 'str', 'rhs2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1618)
# Adding element type (line 422)
str_1619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 40), 'str', 'prob')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1619)
# Adding element type (line 422)
str_1620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 17), 'str', 'args')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1620)
# Adding element type (line 422)
str_1621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 25), 'str', 'lengths')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1621)
# Adding element type (line 422)
str_1622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 36), 'str', '_args')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1622)
# Adding element type (line 422)
str_1623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 45), 'str', 'lengths')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 17), tuple_1615, str_1623)

# Getting the type of 'Rule'
Rule_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Rule')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Rule_1624, '__slots__', tuple_1615)
# Declaration of the 'Entry' class

class Entry(object, ):
    
    # Assigning a Tuple to a Name (line 438):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 440, 4, False)
        # Assigning a type to the variable 'self' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Entry.__init__', ['key', 'value', 'count'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['key', 'value', 'count'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 441):
        
        # Assigning a Name to a Attribute (line 441):
        # Getting the type of 'key' (line 441)
        key_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 19), 'key')
        # Getting the type of 'self' (line 441)
        self_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self')
        # Setting the type of the member 'key' of a type (line 441)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_1626, 'key', key_1625)
        
        # Assigning a Name to a Attribute (line 442):
        
        # Assigning a Name to a Attribute (line 442):
        # Getting the type of 'value' (line 442)
        value_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 21), 'value')
        # Getting the type of 'self' (line 442)
        self_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self')
        # Setting the type of the member 'value' of a type (line 442)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_1628, 'value', value_1627)
        
        # Assigning a Name to a Attribute (line 443):
        
        # Assigning a Name to a Attribute (line 443):
        # Getting the type of 'count' (line 443)
        count_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 21), 'count')
        # Getting the type of 'self' (line 443)
        self_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self')
        # Setting the type of the member 'count' of a type (line 443)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_1630, 'count', count_1629)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 445, 4, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Entry.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Entry.stypy__eq__')
        Entry.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Entry.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Entry.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Entry.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Getting the type of 'self' (line 446)
        self_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 15), 'self')
        # Obtaining the member 'count' of a type (line 446)
        count_1632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 15), self_1631, 'count')
        # Getting the type of 'other' (line 446)
        other_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 29), 'other')
        # Obtaining the member 'count' of a type (line 446)
        count_1634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 29), other_1633, 'count')
        # Applying the binary operator '==' (line 446)
        result_eq_1635 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 15), '==', count_1632, count_1634)
        
        # Assigning a type to the variable 'stypy_return_type' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'stypy_return_type', result_eq_1635)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 445)
        stypy_return_type_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1636)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_1636


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 448, 4, False)
        # Assigning a type to the variable 'self' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Entry.__lt__.__dict__.__setitem__('stypy_localization', localization)
        Entry.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Entry.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Entry.__lt__.__dict__.__setitem__('stypy_function_name', 'Entry.__lt__')
        Entry.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Entry.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Entry.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Entry.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Entry.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Entry.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Entry.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Entry.__lt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lt__(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 449)
        self_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'self')
        # Obtaining the member 'value' of a type (line 449)
        value_1638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 15), self_1637, 'value')
        # Getting the type of 'other' (line 449)
        other_1639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 28), 'other')
        # Obtaining the member 'value' of a type (line 449)
        value_1640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 28), other_1639, 'value')
        # Applying the binary operator '<' (line 449)
        result_lt_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 15), '<', value_1638, value_1640)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 449)
        self_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 44), 'self')
        # Obtaining the member 'value' of a type (line 449)
        value_1643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 44), self_1642, 'value')
        # Getting the type of 'other' (line 449)
        other_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 58), 'other')
        # Obtaining the member 'value' of a type (line 449)
        value_1645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 58), other_1644, 'value')
        # Applying the binary operator '==' (line 449)
        result_eq_1646 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 44), '==', value_1643, value_1645)
        
        
        # Getting the type of 'self' (line 450)
        self_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 48), 'self')
        # Obtaining the member 'count' of a type (line 450)
        count_1648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 48), self_1647, 'count')
        # Getting the type of 'other' (line 450)
        other_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 61), 'other')
        # Obtaining the member 'count' of a type (line 450)
        count_1650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 61), other_1649, 'count')
        # Applying the binary operator '<' (line 450)
        result_lt_1651 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 48), '<', count_1648, count_1650)
        
        # Applying the binary operator 'and' (line 449)
        result_and_keyword_1652 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 44), 'and', result_eq_1646, result_lt_1651)
        
        # Applying the binary operator 'or' (line 449)
        result_or_keyword_1653 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 15), 'or', result_lt_1641, result_and_keyword_1652)
        
        # Assigning a type to the variable 'stypy_return_type' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'stypy_return_type', result_or_keyword_1653)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 448)
        stypy_return_type_1654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1654)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_1654


# Assigning a type to the variable 'Entry' (line 437)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 0), 'Entry', Entry)

# Assigning a Tuple to a Name (line 438):

# Obtaining an instance of the builtin type 'tuple' (line 438)
tuple_1655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 438)
# Adding element type (line 438)
str_1656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 17), 'str', 'key')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 17), tuple_1655, str_1656)
# Adding element type (line 438)
str_1657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 24), 'str', 'value')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 17), tuple_1655, str_1657)
# Adding element type (line 438)
str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 33), 'str', 'count')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 17), tuple_1655, str_1658)

# Getting the type of 'Entry'
Entry_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Entry')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Entry_1659, '__slots__', tuple_1655)

# Assigning a Num to a Name (line 453):

# Assigning a Num to a Name (line 453):
int_1660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 10), 'int')
# Assigning a type to the variable 'INVALID' (line 453)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), 'INVALID', int_1660)
# Declaration of the 'agenda' class

class agenda(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 457, 4, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'agenda.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 458):
        
        # Assigning a List to a Attribute (line 458):
        
        # Obtaining an instance of the builtin type 'list' (line 458)
        list_1661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 458)
        
        # Getting the type of 'self' (line 458)
        self_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'self')
        # Setting the type of the member 'heap' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), self_1662, 'heap', list_1661)
        
        # Assigning a Dict to a Attribute (line 459):
        
        # Assigning a Dict to a Attribute (line 459):
        
        # Obtaining an instance of the builtin type 'dict' (line 459)
        dict_1663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 459)
        
        # Getting the type of 'self' (line 459)
        self_1664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'self')
        # Setting the type of the member 'mapping' of a type (line 459)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), self_1664, 'mapping', dict_1663)
        
        # Assigning a Num to a Attribute (line 460):
        
        # Assigning a Num to a Attribute (line 460):
        int_1665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'int')
        # Getting the type of 'self' (line 460)
        self_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'self')
        # Setting the type of the member 'counter' of a type (line 460)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), self_1666, 'counter', int_1665)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 462, 4, False)
        # Assigning a type to the variable 'self' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        agenda.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        agenda.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        agenda.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        agenda.__setitem__.__dict__.__setitem__('stypy_function_name', 'agenda.__setitem__')
        agenda.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['key', 'value'])
        agenda.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        agenda.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        agenda.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        agenda.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        agenda.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        agenda.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'agenda.__setitem__', ['key', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['key', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        
        # Getting the type of 'key' (line 463)
        key_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'key')
        # Getting the type of 'self' (line 463)
        self_1668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 18), 'self')
        # Obtaining the member 'mapping' of a type (line 463)
        mapping_1669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 18), self_1668, 'mapping')
        # Applying the binary operator 'in' (line 463)
        result_contains_1670 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 11), 'in', key_1667, mapping_1669)
        
        # Testing if the type of an if condition is none (line 463)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 463, 8), result_contains_1670):
            
            # Assigning a Call to a Name (line 470):
            
            # Assigning a Call to a Name (line 470):
            
            # Call to Entry(...): (line 470)
            # Processing the call arguments (line 470)
            # Getting the type of 'key' (line 470)
            key_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'key', False)
            # Getting the type of 'value' (line 470)
            value_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 31), 'value', False)
            # Getting the type of 'self' (line 470)
            self_1699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 38), 'self', False)
            # Obtaining the member 'counter' of a type (line 470)
            counter_1700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 38), self_1699, 'counter')
            # Processing the call keyword arguments (line 470)
            kwargs_1701 = {}
            # Getting the type of 'Entry' (line 470)
            Entry_1696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'Entry', False)
            # Calling Entry(args, kwargs) (line 470)
            Entry_call_result_1702 = invoke(stypy.reporting.localization.Localization(__file__, 470, 20), Entry_1696, *[key_1697, value_1698, counter_1700], **kwargs_1701)
            
            # Assigning a type to the variable 'entry' (line 470)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'entry', Entry_call_result_1702)
            
            # Getting the type of 'self' (line 471)
            self_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'self')
            # Obtaining the member 'counter' of a type (line 471)
            counter_1704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 12), self_1703, 'counter')
            int_1705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 28), 'int')
            # Applying the binary operator '+=' (line 471)
            result_iadd_1706 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 12), '+=', counter_1704, int_1705)
            # Getting the type of 'self' (line 471)
            self_1707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'self')
            # Setting the type of the member 'counter' of a type (line 471)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 12), self_1707, 'counter', result_iadd_1706)
            
            
            # Assigning a Name to a Subscript (line 472):
            
            # Assigning a Name to a Subscript (line 472):
            # Getting the type of 'entry' (line 472)
            entry_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 32), 'entry')
            # Getting the type of 'self' (line 472)
            self_1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'self')
            # Obtaining the member 'mapping' of a type (line 472)
            mapping_1710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), self_1709, 'mapping')
            # Getting the type of 'key' (line 472)
            key_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 25), 'key')
            # Storing an element on a container (line 472)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 12), mapping_1710, (key_1711, entry_1708))
            
            # Call to heappush(...): (line 473)
            # Processing the call arguments (line 473)
            # Getting the type of 'self' (line 473)
            self_1713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 21), 'self', False)
            # Obtaining the member 'heap' of a type (line 473)
            heap_1714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 21), self_1713, 'heap')
            # Getting the type of 'entry' (line 473)
            entry_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 32), 'entry', False)
            # Processing the call keyword arguments (line 473)
            kwargs_1716 = {}
            # Getting the type of 'heappush' (line 473)
            heappush_1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'heappush', False)
            # Calling heappush(args, kwargs) (line 473)
            heappush_call_result_1717 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), heappush_1712, *[heap_1714, entry_1715], **kwargs_1716)
            
        else:
            
            # Testing the type of an if condition (line 463)
            if_condition_1671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 8), result_contains_1670)
            # Assigning a type to the variable 'if_condition_1671' (line 463)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'if_condition_1671', if_condition_1671)
            # SSA begins for if statement (line 463)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 464):
            
            # Assigning a Subscript to a Name (line 464):
            
            # Obtaining the type of the subscript
            # Getting the type of 'key' (line 464)
            key_1672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 36), 'key')
            # Getting the type of 'self' (line 464)
            self_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 23), 'self')
            # Obtaining the member 'mapping' of a type (line 464)
            mapping_1674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 23), self_1673, 'mapping')
            # Obtaining the member '__getitem__' of a type (line 464)
            getitem___1675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 23), mapping_1674, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 464)
            subscript_call_result_1676 = invoke(stypy.reporting.localization.Localization(__file__, 464, 23), getitem___1675, key_1672)
            
            # Assigning a type to the variable 'oldentry' (line 464)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'oldentry', subscript_call_result_1676)
            
            # Assigning a Call to a Name (line 465):
            
            # Assigning a Call to a Name (line 465):
            
            # Call to Entry(...): (line 465)
            # Processing the call arguments (line 465)
            # Getting the type of 'key' (line 465)
            key_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 26), 'key', False)
            # Getting the type of 'value' (line 465)
            value_1679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 31), 'value', False)
            # Getting the type of 'oldentry' (line 465)
            oldentry_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 38), 'oldentry', False)
            # Obtaining the member 'count' of a type (line 465)
            count_1681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 38), oldentry_1680, 'count')
            # Processing the call keyword arguments (line 465)
            kwargs_1682 = {}
            # Getting the type of 'Entry' (line 465)
            Entry_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 20), 'Entry', False)
            # Calling Entry(args, kwargs) (line 465)
            Entry_call_result_1683 = invoke(stypy.reporting.localization.Localization(__file__, 465, 20), Entry_1677, *[key_1678, value_1679, count_1681], **kwargs_1682)
            
            # Assigning a type to the variable 'entry' (line 465)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'entry', Entry_call_result_1683)
            
            # Assigning a Name to a Subscript (line 466):
            
            # Assigning a Name to a Subscript (line 466):
            # Getting the type of 'entry' (line 466)
            entry_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 32), 'entry')
            # Getting the type of 'self' (line 466)
            self_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'self')
            # Obtaining the member 'mapping' of a type (line 466)
            mapping_1686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), self_1685, 'mapping')
            # Getting the type of 'key' (line 466)
            key_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 25), 'key')
            # Storing an element on a container (line 466)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 12), mapping_1686, (key_1687, entry_1684))
            
            # Call to heappush(...): (line 467)
            # Processing the call arguments (line 467)
            # Getting the type of 'self' (line 467)
            self_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'self', False)
            # Obtaining the member 'heap' of a type (line 467)
            heap_1690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 21), self_1689, 'heap')
            # Getting the type of 'entry' (line 467)
            entry_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 32), 'entry', False)
            # Processing the call keyword arguments (line 467)
            kwargs_1692 = {}
            # Getting the type of 'heappush' (line 467)
            heappush_1688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'heappush', False)
            # Calling heappush(args, kwargs) (line 467)
            heappush_call_result_1693 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), heappush_1688, *[heap_1690, entry_1691], **kwargs_1692)
            
            
            # Assigning a Name to a Attribute (line 468):
            
            # Assigning a Name to a Attribute (line 468):
            # Getting the type of 'INVALID' (line 468)
            INVALID_1694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 29), 'INVALID')
            # Getting the type of 'oldentry' (line 468)
            oldentry_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'oldentry')
            # Setting the type of the member 'count' of a type (line 468)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 12), oldentry_1695, 'count', INVALID_1694)
            # SSA branch for the else part of an if statement (line 463)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 470):
            
            # Assigning a Call to a Name (line 470):
            
            # Call to Entry(...): (line 470)
            # Processing the call arguments (line 470)
            # Getting the type of 'key' (line 470)
            key_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'key', False)
            # Getting the type of 'value' (line 470)
            value_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 31), 'value', False)
            # Getting the type of 'self' (line 470)
            self_1699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 38), 'self', False)
            # Obtaining the member 'counter' of a type (line 470)
            counter_1700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 38), self_1699, 'counter')
            # Processing the call keyword arguments (line 470)
            kwargs_1701 = {}
            # Getting the type of 'Entry' (line 470)
            Entry_1696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'Entry', False)
            # Calling Entry(args, kwargs) (line 470)
            Entry_call_result_1702 = invoke(stypy.reporting.localization.Localization(__file__, 470, 20), Entry_1696, *[key_1697, value_1698, counter_1700], **kwargs_1701)
            
            # Assigning a type to the variable 'entry' (line 470)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'entry', Entry_call_result_1702)
            
            # Getting the type of 'self' (line 471)
            self_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'self')
            # Obtaining the member 'counter' of a type (line 471)
            counter_1704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 12), self_1703, 'counter')
            int_1705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 28), 'int')
            # Applying the binary operator '+=' (line 471)
            result_iadd_1706 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 12), '+=', counter_1704, int_1705)
            # Getting the type of 'self' (line 471)
            self_1707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'self')
            # Setting the type of the member 'counter' of a type (line 471)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 12), self_1707, 'counter', result_iadd_1706)
            
            
            # Assigning a Name to a Subscript (line 472):
            
            # Assigning a Name to a Subscript (line 472):
            # Getting the type of 'entry' (line 472)
            entry_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 32), 'entry')
            # Getting the type of 'self' (line 472)
            self_1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'self')
            # Obtaining the member 'mapping' of a type (line 472)
            mapping_1710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), self_1709, 'mapping')
            # Getting the type of 'key' (line 472)
            key_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 25), 'key')
            # Storing an element on a container (line 472)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 12), mapping_1710, (key_1711, entry_1708))
            
            # Call to heappush(...): (line 473)
            # Processing the call arguments (line 473)
            # Getting the type of 'self' (line 473)
            self_1713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 21), 'self', False)
            # Obtaining the member 'heap' of a type (line 473)
            heap_1714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 21), self_1713, 'heap')
            # Getting the type of 'entry' (line 473)
            entry_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 32), 'entry', False)
            # Processing the call keyword arguments (line 473)
            kwargs_1716 = {}
            # Getting the type of 'heappush' (line 473)
            heappush_1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'heappush', False)
            # Calling heappush(args, kwargs) (line 473)
            heappush_call_result_1717 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), heappush_1712, *[heap_1714, entry_1715], **kwargs_1716)
            
            # SSA join for if statement (line 463)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 462)
        stypy_return_type_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1718)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_1718


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        agenda.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        agenda.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        agenda.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        agenda.__getitem__.__dict__.__setitem__('stypy_function_name', 'agenda.__getitem__')
        agenda.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        agenda.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        agenda.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        agenda.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        agenda.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        agenda.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        agenda.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'agenda.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 476)
        key_1719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 28), 'key')
        # Getting the type of 'self' (line 476)
        self_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'self')
        # Obtaining the member 'mapping' of a type (line 476)
        mapping_1721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), self_1720, 'mapping')
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___1722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), mapping_1721, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_1723 = invoke(stypy.reporting.localization.Localization(__file__, 476, 15), getitem___1722, key_1719)
        
        # Obtaining the member 'value' of a type (line 476)
        value_1724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), subscript_call_result_1723, 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type', value_1724)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_1725


    @norecursion
    def __contains__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__contains__'
        module_type_store = module_type_store.open_function_context('__contains__', 478, 4, False)
        # Assigning a type to the variable 'self' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        agenda.__contains__.__dict__.__setitem__('stypy_localization', localization)
        agenda.__contains__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        agenda.__contains__.__dict__.__setitem__('stypy_type_store', module_type_store)
        agenda.__contains__.__dict__.__setitem__('stypy_function_name', 'agenda.__contains__')
        agenda.__contains__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        agenda.__contains__.__dict__.__setitem__('stypy_varargs_param_name', None)
        agenda.__contains__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        agenda.__contains__.__dict__.__setitem__('stypy_call_defaults', defaults)
        agenda.__contains__.__dict__.__setitem__('stypy_call_varargs', varargs)
        agenda.__contains__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        agenda.__contains__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'agenda.__contains__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__contains__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__contains__(...)' code ##################

        
        # Getting the type of 'key' (line 479)
        key_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'key')
        # Getting the type of 'self' (line 479)
        self_1727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 22), 'self')
        # Obtaining the member 'mapping' of a type (line 479)
        mapping_1728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 22), self_1727, 'mapping')
        # Applying the binary operator 'in' (line 479)
        result_contains_1729 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 15), 'in', key_1726, mapping_1728)
        
        # Assigning a type to the variable 'stypy_return_type' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'stypy_return_type', result_contains_1729)
        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 478)
        stypy_return_type_1730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_1730


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 481, 4, False)
        # Assigning a type to the variable 'self' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        agenda.__len__.__dict__.__setitem__('stypy_localization', localization)
        agenda.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        agenda.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        agenda.__len__.__dict__.__setitem__('stypy_function_name', 'agenda.__len__')
        agenda.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        agenda.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        agenda.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        agenda.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        agenda.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        agenda.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        agenda.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'agenda.__len__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to len(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'self' (line 482)
        self_1732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'self', False)
        # Obtaining the member 'mapping' of a type (line 482)
        mapping_1733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 19), self_1732, 'mapping')
        # Processing the call keyword arguments (line 482)
        kwargs_1734 = {}
        # Getting the type of 'len' (line 482)
        len_1731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), 'len', False)
        # Calling len(args, kwargs) (line 482)
        len_call_result_1735 = invoke(stypy.reporting.localization.Localization(__file__, 482, 15), len_1731, *[mapping_1733], **kwargs_1734)
        
        # Assigning a type to the variable 'stypy_return_type' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'stypy_return_type', len_call_result_1735)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 481)
        stypy_return_type_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1736)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_1736


    @norecursion
    def popitem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'popitem'
        module_type_store = module_type_store.open_function_context('popitem', 484, 4, False)
        # Assigning a type to the variable 'self' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        agenda.popitem.__dict__.__setitem__('stypy_localization', localization)
        agenda.popitem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        agenda.popitem.__dict__.__setitem__('stypy_type_store', module_type_store)
        agenda.popitem.__dict__.__setitem__('stypy_function_name', 'agenda.popitem')
        agenda.popitem.__dict__.__setitem__('stypy_param_names_list', [])
        agenda.popitem.__dict__.__setitem__('stypy_varargs_param_name', None)
        agenda.popitem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        agenda.popitem.__dict__.__setitem__('stypy_call_defaults', defaults)
        agenda.popitem.__dict__.__setitem__('stypy_call_varargs', varargs)
        agenda.popitem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        agenda.popitem.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'agenda.popitem', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'popitem', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'popitem(...)' code ##################

        
        # Assigning a Call to a Name (line 485):
        
        # Assigning a Call to a Name (line 485):
        
        # Call to heappop(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'self' (line 485)
        self_1738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 24), 'self', False)
        # Obtaining the member 'heap' of a type (line 485)
        heap_1739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 24), self_1738, 'heap')
        # Processing the call keyword arguments (line 485)
        kwargs_1740 = {}
        # Getting the type of 'heappop' (line 485)
        heappop_1737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'heappop', False)
        # Calling heappop(args, kwargs) (line 485)
        heappop_call_result_1741 = invoke(stypy.reporting.localization.Localization(__file__, 485, 16), heappop_1737, *[heap_1739], **kwargs_1740)
        
        # Assigning a type to the variable 'entry' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'entry', heappop_call_result_1741)
        
        
        # Getting the type of 'entry' (line 486)
        entry_1742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 14), 'entry')
        # Obtaining the member 'count' of a type (line 486)
        count_1743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 14), entry_1742, 'count')
        # Getting the type of 'INVALID' (line 486)
        INVALID_1744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 29), 'INVALID')
        # Applying the binary operator 'is' (line 486)
        result_is__1745 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 14), 'is', count_1743, INVALID_1744)
        
        # Assigning a type to the variable 'result_is__1745' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'result_is__1745', result_is__1745)
        # Testing if the while is going to be iterated (line 486)
        # Testing the type of an if condition (line 486)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 8), result_is__1745)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 486, 8), result_is__1745):
            # SSA begins for while statement (line 486)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 487):
            
            # Assigning a Call to a Name (line 487):
            
            # Call to heappop(...): (line 487)
            # Processing the call arguments (line 487)
            # Getting the type of 'self' (line 487)
            self_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 28), 'self', False)
            # Obtaining the member 'heap' of a type (line 487)
            heap_1748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 28), self_1747, 'heap')
            # Processing the call keyword arguments (line 487)
            kwargs_1749 = {}
            # Getting the type of 'heappop' (line 487)
            heappop_1746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 20), 'heappop', False)
            # Calling heappop(args, kwargs) (line 487)
            heappop_call_result_1750 = invoke(stypy.reporting.localization.Localization(__file__, 487, 20), heappop_1746, *[heap_1748], **kwargs_1749)
            
            # Assigning a type to the variable 'entry' (line 487)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'entry', heappop_call_result_1750)
            # SSA join for while statement (line 486)
            module_type_store = module_type_store.join_ssa_context()

        
        # Deleting a member
        # Getting the type of 'self' (line 488)
        self_1751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'self')
        # Obtaining the member 'mapping' of a type (line 488)
        mapping_1752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), self_1751, 'mapping')
        
        # Obtaining the type of the subscript
        # Getting the type of 'entry' (line 488)
        entry_1753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 25), 'entry')
        # Obtaining the member 'key' of a type (line 488)
        key_1754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 25), entry_1753, 'key')
        # Getting the type of 'self' (line 488)
        self_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'self')
        # Obtaining the member 'mapping' of a type (line 488)
        mapping_1756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), self_1755, 'mapping')
        # Obtaining the member '__getitem__' of a type (line 488)
        getitem___1757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), mapping_1756, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 488)
        subscript_call_result_1758 = invoke(stypy.reporting.localization.Localization(__file__, 488, 12), getitem___1757, key_1754)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 8), mapping_1752, subscript_call_result_1758)
        
        # Obtaining an instance of the builtin type 'tuple' (line 489)
        tuple_1759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 489)
        # Adding element type (line 489)
        # Getting the type of 'entry' (line 489)
        entry_1760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'entry')
        # Obtaining the member 'key' of a type (line 489)
        key_1761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), entry_1760, 'key')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 15), tuple_1759, key_1761)
        # Adding element type (line 489)
        # Getting the type of 'entry' (line 489)
        entry_1762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 26), 'entry')
        # Obtaining the member 'value' of a type (line 489)
        value_1763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 26), entry_1762, 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 15), tuple_1759, value_1763)
        
        # Assigning a type to the variable 'stypy_return_type' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type', tuple_1759)
        
        # ################# End of 'popitem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'popitem' in the type store
        # Getting the type of 'stypy_return_type' (line 484)
        stypy_return_type_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'popitem'
        return stypy_return_type_1764


# Assigning a type to the variable 'agenda' (line 456)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'agenda', agenda)

@norecursion
def batch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'batch'
    module_type_store = module_type_store.open_function_context('batch', 492, 0, False)
    
    # Passed parameters checking function
    batch.stypy_localization = localization
    batch.stypy_type_of_self = None
    batch.stypy_type_store = module_type_store
    batch.stypy_function_name = 'batch'
    batch.stypy_param_names_list = ['rulefile', 'lexiconfile', 'sentfile']
    batch.stypy_varargs_param_name = None
    batch.stypy_kwargs_param_name = None
    batch.stypy_call_defaults = defaults
    batch.stypy_call_varargs = varargs
    batch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'batch', ['rulefile', 'lexiconfile', 'sentfile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'batch', localization, ['rulefile', 'lexiconfile', 'sentfile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'batch(...)' code ##################

    
    # Assigning a Call to a Tuple (line 493):
    
    # Assigning a Call to a Name:
    
    # Call to read_srcg_grammar(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'rulefile' (line 493)
    rulefile_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 39), 'rulefile', False)
    # Getting the type of 'lexiconfile' (line 493)
    lexiconfile_1767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 49), 'lexiconfile', False)
    # Processing the call keyword arguments (line 493)
    kwargs_1768 = {}
    # Getting the type of 'read_srcg_grammar' (line 493)
    read_srcg_grammar_1765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 21), 'read_srcg_grammar', False)
    # Calling read_srcg_grammar(args, kwargs) (line 493)
    read_srcg_grammar_call_result_1769 = invoke(stypy.reporting.localization.Localization(__file__, 493, 21), read_srcg_grammar_1765, *[rulefile_1766, lexiconfile_1767], **kwargs_1768)
    
    # Assigning a type to the variable 'call_assignment_13' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'call_assignment_13', read_srcg_grammar_call_result_1769)
    
    # Assigning a Call to a Name (line 493):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_1772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 4), 'int')
    # Processing the call keyword arguments
    kwargs_1773 = {}
    # Getting the type of 'call_assignment_13' (line 493)
    call_assignment_13_1770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'call_assignment_13', False)
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___1771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 4), call_assignment_13_1770, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_1774 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1771, *[int_1772], **kwargs_1773)
    
    # Assigning a type to the variable 'call_assignment_14' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'call_assignment_14', getitem___call_result_1774)
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'call_assignment_14' (line 493)
    call_assignment_14_1775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'call_assignment_14')
    # Assigning a type to the variable 'rules' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'rules', call_assignment_14_1775)
    
    # Assigning a Call to a Name (line 493):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_1778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 4), 'int')
    # Processing the call keyword arguments
    kwargs_1779 = {}
    # Getting the type of 'call_assignment_13' (line 493)
    call_assignment_13_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'call_assignment_13', False)
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___1777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 4), call_assignment_13_1776, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_1780 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1777, *[int_1778], **kwargs_1779)
    
    # Assigning a type to the variable 'call_assignment_15' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'call_assignment_15', getitem___call_result_1780)
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'call_assignment_15' (line 493)
    call_assignment_15_1781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'call_assignment_15')
    # Assigning a type to the variable 'lexicon' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'lexicon', call_assignment_15_1781)
    
    # Assigning a Subscript to a Name (line 494):
    
    # Assigning a Subscript to a Name (line 494):
    
    # Obtaining the type of the subscript
    int_1782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 26), 'int')
    
    # Obtaining the type of the subscript
    int_1783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 23), 'int')
    
    # Obtaining the type of the subscript
    int_1784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 20), 'int')
    
    # Obtaining the type of the subscript
    int_1785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 17), 'int')
    # Getting the type of 'rules' (line 494)
    rules_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'rules')
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___1787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 11), rules_1786, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 494)
    subscript_call_result_1788 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), getitem___1787, int_1785)
    
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___1789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 11), subscript_call_result_1788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 494)
    subscript_call_result_1790 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), getitem___1789, int_1784)
    
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___1791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 11), subscript_call_result_1790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 494)
    subscript_call_result_1792 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), getitem___1791, int_1783)
    
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___1793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 11), subscript_call_result_1792, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 494)
    subscript_call_result_1794 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), getitem___1793, int_1782)
    
    # Assigning a type to the variable 'root' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'root', subscript_call_result_1794)
    
    # Assigning a Call to a Name (line 495):
    
    # Assigning a Call to a Name (line 495):
    
    # Call to splitgrammar(...): (line 495)
    # Processing the call arguments (line 495)
    # Getting the type of 'rules' (line 495)
    rules_1796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 27), 'rules', False)
    # Getting the type of 'lexicon' (line 495)
    lexicon_1797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 34), 'lexicon', False)
    # Processing the call keyword arguments (line 495)
    kwargs_1798 = {}
    # Getting the type of 'splitgrammar' (line 495)
    splitgrammar_1795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 14), 'splitgrammar', False)
    # Calling splitgrammar(args, kwargs) (line 495)
    splitgrammar_call_result_1799 = invoke(stypy.reporting.localization.Localization(__file__, 495, 14), splitgrammar_1795, *[rules_1796, lexicon_1797], **kwargs_1798)
    
    # Assigning a type to the variable 'grammar' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'grammar', splitgrammar_call_result_1799)
    
    # Assigning a Call to a Name (line 496):
    
    # Assigning a Call to a Name (line 496):
    
    # Call to splitlines(...): (line 496)
    # Processing the call keyword arguments (line 496)
    kwargs_1808 = {}
    
    # Call to read(...): (line 496)
    # Processing the call keyword arguments (line 496)
    kwargs_1805 = {}
    
    # Call to open(...): (line 496)
    # Processing the call arguments (line 496)
    # Getting the type of 'sentfile' (line 496)
    sentfile_1801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 17), 'sentfile', False)
    # Processing the call keyword arguments (line 496)
    kwargs_1802 = {}
    # Getting the type of 'open' (line 496)
    open_1800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'open', False)
    # Calling open(args, kwargs) (line 496)
    open_call_result_1803 = invoke(stypy.reporting.localization.Localization(__file__, 496, 12), open_1800, *[sentfile_1801], **kwargs_1802)
    
    # Obtaining the member 'read' of a type (line 496)
    read_1804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 12), open_call_result_1803, 'read')
    # Calling read(args, kwargs) (line 496)
    read_call_result_1806 = invoke(stypy.reporting.localization.Localization(__file__, 496, 12), read_1804, *[], **kwargs_1805)
    
    # Obtaining the member 'splitlines' of a type (line 496)
    splitlines_1807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 12), read_call_result_1806, 'splitlines')
    # Calling splitlines(args, kwargs) (line 496)
    splitlines_call_result_1809 = invoke(stypy.reporting.localization.Localization(__file__, 496, 12), splitlines_1807, *[], **kwargs_1808)
    
    # Assigning a type to the variable 'lines' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'lines', splitlines_call_result_1809)
    
    # Assigning a ListComp to a Name (line 497):
    
    # Assigning a ListComp to a Name (line 497):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'lines' (line 497)
    lines_1821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 62), 'lines')
    comprehension_1822 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), lines_1821)
    # Assigning a type to the variable 'sent' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'sent', comprehension_1822)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 497)
    # Processing the call keyword arguments (line 497)
    kwargs_1817 = {}
    # Getting the type of 'sent' (line 497)
    sent_1815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 36), 'sent', False)
    # Obtaining the member 'split' of a type (line 497)
    split_1816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 36), sent_1815, 'split')
    # Calling split(args, kwargs) (line 497)
    split_call_result_1818 = invoke(stypy.reporting.localization.Localization(__file__, 497, 36), split_1816, *[], **kwargs_1817)
    
    comprehension_1819 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 14), split_call_result_1818)
    # Assigning a type to the variable 'a' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 14), 'a', comprehension_1819)
    
    # Call to split(...): (line 497)
    # Processing the call arguments (line 497)
    str_1812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 22), 'str', '/')
    # Processing the call keyword arguments (line 497)
    kwargs_1813 = {}
    # Getting the type of 'a' (line 497)
    a_1810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 14), 'a', False)
    # Obtaining the member 'split' of a type (line 497)
    split_1811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 14), a_1810, 'split')
    # Calling split(args, kwargs) (line 497)
    split_call_result_1814 = invoke(stypy.reporting.localization.Localization(__file__, 497, 14), split_1811, *[str_1812], **kwargs_1813)
    
    list_1820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 14), list_1820, split_call_result_1814)
    list_1823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), list_1823, list_1820)
    # Assigning a type to the variable 'sents' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'sents', list_1823)
    
    # Getting the type of 'sents' (line 498)
    sents_1824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 21), 'sents')
    # Assigning a type to the variable 'sents_1824' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'sents_1824', sents_1824)
    # Testing if the for loop is going to be iterated (line 498)
    # Testing the type of a for loop iterable (line 498)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 498, 4), sents_1824)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 498, 4), sents_1824):
        # Getting the type of the for loop variable (line 498)
        for_loop_var_1825 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 498, 4), sents_1824)
        # Assigning a type to the variable 'wordstags' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'wordstags', for_loop_var_1825)
        # SSA begins for a for statement (line 498)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 499):
        
        # Assigning a ListComp to a Name (line 499):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'wordstags' (line 499)
        wordstags_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 'wordstags')
        comprehension_1831 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 16), wordstags_1830)
        # Assigning a type to the variable 'a' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'a', comprehension_1831)
        
        # Obtaining the type of the subscript
        int_1826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 18), 'int')
        # Getting the type of 'a' (line 499)
        a_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'a')
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___1828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 16), a_1827, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_1829 = invoke(stypy.reporting.localization.Localization(__file__, 499, 16), getitem___1828, int_1826)
        
        list_1832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 16), list_1832, subscript_call_result_1829)
        # Assigning a type to the variable 'sent' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'sent', list_1832)
        
        # Assigning a ListComp to a Name (line 500):
        
        # Assigning a ListComp to a Name (line 500):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'wordstags' (line 500)
        wordstags_1837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 30), 'wordstags')
        comprehension_1838 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 16), wordstags_1837)
        # Assigning a type to the variable 'a' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), 'a', comprehension_1838)
        
        # Obtaining the type of the subscript
        int_1833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 18), 'int')
        # Getting the type of 'a' (line 500)
        a_1834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), 'a')
        # Obtaining the member '__getitem__' of a type (line 500)
        getitem___1835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 16), a_1834, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 500)
        subscript_call_result_1836 = invoke(stypy.reporting.localization.Localization(__file__, 500, 16), getitem___1835, int_1833)
        
        list_1839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 16), list_1839, subscript_call_result_1836)
        # Assigning a type to the variable 'tags' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tags', list_1839)
        
        # Call to write(...): (line 501)
        # Processing the call arguments (line 501)
        str_1842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 21), 'str', 'parsing: %s\n')
        
        # Call to join(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'sent' (line 501)
        sent_1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'sent', False)
        # Processing the call keyword arguments (line 501)
        kwargs_1846 = {}
        str_1843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 39), 'str', ' ')
        # Obtaining the member 'join' of a type (line 501)
        join_1844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 39), str_1843, 'join')
        # Calling join(args, kwargs) (line 501)
        join_call_result_1847 = invoke(stypy.reporting.localization.Localization(__file__, 501, 39), join_1844, *[sent_1845], **kwargs_1846)
        
        # Applying the binary operator '%' (line 501)
        result_mod_1848 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 21), '%', str_1842, join_call_result_1847)
        
        # Processing the call keyword arguments (line 501)
        kwargs_1849 = {}
        # Getting the type of 'stderr' (line 501)
        stderr_1840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'stderr', False)
        # Obtaining the member 'write' of a type (line 501)
        write_1841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), stderr_1840, 'write')
        # Calling write(args, kwargs) (line 501)
        write_call_result_1850 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), write_1841, *[result_mod_1848], **kwargs_1849)
        
        
        # Assigning a Call to a Tuple (line 502):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'sent' (line 502)
        sent_1852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 29), 'sent', False)
        # Getting the type of 'grammar' (line 502)
        grammar_1853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 35), 'grammar', False)
        # Getting the type of 'tags' (line 502)
        tags_1854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 44), 'tags', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'root' (line 502)
        root_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 63), 'root', False)
        # Getting the type of 'grammar' (line 502)
        grammar_1856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 50), 'grammar', False)
        # Obtaining the member 'toid' of a type (line 502)
        toid_1857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 50), grammar_1856, 'toid')
        # Obtaining the member '__getitem__' of a type (line 502)
        getitem___1858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 50), toid_1857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 502)
        subscript_call_result_1859 = invoke(stypy.reporting.localization.Localization(__file__, 502, 50), getitem___1858, root_1855)
        
        # Getting the type of 'False' (line 502)
        False_1860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 70), 'False', False)
        # Processing the call keyword arguments (line 502)
        kwargs_1861 = {}
        # Getting the type of 'parse' (line 502)
        parse_1851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 23), 'parse', False)
        # Calling parse(args, kwargs) (line 502)
        parse_call_result_1862 = invoke(stypy.reporting.localization.Localization(__file__, 502, 23), parse_1851, *[sent_1852, grammar_1853, tags_1854, subscript_call_result_1859, False_1860], **kwargs_1861)
        
        # Assigning a type to the variable 'call_assignment_16' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'call_assignment_16', parse_call_result_1862)
        
        # Assigning a Call to a Name (line 502):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 8), 'int')
        # Processing the call keyword arguments
        kwargs_1866 = {}
        # Getting the type of 'call_assignment_16' (line 502)
        call_assignment_16_1863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'call_assignment_16', False)
        # Obtaining the member '__getitem__' of a type (line 502)
        getitem___1864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), call_assignment_16_1863, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1867 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1864, *[int_1865], **kwargs_1866)
        
        # Assigning a type to the variable 'call_assignment_17' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'call_assignment_17', getitem___call_result_1867)
        
        # Assigning a Name to a Name (line 502):
        # Getting the type of 'call_assignment_17' (line 502)
        call_assignment_17_1868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'call_assignment_17')
        # Assigning a type to the variable 'chart' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'chart', call_assignment_17_1868)
        
        # Assigning a Call to a Name (line 502):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_1871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 8), 'int')
        # Processing the call keyword arguments
        kwargs_1872 = {}
        # Getting the type of 'call_assignment_16' (line 502)
        call_assignment_16_1869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'call_assignment_16', False)
        # Obtaining the member '__getitem__' of a type (line 502)
        getitem___1870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), call_assignment_16_1869, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_1873 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1870, *[int_1871], **kwargs_1872)
        
        # Assigning a type to the variable 'call_assignment_18' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'call_assignment_18', getitem___call_result_1873)
        
        # Assigning a Name to a Name (line 502):
        # Getting the type of 'call_assignment_18' (line 502)
        call_assignment_18_1874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'call_assignment_18')
        # Assigning a type to the variable 'start' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 15), 'start', call_assignment_18_1874)
        # Getting the type of 'start' (line 503)
        start_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 'start')
        # Testing if the type of an if condition is none (line 503)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 503, 8), start_1875):
            pass
        else:
            
            # Testing the type of an if condition (line 503)
            if_condition_1876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 503, 8), start_1875)
            # Assigning a type to the variable 'if_condition_1876' (line 503)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'if_condition_1876', if_condition_1876)
            # SSA begins for if statement (line 503)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 504):
            
            # Assigning a Call to a Name:
            
            # Call to mostprobablederivation(...): (line 504)
            # Processing the call arguments (line 504)
            # Getting the type of 'chart' (line 504)
            chart_1878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 42), 'chart', False)
            # Getting the type of 'start' (line 504)
            start_1879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 49), 'start', False)
            # Getting the type of 'grammar' (line 504)
            grammar_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 56), 'grammar', False)
            # Obtaining the member 'tolabel' of a type (line 504)
            tolabel_1881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 56), grammar_1880, 'tolabel')
            # Processing the call keyword arguments (line 504)
            kwargs_1882 = {}
            # Getting the type of 'mostprobablederivation' (line 504)
            mostprobablederivation_1877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 19), 'mostprobablederivation', False)
            # Calling mostprobablederivation(args, kwargs) (line 504)
            mostprobablederivation_call_result_1883 = invoke(stypy.reporting.localization.Localization(__file__, 504, 19), mostprobablederivation_1877, *[chart_1878, start_1879, tolabel_1881], **kwargs_1882)
            
            # Assigning a type to the variable 'call_assignment_19' (line 504)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'call_assignment_19', mostprobablederivation_call_result_1883)
            
            # Assigning a Call to a Name (line 504):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_1886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 12), 'int')
            # Processing the call keyword arguments
            kwargs_1887 = {}
            # Getting the type of 'call_assignment_19' (line 504)
            call_assignment_19_1884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'call_assignment_19', False)
            # Obtaining the member '__getitem__' of a type (line 504)
            getitem___1885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), call_assignment_19_1884, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_1888 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1885, *[int_1886], **kwargs_1887)
            
            # Assigning a type to the variable 'call_assignment_20' (line 504)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'call_assignment_20', getitem___call_result_1888)
            
            # Assigning a Name to a Name (line 504):
            # Getting the type of 'call_assignment_20' (line 504)
            call_assignment_20_1889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'call_assignment_20')
            # Assigning a type to the variable 't' (line 504)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 't', call_assignment_20_1889)
            
            # Assigning a Call to a Name (line 504):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_1892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 12), 'int')
            # Processing the call keyword arguments
            kwargs_1893 = {}
            # Getting the type of 'call_assignment_19' (line 504)
            call_assignment_19_1890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'call_assignment_19', False)
            # Obtaining the member '__getitem__' of a type (line 504)
            getitem___1891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), call_assignment_19_1890, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_1894 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1891, *[int_1892], **kwargs_1893)
            
            # Assigning a type to the variable 'call_assignment_21' (line 504)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'call_assignment_21', getitem___call_result_1894)
            
            # Assigning a Name to a Name (line 504):
            # Getting the type of 'call_assignment_21' (line 504)
            call_assignment_21_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'call_assignment_21')
            # Assigning a type to the variable 'p' (line 504)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 15), 'p', call_assignment_21_1895)
            
            # Call to exp(...): (line 506)
            # Processing the call arguments (line 506)
            
            # Getting the type of 'p' (line 506)
            p_1897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'p', False)
            # Applying the 'usub' unary operator (line 506)
            result___neg___1898 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 16), 'usub', p_1897)
            
            # Processing the call keyword arguments (line 506)
            kwargs_1899 = {}
            # Getting the type of 'exp' (line 506)
            exp_1896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 'exp', False)
            # Calling exp(args, kwargs) (line 506)
            exp_call_result_1900 = invoke(stypy.reporting.localization.Localization(__file__, 506, 12), exp_1896, *[result___neg___1898], **kwargs_1899)
            
            # SSA branch for the else part of an if statement (line 503)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 503)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'batch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'batch' in the type store
    # Getting the type of 'stypy_return_type' (line 492)
    stypy_return_type_1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1901)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'batch'
    return stypy_return_type_1901

# Assigning a type to the variable 'batch' (line 492)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'batch', batch)

@norecursion
def demo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'demo'
    module_type_store = module_type_store.open_function_context('demo', 511, 0, False)
    
    # Passed parameters checking function
    demo.stypy_localization = localization
    demo.stypy_type_of_self = None
    demo.stypy_type_store = module_type_store
    demo.stypy_function_name = 'demo'
    demo.stypy_param_names_list = []
    demo.stypy_varargs_param_name = None
    demo.stypy_kwargs_param_name = None
    demo.stypy_call_defaults = defaults
    demo.stypy_call_varargs = varargs
    demo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'demo', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'demo', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'demo(...)' code ##################

    
    # Assigning a List to a Name (line 512):
    
    # Assigning a List to a Name (line 512):
    
    # Obtaining an instance of the builtin type 'list' (line 512)
    list_1902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 512)
    # Adding element type (line 512)
    
    # Obtaining an instance of the builtin type 'tuple' (line 513)
    tuple_1903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 513)
    # Adding element type (line 513)
    
    # Obtaining an instance of the builtin type 'tuple' (line 513)
    tuple_1904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 513)
    # Adding element type (line 513)
    
    # Obtaining an instance of the builtin type 'tuple' (line 513)
    tuple_1905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 513)
    # Adding element type (line 513)
    str_1906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 11), 'str', 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 11), tuple_1905, str_1906)
    # Adding element type (line 513)
    str_1907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 16), 'str', 'VP2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 11), tuple_1905, str_1907)
    # Adding element type (line 513)
    str_1908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 23), 'str', 'VMFIN')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 11), tuple_1905, str_1908)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 10), tuple_1904, tuple_1905)
    # Adding element type (line 513)
    
    # Obtaining an instance of the builtin type 'tuple' (line 513)
    tuple_1909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 513)
    # Adding element type (line 513)
    
    # Obtaining an instance of the builtin type 'tuple' (line 513)
    tuple_1910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 513)
    # Adding element type (line 513)
    int_1911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 35), tuple_1910, int_1911)
    # Adding element type (line 513)
    int_1912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 35), tuple_1910, int_1912)
    # Adding element type (line 513)
    int_1913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 35), tuple_1910, int_1913)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 34), tuple_1909, tuple_1910)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 10), tuple_1904, tuple_1909)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 9), tuple_1903, tuple_1904)
    # Adding element type (line 513)
    
    # Call to log(...): (line 513)
    # Processing the call arguments (line 513)
    float_1915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 52), 'float')
    # Processing the call keyword arguments (line 513)
    kwargs_1916 = {}
    # Getting the type of 'log' (line 513)
    log_1914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 48), 'log', False)
    # Calling log(args, kwargs) (line 513)
    log_call_result_1917 = invoke(stypy.reporting.localization.Localization(__file__, 513, 48), log_1914, *[float_1915], **kwargs_1916)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 9), tuple_1903, log_call_result_1917)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), list_1902, tuple_1903)
    # Adding element type (line 512)
    
    # Obtaining an instance of the builtin type 'tuple' (line 514)
    tuple_1918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 514)
    # Adding element type (line 514)
    
    # Obtaining an instance of the builtin type 'tuple' (line 514)
    tuple_1919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 514)
    # Adding element type (line 514)
    
    # Obtaining an instance of the builtin type 'tuple' (line 514)
    tuple_1920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 514)
    # Adding element type (line 514)
    str_1921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 11), 'str', 'VP2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 11), tuple_1920, str_1921)
    # Adding element type (line 514)
    str_1922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 18), 'str', 'VP2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 11), tuple_1920, str_1922)
    # Adding element type (line 514)
    str_1923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 25), 'str', 'VAINF')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 11), tuple_1920, str_1923)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 10), tuple_1919, tuple_1920)
    # Adding element type (line 514)
    
    # Obtaining an instance of the builtin type 'tuple' (line 514)
    tuple_1924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 514)
    # Adding element type (line 514)
    
    # Obtaining an instance of the builtin type 'tuple' (line 514)
    tuple_1925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 514)
    # Adding element type (line 514)
    int_1926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 37), tuple_1925, int_1926)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 36), tuple_1924, tuple_1925)
    # Adding element type (line 514)
    
    # Obtaining an instance of the builtin type 'tuple' (line 514)
    tuple_1927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 514)
    # Adding element type (line 514)
    int_1928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 43), tuple_1927, int_1928)
    # Adding element type (line 514)
    int_1929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 43), tuple_1927, int_1929)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 36), tuple_1924, tuple_1927)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 10), tuple_1919, tuple_1924)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 9), tuple_1918, tuple_1919)
    # Adding element type (line 514)
    
    # Call to log(...): (line 514)
    # Processing the call arguments (line 514)
    float_1931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 56), 'float')
    # Processing the call keyword arguments (line 514)
    kwargs_1932 = {}
    # Getting the type of 'log' (line 514)
    log_1930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 52), 'log', False)
    # Calling log(args, kwargs) (line 514)
    log_call_result_1933 = invoke(stypy.reporting.localization.Localization(__file__, 514, 52), log_1930, *[float_1931], **kwargs_1932)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 9), tuple_1918, log_call_result_1933)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), list_1902, tuple_1918)
    # Adding element type (line 512)
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_1934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_1935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_1936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    str_1937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 11), 'str', 'VP2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 11), tuple_1936, str_1937)
    # Adding element type (line 515)
    str_1938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 18), 'str', 'PROAV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 11), tuple_1936, str_1938)
    # Adding element type (line 515)
    str_1939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 27), 'str', 'VVPP')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 11), tuple_1936, str_1939)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 10), tuple_1935, tuple_1936)
    # Adding element type (line 515)
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_1940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_1941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    int_1942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 38), tuple_1941, int_1942)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 37), tuple_1940, tuple_1941)
    # Adding element type (line 515)
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_1943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    int_1944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 44), tuple_1943, int_1944)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 37), tuple_1940, tuple_1943)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 10), tuple_1935, tuple_1940)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 9), tuple_1934, tuple_1935)
    # Adding element type (line 515)
    
    # Call to log(...): (line 515)
    # Processing the call arguments (line 515)
    float_1946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 55), 'float')
    # Processing the call keyword arguments (line 515)
    kwargs_1947 = {}
    # Getting the type of 'log' (line 515)
    log_1945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 51), 'log', False)
    # Calling log(args, kwargs) (line 515)
    log_call_result_1948 = invoke(stypy.reporting.localization.Localization(__file__, 515, 51), log_1945, *[float_1946], **kwargs_1947)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 9), tuple_1934, log_call_result_1948)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), list_1902, tuple_1934)
    # Adding element type (line 512)
    
    # Obtaining an instance of the builtin type 'tuple' (line 516)
    tuple_1949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 516)
    # Adding element type (line 516)
    
    # Obtaining an instance of the builtin type 'tuple' (line 516)
    tuple_1950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 516)
    # Adding element type (line 516)
    
    # Obtaining an instance of the builtin type 'tuple' (line 516)
    tuple_1951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 516)
    # Adding element type (line 516)
    str_1952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 11), 'str', 'VP2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 11), tuple_1951, str_1952)
    # Adding element type (line 516)
    str_1953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 18), 'str', 'VP2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 11), tuple_1951, str_1953)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 10), tuple_1950, tuple_1951)
    # Adding element type (line 516)
    
    # Obtaining an instance of the builtin type 'tuple' (line 516)
    tuple_1954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 516)
    # Adding element type (line 516)
    
    # Obtaining an instance of the builtin type 'tuple' (line 516)
    tuple_1955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 516)
    # Adding element type (line 516)
    int_1956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 28), tuple_1955, int_1956)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 27), tuple_1954, tuple_1955)
    # Adding element type (line 516)
    
    # Obtaining an instance of the builtin type 'tuple' (line 516)
    tuple_1957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 516)
    # Adding element type (line 516)
    int_1958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 34), tuple_1957, int_1958)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 27), tuple_1954, tuple_1957)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 10), tuple_1950, tuple_1954)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 9), tuple_1949, tuple_1950)
    # Adding element type (line 516)
    
    # Call to log(...): (line 516)
    # Processing the call arguments (line 516)
    float_1960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 45), 'float')
    # Processing the call keyword arguments (line 516)
    kwargs_1961 = {}
    # Getting the type of 'log' (line 516)
    log_1959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 41), 'log', False)
    # Calling log(args, kwargs) (line 516)
    log_call_result_1962 = invoke(stypy.reporting.localization.Localization(__file__, 516, 41), log_1959, *[float_1960], **kwargs_1961)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 9), tuple_1949, log_call_result_1962)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), list_1902, tuple_1949)
    
    # Assigning a type to the variable 'rules' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'rules', list_1902)
    
    # Assigning a List to a Name (line 517):
    
    # Assigning a List to a Name (line 517):
    
    # Obtaining an instance of the builtin type 'list' (line 517)
    list_1963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 517)
    # Adding element type (line 517)
    
    # Obtaining an instance of the builtin type 'tuple' (line 518)
    tuple_1964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 518)
    # Adding element type (line 518)
    
    # Obtaining an instance of the builtin type 'tuple' (line 518)
    tuple_1965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 518)
    # Adding element type (line 518)
    
    # Obtaining an instance of the builtin type 'tuple' (line 518)
    tuple_1966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 518)
    # Adding element type (line 518)
    str_1967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 11), 'str', 'PROAV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 11), tuple_1966, str_1967)
    # Adding element type (line 518)
    str_1968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 20), 'str', 'Epsilon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 11), tuple_1966, str_1968)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 10), tuple_1965, tuple_1966)
    # Adding element type (line 518)
    str_1969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 32), 'str', 'Darueber')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 10), tuple_1965, str_1969)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 9), tuple_1964, tuple_1965)
    # Adding element type (line 518)
    float_1970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 9), tuple_1964, float_1970)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 14), list_1963, tuple_1964)
    # Adding element type (line 517)
    
    # Obtaining an instance of the builtin type 'tuple' (line 519)
    tuple_1971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 519)
    # Adding element type (line 519)
    
    # Obtaining an instance of the builtin type 'tuple' (line 519)
    tuple_1972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 519)
    # Adding element type (line 519)
    
    # Obtaining an instance of the builtin type 'tuple' (line 519)
    tuple_1973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 519)
    # Adding element type (line 519)
    str_1974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 11), 'str', 'VAINF')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 11), tuple_1973, str_1974)
    # Adding element type (line 519)
    str_1975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 20), 'str', 'Epsilon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 11), tuple_1973, str_1975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 10), tuple_1972, tuple_1973)
    # Adding element type (line 519)
    str_1976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 32), 'str', 'werden')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 10), tuple_1972, str_1976)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 9), tuple_1971, tuple_1972)
    # Adding element type (line 519)
    float_1977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 9), tuple_1971, float_1977)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 14), list_1963, tuple_1971)
    # Adding element type (line 517)
    
    # Obtaining an instance of the builtin type 'tuple' (line 520)
    tuple_1978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 520)
    # Adding element type (line 520)
    
    # Obtaining an instance of the builtin type 'tuple' (line 520)
    tuple_1979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 520)
    # Adding element type (line 520)
    
    # Obtaining an instance of the builtin type 'tuple' (line 520)
    tuple_1980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 520)
    # Adding element type (line 520)
    str_1981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 11), 'str', 'VMFIN')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 11), tuple_1980, str_1981)
    # Adding element type (line 520)
    str_1982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 20), 'str', 'Epsilon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 11), tuple_1980, str_1982)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 10), tuple_1979, tuple_1980)
    # Adding element type (line 520)
    str_1983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 32), 'str', 'muss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 10), tuple_1979, str_1983)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 9), tuple_1978, tuple_1979)
    # Adding element type (line 520)
    float_1984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 41), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 9), tuple_1978, float_1984)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 14), list_1963, tuple_1978)
    # Adding element type (line 517)
    
    # Obtaining an instance of the builtin type 'tuple' (line 521)
    tuple_1985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 521)
    # Adding element type (line 521)
    
    # Obtaining an instance of the builtin type 'tuple' (line 521)
    tuple_1986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 521)
    # Adding element type (line 521)
    
    # Obtaining an instance of the builtin type 'tuple' (line 521)
    tuple_1987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 521)
    # Adding element type (line 521)
    str_1988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 11), 'str', 'VVPP')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 11), tuple_1987, str_1988)
    # Adding element type (line 521)
    str_1989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 19), 'str', 'Epsilon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 11), tuple_1987, str_1989)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 10), tuple_1986, tuple_1987)
    # Adding element type (line 521)
    str_1990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 31), 'str', 'nachgedacht')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 10), tuple_1986, str_1990)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 9), tuple_1985, tuple_1986)
    # Adding element type (line 521)
    float_1991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 9), tuple_1985, float_1991)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 14), list_1963, tuple_1985)
    
    # Assigning a type to the variable 'lexicon' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'lexicon', list_1963)
    
    # Assigning a Call to a Name (line 522):
    
    # Assigning a Call to a Name (line 522):
    
    # Call to splitgrammar(...): (line 522)
    # Processing the call arguments (line 522)
    # Getting the type of 'rules' (line 522)
    rules_1993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 27), 'rules', False)
    # Getting the type of 'lexicon' (line 522)
    lexicon_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 34), 'lexicon', False)
    # Processing the call keyword arguments (line 522)
    kwargs_1995 = {}
    # Getting the type of 'splitgrammar' (line 522)
    splitgrammar_1992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 14), 'splitgrammar', False)
    # Calling splitgrammar(args, kwargs) (line 522)
    splitgrammar_call_result_1996 = invoke(stypy.reporting.localization.Localization(__file__, 522, 14), splitgrammar_1992, *[rules_1993, lexicon_1994], **kwargs_1995)
    
    # Assigning a type to the variable 'grammar' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'grammar', splitgrammar_call_result_1996)
    
    # Assigning a Call to a Tuple (line 524):
    
    # Assigning a Call to a Name:
    
    # Call to parse(...): (line 524)
    # Processing the call arguments (line 524)
    
    # Call to split(...): (line 524)
    # Processing the call keyword arguments (line 524)
    kwargs_2000 = {}
    str_1998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 25), 'str', 'Darueber muss nachgedacht werden')
    # Obtaining the member 'split' of a type (line 524)
    split_1999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 25), str_1998, 'split')
    # Calling split(args, kwargs) (line 524)
    split_call_result_2001 = invoke(stypy.reporting.localization.Localization(__file__, 524, 25), split_1999, *[], **kwargs_2000)
    
    # Getting the type of 'grammar' (line 525)
    grammar_2002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'grammar', False)
    
    # Call to split(...): (line 525)
    # Processing the call keyword arguments (line 525)
    kwargs_2005 = {}
    str_2003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 34), 'str', 'PROAV VMFIN VVPP VAINF')
    # Obtaining the member 'split' of a type (line 525)
    split_2004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 34), str_2003, 'split')
    # Calling split(args, kwargs) (line 525)
    split_call_result_2006 = invoke(stypy.reporting.localization.Localization(__file__, 525, 34), split_2004, *[], **kwargs_2005)
    
    
    # Obtaining the type of the subscript
    str_2007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 81), 'str', 'S')
    # Getting the type of 'grammar' (line 525)
    grammar_2008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 68), 'grammar', False)
    # Obtaining the member 'toid' of a type (line 525)
    toid_2009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 68), grammar_2008, 'toid')
    # Obtaining the member '__getitem__' of a type (line 525)
    getitem___2010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 68), toid_2009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 525)
    subscript_call_result_2011 = invoke(stypy.reporting.localization.Localization(__file__, 525, 68), getitem___2010, str_2007)
    
    # Getting the type of 'False' (line 525)
    False_2012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 87), 'False', False)
    # Processing the call keyword arguments (line 524)
    kwargs_2013 = {}
    # Getting the type of 'parse' (line 524)
    parse_1997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'parse', False)
    # Calling parse(args, kwargs) (line 524)
    parse_call_result_2014 = invoke(stypy.reporting.localization.Localization(__file__, 524, 19), parse_1997, *[split_call_result_2001, grammar_2002, split_call_result_2006, subscript_call_result_2011, False_2012], **kwargs_2013)
    
    # Assigning a type to the variable 'call_assignment_22' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'call_assignment_22', parse_call_result_2014)
    
    # Assigning a Call to a Name (line 524):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_2017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 4), 'int')
    # Processing the call keyword arguments
    kwargs_2018 = {}
    # Getting the type of 'call_assignment_22' (line 524)
    call_assignment_22_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'call_assignment_22', False)
    # Obtaining the member '__getitem__' of a type (line 524)
    getitem___2016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 4), call_assignment_22_2015, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_2019 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___2016, *[int_2017], **kwargs_2018)
    
    # Assigning a type to the variable 'call_assignment_23' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'call_assignment_23', getitem___call_result_2019)
    
    # Assigning a Name to a Name (line 524):
    # Getting the type of 'call_assignment_23' (line 524)
    call_assignment_23_2020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'call_assignment_23')
    # Assigning a type to the variable 'chart' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'chart', call_assignment_23_2020)
    
    # Assigning a Call to a Name (line 524):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_2023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 4), 'int')
    # Processing the call keyword arguments
    kwargs_2024 = {}
    # Getting the type of 'call_assignment_22' (line 524)
    call_assignment_22_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'call_assignment_22', False)
    # Obtaining the member '__getitem__' of a type (line 524)
    getitem___2022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 4), call_assignment_22_2021, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_2025 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___2022, *[int_2023], **kwargs_2024)
    
    # Assigning a type to the variable 'call_assignment_24' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'call_assignment_24', getitem___call_result_2025)
    
    # Assigning a Name to a Name (line 524):
    # Getting the type of 'call_assignment_24' (line 524)
    call_assignment_24_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'call_assignment_24')
    # Assigning a type to the variable 'start' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 11), 'start', call_assignment_24_2026)
    
    # Call to pprint_chart(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'chart' (line 526)
    chart_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 17), 'chart', False)
    
    # Call to split(...): (line 526)
    # Processing the call keyword arguments (line 526)
    kwargs_2031 = {}
    str_2029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 24), 'str', 'Darueber muss nachgedacht werden')
    # Obtaining the member 'split' of a type (line 526)
    split_2030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 24), str_2029, 'split')
    # Calling split(args, kwargs) (line 526)
    split_call_result_2032 = invoke(stypy.reporting.localization.Localization(__file__, 526, 24), split_2030, *[], **kwargs_2031)
    
    # Getting the type of 'grammar' (line 527)
    grammar_2033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 17), 'grammar', False)
    # Obtaining the member 'tolabel' of a type (line 527)
    tolabel_2034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 17), grammar_2033, 'tolabel')
    # Processing the call keyword arguments (line 526)
    kwargs_2035 = {}
    # Getting the type of 'pprint_chart' (line 526)
    pprint_chart_2027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'pprint_chart', False)
    # Calling pprint_chart(args, kwargs) (line 526)
    pprint_chart_call_result_2036 = invoke(stypy.reporting.localization.Localization(__file__, 526, 4), pprint_chart_2027, *[chart_2028, split_call_result_2032, tolabel_2034], **kwargs_2035)
    
    # Evaluating assert statement condition
    
    
    # Call to mostprobablederivation(...): (line 528)
    # Processing the call arguments (line 528)
    # Getting the type of 'chart' (line 528)
    chart_2038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 35), 'chart', False)
    # Getting the type of 'start' (line 528)
    start_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 42), 'start', False)
    # Getting the type of 'grammar' (line 528)
    grammar_2040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 49), 'grammar', False)
    # Obtaining the member 'tolabel' of a type (line 528)
    tolabel_2041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 49), grammar_2040, 'tolabel')
    # Processing the call keyword arguments (line 528)
    kwargs_2042 = {}
    # Getting the type of 'mostprobablederivation' (line 528)
    mostprobablederivation_2037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'mostprobablederivation', False)
    # Calling mostprobablederivation(args, kwargs) (line 528)
    mostprobablederivation_call_result_2043 = invoke(stypy.reporting.localization.Localization(__file__, 528, 12), mostprobablederivation_2037, *[chart_2038, start_2039, tolabel_2041], **kwargs_2042)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 529)
    tuple_2044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 529)
    # Adding element type (line 529)
    str_2045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 13), 'str', '(S (VP2 (VP2 (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 13), tuple_2044, str_2045)
    # Adding element type (line 529)
    
    
    # Call to log(...): (line 529)
    # Processing the call arguments (line 529)
    float_2047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 76), 'float')
    # Processing the call keyword arguments (line 529)
    kwargs_2048 = {}
    # Getting the type of 'log' (line 529)
    log_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 72), 'log', False)
    # Calling log(args, kwargs) (line 529)
    log_call_result_2049 = invoke(stypy.reporting.localization.Localization(__file__, 529, 72), log_2046, *[float_2047], **kwargs_2048)
    
    # Applying the 'usub' unary operator (line 529)
    result___neg___2050 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 71), 'usub', log_call_result_2049)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 13), tuple_2044, result___neg___2050)
    
    # Applying the binary operator '==' (line 528)
    result_eq_2051 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 12), '==', mostprobablederivation_call_result_2043, tuple_2044)
    
    assert_2052 = result_eq_2051
    # Assigning a type to the variable 'assert_2052' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'assert_2052', result_eq_2051)
    # Evaluating assert statement condition
    
    # Call to do(...): (line 530)
    # Processing the call arguments (line 530)
    str_2054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 14), 'str', 'Darueber muss nachgedacht werden')
    # Getting the type of 'grammar' (line 530)
    grammar_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 50), 'grammar', False)
    # Processing the call keyword arguments (line 530)
    kwargs_2056 = {}
    # Getting the type of 'do' (line 530)
    do_2053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'do', False)
    # Calling do(args, kwargs) (line 530)
    do_call_result_2057 = invoke(stypy.reporting.localization.Localization(__file__, 530, 11), do_2053, *[str_2054, grammar_2055], **kwargs_2056)
    
    assert_2058 = do_call_result_2057
    # Assigning a type to the variable 'assert_2058' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'assert_2058', do_call_result_2057)
    # Evaluating assert statement condition
    
    # Call to do(...): (line 531)
    # Processing the call arguments (line 531)
    str_2060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 14), 'str', 'Darueber muss nachgedacht werden werden')
    # Getting the type of 'grammar' (line 531)
    grammar_2061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 57), 'grammar', False)
    # Processing the call keyword arguments (line 531)
    kwargs_2062 = {}
    # Getting the type of 'do' (line 531)
    do_2059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 11), 'do', False)
    # Calling do(args, kwargs) (line 531)
    do_call_result_2063 = invoke(stypy.reporting.localization.Localization(__file__, 531, 11), do_2059, *[str_2060, grammar_2061], **kwargs_2062)
    
    assert_2064 = do_call_result_2063
    # Assigning a type to the variable 'assert_2064' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'assert_2064', do_call_result_2063)
    # Evaluating assert statement condition
    
    # Call to do(...): (line 532)
    # Processing the call arguments (line 532)
    str_2066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 14), 'str', 'Darueber muss nachgedacht werden werden werden')
    # Getting the type of 'grammar' (line 532)
    grammar_2067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 64), 'grammar', False)
    # Processing the call keyword arguments (line 532)
    kwargs_2068 = {}
    # Getting the type of 'do' (line 532)
    do_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 11), 'do', False)
    # Calling do(args, kwargs) (line 532)
    do_call_result_2069 = invoke(stypy.reporting.localization.Localization(__file__, 532, 11), do_2065, *[str_2066, grammar_2067], **kwargs_2068)
    
    assert_2070 = do_call_result_2069
    # Assigning a type to the variable 'assert_2070' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'assert_2070', do_call_result_2069)
    # Evaluating assert statement condition
    
    
    # Call to do(...): (line 534)
    # Processing the call arguments (line 534)
    str_2072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 18), 'str', 'werden nachgedacht muss Darueber')
    # Getting the type of 'grammar' (line 534)
    grammar_2073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 54), 'grammar', False)
    # Processing the call keyword arguments (line 534)
    kwargs_2074 = {}
    # Getting the type of 'do' (line 534)
    do_2071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 15), 'do', False)
    # Calling do(args, kwargs) (line 534)
    do_call_result_2075 = invoke(stypy.reporting.localization.Localization(__file__, 534, 15), do_2071, *[str_2072, grammar_2073], **kwargs_2074)
    
    # Applying the 'not' unary operator (line 534)
    result_not__2076 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 11), 'not', do_call_result_2075)
    
    assert_2077 = result_not__2076
    # Assigning a type to the variable 'assert_2077' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'assert_2077', result_not__2076)
    
    # ################# End of 'demo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'demo' in the type store
    # Getting the type of 'stypy_return_type' (line 511)
    stypy_return_type_2078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2078)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'demo'
    return stypy_return_type_2078

# Assigning a type to the variable 'demo' (line 511)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), 'demo', demo)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 538, 0, False)
    
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

    
    
    # Call to range(...): (line 542)
    # Processing the call arguments (line 542)
    int_2080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 19), 'int')
    # Processing the call keyword arguments (line 542)
    kwargs_2081 = {}
    # Getting the type of 'range' (line 542)
    range_2079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 13), 'range', False)
    # Calling range(args, kwargs) (line 542)
    range_call_result_2082 = invoke(stypy.reporting.localization.Localization(__file__, 542, 13), range_2079, *[int_2080], **kwargs_2081)
    
    # Assigning a type to the variable 'range_call_result_2082' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'range_call_result_2082', range_call_result_2082)
    # Testing if the for loop is going to be iterated (line 542)
    # Testing the type of a for loop iterable (line 542)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 542, 4), range_call_result_2082)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 542, 4), range_call_result_2082):
        # Getting the type of the for loop variable (line 542)
        for_loop_var_2083 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 542, 4), range_call_result_2082)
        # Assigning a type to the variable 'i' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'i', for_loop_var_2083)
        # SSA begins for a for statement (line 542)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to demo(...): (line 543)
        # Processing the call keyword arguments (line 543)
        kwargs_2085 = {}
        # Getting the type of 'demo' (line 543)
        demo_2084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'demo', False)
        # Calling demo(args, kwargs) (line 543)
        demo_call_result_2086 = invoke(stypy.reporting.localization.Localization(__file__, 543, 8), demo_2084, *[], **kwargs_2085)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 572)
    True_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'stypy_return_type', True_2087)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 538)
    stypy_return_type_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2088)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_2088

# Assigning a type to the variable 'run' (line 538)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'run', run)

# Call to run(...): (line 575)
# Processing the call keyword arguments (line 575)
kwargs_2090 = {}
# Getting the type of 'run' (line 575)
run_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 0), 'run', False)
# Calling run(args, kwargs) (line 575)
run_call_result_2091 = invoke(stypy.reporting.localization.Localization(__file__, 575, 0), run_2089, *[], **kwargs_2090)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

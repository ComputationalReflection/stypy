
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A Python implementation of the _bh_ Olden benchmark.
3: The Olden benchmark implements the Barnes-Hut benchmark
4: that is decribed in:
5: 
6: J. Barnes and P. Hut, "A hierarchical o(N log N) force-calculation algorithm",
7: Nature, 324:446-449, Dec. 1986
8: 
9: The original code in the Olden benchmark suite is derived from the
10: ftp://hubble.ifa.hawaii.edu/pub/barnes/treecode
11: source distributed by Barnes.
12: 
13: This code comes from the third Java version.
14: This uses copy() instead of Vec3.clone(), and it's adapted for ShedSkin.
15: '''
16: 
17: from time import clock
18: from sys import stderr, maxint, argv
19: from copy import copy
20: from math import sqrt, pi, floor
21: 
22: class Random(object):
23:     '''
24:     Basic uniform random generator: Minimal Standard in Park and
25:     Miller (1988): "Random Number Generators: Good Ones Are Hard to
26:     Find", Comm. of the ACM, 31, 1192-1201.
27:     Parameters: m = 2^31-1, a=48271.
28: 
29:     Adapted from Pascal code by Jesper Lund:
30:     http:#www.gnu-pascal.de/crystal/gpc/en/mail1390.html
31:     '''
32:     __slots__ = ["seed"]
33:     m = maxint
34:     a = 48271
35:     q = m / a
36:     r = m % a
37: 
38:     def __init__(self, the_seed):
39:         self.seed = the_seed
40: 
41:     def uniform(self, min, max):
42:         k = self.seed / Random.q
43:         self.seed = Random.a * (self.seed - k * Random.q) - Random.r * k
44:         if self.seed < 1:
45:             self.seed += Random.m
46:         r = float(self.seed) / Random.m
47:         return r * (max - min) + min
48: 
49: 
50: class Vec3(object):
51:     '''
52:     A class representing a three dimensional vector that implements
53:     several math operations.  To improve speed we implement the
54:     vector as an array of doubles rather than use the exising
55:     code in the java.util.class.
56:     '''
57:     __slots__ = ["d0", "d1", "d2"]
58:     # The number of dimensions in the vector
59:     NDIM = 3
60: 
61:     def __init__(self):
62:         '''Construct an empty 3 dimensional vector for use in Barnes-Hut algorithm.'''
63:         self.d0 = 0.0
64:         self.d1 = 0.0
65:         self.d2 = 0.0
66: 
67:     def __getitem__(self, i):
68:         '''
69:         Return the value at the i'th index of the vector.
70:         @param i the vector index
71:         @return the value at the i'th index of the vector.
72:         '''
73:         if i == 0:
74:             return self.d0
75:         elif i == 1:
76:             return self.d1
77:         else:
78:             return self.d2
79: 
80:     def __setitem__(self, i, v):
81:         '''
82:         Set the value of the i'th index of the vector.
83:         @param i the vector index
84:         @param v the value to store
85:         '''
86:         if i == 0:
87:             self.d0 = v
88:         elif i == 1:
89:             self.d1 = v
90:         else:
91:             self.d2 = v
92: 
93:     def __iadd__(self, u):
94:         '''
95:         Add two vectors and the result is placed in self vector.
96:         @param u the other operand of the addition
97:         '''
98:         self.d0 += u.d0
99:         self.d1 += u.d1
100:         self.d2 += u.d2
101:         return self
102: 
103:     def __isub__(self, u):
104:         '''
105:         Subtract two vectors and the result is placed in self vector.
106:         This vector contain the first operand.
107:         @param u the other operand of the subtraction.
108:         '''
109:         self.d0 -= u.d0
110:         self.d1 -= u.d1
111:         self.d2 -= u.d2
112:         return self
113: 
114:     def __imul__(self, s):
115:         '''
116:         Multiply the vector times a scalar.
117:         @param s the scalar value
118:         '''
119:         self.d0 *= s
120:         self.d1 *= s
121:         self.d2 *= s
122:         return self
123: 
124:     def __idiv__(self, s):
125:         '''
126:         Divide each element of the vector by a scalar value.
127:         @param s the scalar value.
128:         '''
129:         self.d0 /= s
130:         self.d1 /= s
131:         self.d2 /= s
132:         return self
133: 
134:     def add_scalar(self, u, s):
135:         self.d0 = u.d0 + s
136:         self.d1 = u.d1 + s
137:         self.d2 = u.d2 + s
138: 
139:     def subtraction2(self, u, v):
140:         '''
141:         Subtract two vectors and the result is placed in self vector.
142:         @param u the first operand of the subtraction.
143:         @param v the second opernd of the subtraction
144:         '''
145:         self.d0 = u.d0 - v.d0
146:         self.d1 = u.d1 - v.d1
147:         self.d2 = u.d2 - v.d2
148: 
149:     def mult_scalar2(self, u, s):
150:         '''
151:         Multiply the vector times a scalar and place the result in self vector.
152:         @param u the vector
153:         @param s the scalar value
154:         '''
155:         self.d0 = u.d0 * s
156:         self.d1 = u.d1 * s
157:         self.d2 = u.d2 * s
158: 
159:     def dot(self):
160:         '''
161:         Return the dot product of a vector.
162:         @return the dot product of a vector.
163:         '''
164:         return self.d0 * self.d0 + self.d1 * self.d1 + self.d2 * self.d2
165: 
166:     def __repr__(self):
167:         return "%.17f %.17f %.17f " % (self.d0, self.d1, self.d2)
168: 
169: 
170: class HG(object):
171:     '''
172:     A sub class which is used to compute and save information during the
173:     gravity computation phase.
174:     '''
175:     __slots__ = ["pskip", "pos0", "phi0", "acc0"]
176:     def __init__(self, b, p):
177:         '''
178:         Create a  object.
179:         @param b the body object
180:         @param p a vector that represents the body
181:         '''
182:         # Body to skip in force evaluation
183:         self.pskip = b
184: 
185:         # Poat which to evaluate field
186:         self.pos0 = copy(p)
187: 
188:         # Computed potential at pos0
189:         self.phi0 = 0.0
190: 
191:         # computed acceleration at pos0
192:         self.acc0 = Vec3()
193: 
194: 
195: class Node(object):
196:     '''A class that represents the common fields of a cell or body data structure.'''
197:     # highest bit of coord
198:     IMAX = 1073741824
199: 
200:     # potential softening parameter
201:     EPS = 0.05
202: 
203:     def __init__(self):
204:         '''Construct an empty node'''
205:         self.mass = 0.0 # mass of the node
206:         self.pos = Vec3() # Position of the node
207: 
208:     def load_tree(self, p, xpic, l, root):
209:         raise NotImplementedError()
210: 
211:     def hack_cofm(self):
212:         raise NotImplementedError()
213: 
214:     def walk_sub_tree(self, dsq, hg):
215:         raise NotImplementedError()
216: 
217:     @staticmethod
218:     def old_sub_index(ic, l):
219:         i = 0
220:         for k in xrange(Vec3.NDIM):
221:             if (int(ic[k]) & l) != 0:
222:                 i += Cell.NSUB >> (k + 1)
223:         return i
224: 
225:     def __repr__(self):
226:         return "%f : %f" % (self.mass, self.pos)
227: 
228:     def grav_sub(self, hg):
229:         '''Compute a single body-body or body-cell interaction'''
230:         dr = Vec3()
231:         dr.subtraction2(self.pos, hg.pos0)
232: 
233:         drsq = dr.dot() + (Node.EPS * Node.EPS)
234:         drabs = sqrt(drsq)
235: 
236:         phii = self.mass / drabs
237:         hg.phi0 -= phii
238:         mor3 = phii / drsq
239:         dr *= mor3
240:         hg.acc0 += dr
241:         return hg
242: 
243: 
244: class Body(Node):
245:     '''A class used to representing particles in the N-body simulation.'''
246:     def __init__(self):
247:         '''Create an empty body.'''
248:         Node.__init__(self)
249:         self.vel = Vec3()
250:         self.acc = Vec3()
251:         self.new_acc = Vec3()
252:         self.phi = 0.0
253: 
254:     def expand_box(self, tree, nsteps):
255:         '''
256:         Enlarge cubical "box", salvaging existing tree structure.
257:         @param tree the root of the tree.
258:         @param nsteps the current time step
259:         '''
260:         rmid = Vec3()
261: 
262:         inbox = self.ic_test(tree)
263:         while not inbox:
264:             rsize = tree.rsize
265:             rmid.add_scalar(tree.rmin, 0.5 * rsize)
266: 
267:             for k in xrange(Vec3.NDIM):
268:                 if self.pos[k] < rmid[k]:
269:                     rmin = tree.rmin[k]
270:                     tree.rmin[k] = rmin - rsize
271: 
272:             tree.rsize = 2.0 * rsize
273:             if tree.root is not None:
274:                 ic = tree.int_coord(rmid)
275:                 if ic is None:
276:                     raise Exception("Value is out of bounds")
277:                 k = Node.old_sub_index(ic, Node.IMAX >> 1)
278:                 newt = Cell()
279:                 newt.subp[k] = tree.root
280:                 tree.root = newt
281:                 inbox = self.ic_test(tree)
282: 
283:     def ic_test(self, tree):
284:         '''Check the bounds of the body and return True if it isn't in the correct bounds.'''
285:         pos0 = self.pos[0]
286:         pos1 = self.pos[1]
287:         pos2 = self.pos[2]
288: 
289:         # by default, it is in bounds
290:         result = True
291: 
292:         xsc = (pos0 - tree.rmin[0]) / tree.rsize
293:         if not (0.0 < xsc and xsc < 1.0):
294:             result = False
295: 
296:         xsc = (pos1 - tree.rmin[1]) / tree.rsize
297:         if not (0.0 < xsc and xsc < 1.0):
298:             result = False
299: 
300:         xsc = (pos2 - tree.rmin[2]) / tree.rsize
301:         if not (0.0 < xsc and xsc < 1.0):
302:             result = False
303: 
304:         return result
305: 
306:     def load_tree(self, p, xpic, l, tree):
307:         '''
308:         Descend and insert particle.  We're at a body so we need to
309:         create a cell and attach self body to the cell.
310:         @param p the body to insert
311:         @param xpic
312:         @param l
313:         @param tree the root of the data structure
314:         @return the subtree with the body inserted
315:         '''
316:         # create a Cell
317:         retval = Cell()
318:         si = self.sub_index(tree, l)
319:         # attach self node to the cell
320:         retval.subp[si] = self
321: 
322:         # move down one level
323:         si = Node.old_sub_index(xpic, l)
324:         rt = retval.subp[si]
325:         if rt is not None:
326:             retval.subp[si] = rt.load_tree(p, xpic, l >> 1, tree)
327:         else:
328:             retval.subp[si] = p
329:         return retval
330: 
331:     def hack_cofm(self):
332:         '''
333:         Descend tree finding center of mass coordinates
334:         @return the mass of self node
335:         '''
336:         return self.mass
337: 
338:     def sub_index(self, tree, l):
339:         '''
340:         Determine which subcell to select.
341:         Combination of int_coord and old_sub_index.
342:         @param t the root of the tree
343:         '''
344:         xp = Vec3()
345: 
346:         xsc = (self.pos[0] - tree.rmin[0]) / tree.rsize
347:         xp[0] = floor(Node.IMAX * xsc)
348: 
349:         xsc = (self.pos[1] - tree.rmin[1]) / tree.rsize
350:         xp[1] = floor(Node.IMAX * xsc)
351: 
352:         xsc = (self.pos[2] - tree.rmin[2]) / tree.rsize
353:         xp[2] = floor(Node.IMAX * xsc)
354: 
355:         i = 0
356:         for k in xrange(Vec3.NDIM):
357:             if (int(xp[k]) & l) != 0:
358:                 i += Cell.NSUB >> (k + 1)
359:         return i
360: 
361:     def hack_gravity(self, rsize, root):
362:         '''
363:         Evaluate gravitational field on the body.
364:         The original olden version calls a routine named "walkscan",
365:         but we use the same name that is in the Barnes code.
366:         '''
367:         hg = HG(self, self.pos)
368:         hg = root.walk_sub_tree(rsize * rsize, hg)
369:         self.phi = hg.phi0
370:         self.new_acc = hg.acc0
371: 
372:     def walk_sub_tree(self, dsq, hg):
373:         '''Recursively walk the tree to do hackwalk calculation'''
374:         if self != hg.pskip:
375:             hg = self.grav_sub(hg)
376:         return hg
377: 
378:     def __repr__(self):
379:         '''
380:         Return a string represenation of a body.
381:         @return a string represenation of a body.
382:         '''
383:         return "Body " + Node.__repr__(self)
384: 
385: 
386: class Cell(Node):
387:     '''A class used to represent internal nodes in the tree'''
388:     # subcells per cell
389:     NSUB = 8 # 1 << NDIM
390: 
391:     def __init__(self):
392:         # The children of self cell node.  Each entry may contain either
393:         # another cell or a body.
394:         Node.__init__(self)
395:         self.subp = [None] * Cell.NSUB
396: 
397:     def load_tree(self, p, xpic, l, tree):
398:         '''
399:         Descend and insert particle.  We're at a cell so
400:         we need to move down the tree.
401:         @param p the body to insert into the tree
402:         @param xpic
403:         @param l
404:         @param tree the root of the tree
405:         @return the subtree with the body inserted
406:         '''
407:         # move down one level
408:         si = Node.old_sub_index(xpic, l)
409:         rt = self.subp[si]
410:         if rt is not None:
411:             self.subp[si] = rt.load_tree(p, xpic, l >> 1, tree)
412:         else:
413:             self.subp[si] = p
414:         return self
415: 
416:     def hack_cofm(self):
417:         '''
418:         Descend tree finding center of mass coordinates
419:         @return the mass of self node
420:         '''
421:         mq = 0.0
422:         tmp_pos = Vec3()
423:         tmpv = Vec3()
424:         for i in xrange(Cell.NSUB):
425:             r = self.subp[i]
426:             if r is not None:
427:                 mr = r.hack_cofm()
428:                 mq = mr + mq
429:                 tmpv.mult_scalar2(r.pos, mr)
430:                 tmp_pos += tmpv
431:         self.mass = mq
432:         self.pos = tmp_pos
433:         self.pos /= self.mass
434:         return mq
435: 
436: 
437:     def walk_sub_tree(self, dsq, hg):
438:         '''Recursively walk the tree to do hackwalk calculation'''
439:         if self.subdiv_p(dsq, hg):
440:             for k in xrange(Cell.NSUB):
441:                 r = self.subp[k]
442:                 if r is not None:
443:                     hg = r.walk_sub_tree(dsq / 4.0, hg)
444:         else:
445:             hg = self.grav_sub(hg)
446:         return hg
447: 
448:     def subdiv_p(self, dsq, hg):
449:         '''
450:         Decide if the cell is too close to accept as a single term.
451:         @return True if the cell is too close.
452:         '''
453:         dr = Vec3()
454:         dr.subtraction2(self.pos, hg.pos0)
455:         drsq = dr.dot()
456: 
457:         # in the original olden version drsp is multiplied by 1.0
458:         return drsq < dsq
459: 
460:     def __repr__(self):
461:         '''
462:         Return a string represenation of a cell.
463:         @return a string represenation of a cell.
464:         '''
465:         return "Cell " + Node.__repr__(self)
466: 
467: 
468: class Tree:
469:     '''
470:     A class that represents the root of the data structure used
471:     to represent the N-bodies in the Barnes-Hut algorithm.
472:     '''
473:     def __init__(self):
474:         '''Construct the root of the data structure that represents the N-bodies.'''
475:         self.bodies = [] # The complete list of bodies that have been created.
476:         self.rmin = Vec3()
477:         self.rsize = -2.0 * -2.0
478:         self.root = None # A reference to the root node.
479:         self.rmin[0] = -2.0
480:         self.rmin[1] = -2.0
481:         self.rmin[2] = -2.0
482: 
483:     def create_test_data(self, nbody):
484:         '''
485:         Create the testdata used in the benchmark.
486:         @param nbody the number of bodies to create
487:         '''
488:         cmr = Vec3()
489:         cmv = Vec3()
490: 
491:         rsc = 3.0 * pi / 16.0
492:         vsc = sqrt(1.0 / rsc)
493:         seed = 123
494:         rnd = Random(seed)
495:         self.bodies = [None] * nbody
496:         aux_mass = 1.0 / float(nbody)
497: 
498:         for i in xrange(nbody):
499:             p = Body()
500:             self.bodies[i] = p
501:             p.mass = aux_mass
502: 
503:             t1 = rnd.uniform(0.0, 0.999)
504:             t1 = pow(t1, (-2.0 / 3.0)) - 1.0
505:             r = 1.0 / sqrt(t1)
506: 
507:             coeff = 4.0
508:             for k in xrange(Vec3.NDIM):
509:                 r = rnd.uniform(0.0, 0.999)
510:                 p.pos[k] = coeff * r
511: 
512:             cmr += p.pos
513: 
514:             while True:
515:                 x = rnd.uniform(0.0, 1.0)
516:                 y = rnd.uniform(0.0, 0.1)
517:                 if y <= (x * x * pow(1.0 - x * x, 3.5)):
518:                     break
519:             v = sqrt(2.0) * x / pow(1 + r * r, 0.25)
520: 
521:             rad = vsc * v
522:             while True:
523:                 for k in xrange(Vec3.NDIM):
524:                     p.vel[k] = rnd.uniform(-1.0, 1.0)
525:                 rsq = p.vel.dot()
526:                 if rsq <= 1.0:
527:                     break
528:             rsc1 = rad / sqrt(rsq)
529:             p.vel *= rsc1
530:             cmv += p.vel
531: 
532:         cmr /= float(nbody)
533:         cmv /= float(nbody)
534: 
535:         for b in self.bodies:
536:             b.pos -= cmr
537:             b.vel -= cmv
538: 
539:     def step_system(self, nstep):
540:         '''
541:         Advance the N-body system one time-step.
542:         @param nstep the current time step
543:         '''
544:         # free the tree
545:         self.root = None
546: 
547:         self.make_tree(nstep)
548: 
549:         # compute the gravity for all the particles
550:         for b in reversed(self.bodies):
551:             b.hack_gravity(self.rsize, self.root)
552:         Tree.vp(self.bodies, nstep)
553: 
554:     def make_tree(self, nstep):
555:         '''
556:         Initialize the tree structure for hack force calculation.
557:         @param nsteps the current time step
558:         '''
559:         for q in reversed(self.bodies):
560:             if q.mass != 0.0:
561:                 q.expand_box(self, nstep)
562:                 xqic = self.int_coord(q.pos)
563:                 if self.root is None:
564:                     self.root = q
565:                 else:
566:                     self.root = self.root.load_tree(q, xqic, Node.IMAX >> 1, self)
567:         self.root.hack_cofm()
568: 
569:     def int_coord(self, vp):
570:         '''
571:         Compute integerized coordinates.
572:         @return the coordinates or None if rp is out of bounds
573:         '''
574:         xp = Vec3()
575: 
576:         xsc = (vp[0] - self.rmin[0]) / self.rsize
577:         if 0.0 <= xsc and xsc < 1.0:
578:             xp[0] = floor(Node.IMAX * xsc)
579:         else:
580:             return None
581: 
582:         xsc = (vp[1] - self.rmin[1]) / self.rsize
583:         if 0.0 <= xsc and xsc < 1.0:
584:             xp[1] = floor(Node.IMAX * xsc)
585:         else:
586:             return None
587: 
588:         xsc = (vp[2] - self.rmin[2]) / self.rsize
589:         if 0.0 <= xsc and xsc < 1.0:
590:             xp[2] = floor(Node.IMAX * xsc)
591:         else:
592:             return None
593: 
594:         return xp
595: 
596:     @staticmethod
597:     def vp(bodies, nstep):
598:         dacc = Vec3()
599:         dvel = Vec3()
600:         dthf = 0.5 * BH.DTIME
601: 
602:         for b in reversed(bodies):
603:             acc1 = copy(b.new_acc)
604:             if nstep > 0:
605:                 dacc.subtraction2(acc1, b.acc)
606:                 dvel.mult_scalar2(dacc, dthf)
607:                 dvel += b.vel
608:                 b.vel = copy(dvel)
609: 
610:             b.acc = copy(acc1)
611:             dvel.mult_scalar2(b.acc, dthf)
612: 
613:             vel1 = copy(b.vel)
614:             vel1 += dvel
615:             dpos = copy(vel1)
616:             dpos *= BH.DTIME
617:             dpos += b.pos
618:             b.pos = copy(dpos)
619:             vel1 += dvel
620:             b.vel = copy(vel1)
621: 
622: 
623: class BH(object):
624:     DTIME = 0.0125
625:     TSTOP = 2.0
626: 
627:     # The user specified number of bodies to create.
628:     nbody = 0
629: 
630:     # The maximum number of time steps to take in the simulation
631:     nsteps = 10
632: 
633:     # Should we prinformation messsages
634:     print_msgs = False
635: 
636:     # Should we prdetailed results
637:     print_results = False
638: 
639:     @staticmethod
640:     def main(args):
641:         BH.parse_cmd_line(args)
642: 
643:         if BH.print_msgs:
644:             print "nbody =", BH.nbody
645: 
646:         start0 = clock()
647:         root = Tree()
648:         root.create_test_data(BH.nbody)
649:         end0 = clock()
650:         if BH.print_msgs:
651:               print "Bodies created"
652: 
653:         start1 = clock()
654:         tnow = 0.0
655:         i = 0
656:         while (tnow < BH.TSTOP + 0.1 * BH.DTIME) and i < BH.nsteps:
657:             root.step_system(i)
658:             i += 1
659:             tnow += BH.DTIME
660:         end1 = clock()
661: 
662:         if BH.print_results:
663:             for j, b in enumerate(root.bodies):
664:                 print "body %d: %s" % (j, b.pos)
665: 
666:         if BH.print_msgs:
667:             print "Build Time %.3f" % (end0 - start0)
668:             print "Compute Time %.3f" % (end1 - start1)
669:             print "Total Time %.3f" % (end1 - start0)
670: ##        print "Done!"
671: 
672:     @staticmethod
673:     def parse_cmd_line(args):
674:         i = 1
675:         while i < len(args) and args[i].startswith("-"):
676:             arg = args[i]
677:             i += 1
678: 
679:             # check for options that require arguments
680:             if arg == "-b":
681:                 if i < len(args):
682:                     BH.nbody = int(args[i])
683:                     i += 1
684:                 else:
685:                     raise Exception("-l requires the number of levels")
686:             elif arg == "-s":
687:                 if i < len(args):
688:                     BH.nsteps = int(args[i])
689:                     i += 1
690:                 else:
691:                     raise Exception("-l requires the number of levels")
692:             elif arg == "-m":
693:                 BH.print_msgs = True
694:             elif arg == "-p":
695:                 BH.print_results = True
696:             elif arg == "-h":
697:                 BH.usage()
698: 
699:         if BH.nbody == 0:
700:             BH.usage()
701: 
702:     @staticmethod
703:     def usage():
704:         '''The usage routine which describes the program options.'''
705:         print >>stderr, "usage: python bh.py -b <size> [-s <steps>] [-p] [-m] [-h]"
706:         print >>stderr, "  -b the number of bodies"
707:         print >>stderr, "  -s the max. number of time steps (default=10)"
708:         print >>stderr, "  -p (print detailed results)"
709:         print >>stderr, "  -m (print information messages"
710:         print >>stderr, "  -h (self message)"
711:         raise SystemExit()
712: 
713: 
714: def run():
715:     args = ["bh.py", "-b", "500"]
716:     BH.main(args)
717:     return True
718: 
719: run()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\nA Python implementation of the _bh_ Olden benchmark.\nThe Olden benchmark implements the Barnes-Hut benchmark\nthat is decribed in:\n\nJ. Barnes and P. Hut, "A hierarchical o(N log N) force-calculation algorithm",\nNature, 324:446-449, Dec. 1986\n\nThe original code in the Olden benchmark suite is derived from the\nftp://hubble.ifa.hawaii.edu/pub/barnes/treecode\nsource distributed by Barnes.\n\nThis code comes from the third Java version.\nThis uses copy() instead of Vec3.clone(), and it\'s adapted for ShedSkin.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from time import clock' statement (line 17)
try:
    from time import clock

except:
    clock = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'time', None, module_type_store, ['clock'], [clock])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from sys import stderr, maxint, argv' statement (line 18)
try:
    from sys import stderr, maxint, argv

except:
    stderr = UndefinedType
    maxint = UndefinedType
    argv = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'sys', None, module_type_store, ['stderr', 'maxint', 'argv'], [stderr, maxint, argv])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from copy import copy' statement (line 19)
try:
    from copy import copy

except:
    copy = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'copy', None, module_type_store, ['copy'], [copy])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from math import sqrt, pi, floor' statement (line 20)
try:
    from math import sqrt, pi, floor

except:
    sqrt = UndefinedType
    pi = UndefinedType
    floor = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'math', None, module_type_store, ['sqrt', 'pi', 'floor'], [sqrt, pi, floor])

# Declaration of the 'Random' class

class Random(object, ):
    str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n    Basic uniform random generator: Minimal Standard in Park and\n    Miller (1988): "Random Number Generators: Good Ones Are Hard to\n    Find", Comm. of the ACM, 31, 1192-1201.\n    Parameters: m = 2^31-1, a=48271.\n\n    Adapted from Pascal code by Jesper Lund:\n    http:#www.gnu-pascal.de/crystal/gpc/en/mail1390.html\n    ')

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Random.__init__', ['the_seed'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['the_seed'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 39):
        # Getting the type of 'the_seed' (line 39)
        the_seed_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'the_seed')
        # Getting the type of 'self' (line 39)
        self_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'seed' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_4, 'seed', the_seed_3)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def uniform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'uniform'
        module_type_store = module_type_store.open_function_context('uniform', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Random.uniform.__dict__.__setitem__('stypy_localization', localization)
        Random.uniform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Random.uniform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Random.uniform.__dict__.__setitem__('stypy_function_name', 'Random.uniform')
        Random.uniform.__dict__.__setitem__('stypy_param_names_list', ['min', 'max'])
        Random.uniform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Random.uniform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Random.uniform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Random.uniform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Random.uniform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Random.uniform.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Random.uniform', ['min', 'max'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'uniform', localization, ['min', 'max'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'uniform(...)' code ##################

        
        # Assigning a BinOp to a Name (line 42):
        # Getting the type of 'self' (line 42)
        self_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'self')
        # Obtaining the member 'seed' of a type (line 42)
        seed_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), self_5, 'seed')
        # Getting the type of 'Random' (line 42)
        Random_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'Random')
        # Obtaining the member 'q' of a type (line 42)
        q_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), Random_7, 'q')
        # Applying the binary operator 'div' (line 42)
        result_div_9 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 12), 'div', seed_6, q_8)
        
        # Assigning a type to the variable 'k' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'k', result_div_9)
        
        # Assigning a BinOp to a Attribute (line 43):
        # Getting the type of 'Random' (line 43)
        Random_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'Random')
        # Obtaining the member 'a' of a type (line 43)
        a_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), Random_10, 'a')
        # Getting the type of 'self' (line 43)
        self_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'self')
        # Obtaining the member 'seed' of a type (line 43)
        seed_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), self_12, 'seed')
        # Getting the type of 'k' (line 43)
        k_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 44), 'k')
        # Getting the type of 'Random' (line 43)
        Random_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 48), 'Random')
        # Obtaining the member 'q' of a type (line 43)
        q_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 48), Random_15, 'q')
        # Applying the binary operator '*' (line 43)
        result_mul_17 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 44), '*', k_14, q_16)
        
        # Applying the binary operator '-' (line 43)
        result_sub_18 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 32), '-', seed_13, result_mul_17)
        
        # Applying the binary operator '*' (line 43)
        result_mul_19 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 20), '*', a_11, result_sub_18)
        
        # Getting the type of 'Random' (line 43)
        Random_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 60), 'Random')
        # Obtaining the member 'r' of a type (line 43)
        r_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 60), Random_20, 'r')
        # Getting the type of 'k' (line 43)
        k_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 71), 'k')
        # Applying the binary operator '*' (line 43)
        result_mul_23 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 60), '*', r_21, k_22)
        
        # Applying the binary operator '-' (line 43)
        result_sub_24 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 20), '-', result_mul_19, result_mul_23)
        
        # Getting the type of 'self' (line 43)
        self_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'seed' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_25, 'seed', result_sub_24)
        
        # Getting the type of 'self' (line 44)
        self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'self')
        # Obtaining the member 'seed' of a type (line 44)
        seed_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), self_26, 'seed')
        int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
        # Applying the binary operator '<' (line 44)
        result_lt_29 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '<', seed_27, int_28)
        
        # Testing if the type of an if condition is none (line 44)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 8), result_lt_29):
            pass
        else:
            
            # Testing the type of an if condition (line 44)
            if_condition_30 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_lt_29)
            # Assigning a type to the variable 'if_condition_30' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_30', if_condition_30)
            # SSA begins for if statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'self' (line 45)
            self_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'self')
            # Obtaining the member 'seed' of a type (line 45)
            seed_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), self_31, 'seed')
            # Getting the type of 'Random' (line 45)
            Random_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'Random')
            # Obtaining the member 'm' of a type (line 45)
            m_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 25), Random_33, 'm')
            # Applying the binary operator '+=' (line 45)
            result_iadd_35 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 12), '+=', seed_32, m_34)
            # Getting the type of 'self' (line 45)
            self_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'self')
            # Setting the type of the member 'seed' of a type (line 45)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), self_36, 'seed', result_iadd_35)
            
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 46):
        
        # Call to float(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'self', False)
        # Obtaining the member 'seed' of a type (line 46)
        seed_39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), self_38, 'seed')
        # Processing the call keyword arguments (line 46)
        kwargs_40 = {}
        # Getting the type of 'float' (line 46)
        float_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'float', False)
        # Calling float(args, kwargs) (line 46)
        float_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), float_37, *[seed_39], **kwargs_40)
        
        # Getting the type of 'Random' (line 46)
        Random_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'Random')
        # Obtaining the member 'm' of a type (line 46)
        m_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 31), Random_42, 'm')
        # Applying the binary operator 'div' (line 46)
        result_div_44 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 12), 'div', float_call_result_41, m_43)
        
        # Assigning a type to the variable 'r' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'r', result_div_44)
        # Getting the type of 'r' (line 47)
        r_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'r')
        # Getting the type of 'max' (line 47)
        max_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'max')
        # Getting the type of 'min' (line 47)
        min_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'min')
        # Applying the binary operator '-' (line 47)
        result_sub_48 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 20), '-', max_46, min_47)
        
        # Applying the binary operator '*' (line 47)
        result_mul_49 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 15), '*', r_45, result_sub_48)
        
        # Getting the type of 'min' (line 47)
        min_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'min')
        # Applying the binary operator '+' (line 47)
        result_add_51 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 15), '+', result_mul_49, min_50)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', result_add_51)
        
        # ################# End of 'uniform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'uniform' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'uniform'
        return stypy_return_type_52


# Assigning a type to the variable 'Random' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'Random', Random)

# Assigning a List to a Name (line 32):

# Obtaining an instance of the builtin type 'list' (line 32)
list_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
str_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 17), 'str', 'seed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 16), list_53, str_54)

# Getting the type of 'Random'
Random_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_55, '__slots__', list_53)

# Assigning a Name to a Name (line 33):
# Getting the type of 'maxint' (line 33)
maxint_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'maxint')
# Getting the type of 'Random'
Random_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Setting the type of the member 'm' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_57, 'm', maxint_56)

# Assigning a Num to a Name (line 34):
int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'int')
# Getting the type of 'Random'
Random_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Setting the type of the member 'a' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_59, 'a', int_58)

# Assigning a BinOp to a Name (line 35):
# Getting the type of 'Random'
Random_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Obtaining the member 'm' of a type
m_61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_60, 'm')
# Getting the type of 'Random'
Random_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Obtaining the member 'a' of a type
a_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_62, 'a')
# Applying the binary operator 'div' (line 35)
result_div_64 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 8), 'div', m_61, a_63)

# Getting the type of 'Random'
Random_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Setting the type of the member 'q' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_65, 'q', result_div_64)

# Assigning a BinOp to a Name (line 36):
# Getting the type of 'Random'
Random_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Obtaining the member 'm' of a type
m_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_66, 'm')
# Getting the type of 'Random'
Random_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Obtaining the member 'a' of a type
a_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_68, 'a')
# Applying the binary operator '%' (line 36)
result_mod_70 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 8), '%', m_67, a_69)

# Getting the type of 'Random'
Random_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Random')
# Setting the type of the member 'r' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Random_71, 'r', result_mod_70)
# Declaration of the 'Vec3' class

class Vec3(object, ):
    str_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    A class representing a three dimensional vector that implements\n    several math operations.  To improve speed we implement the\n    vector as an array of doubles rather than use the exising\n    code in the java.util.class.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.__init__', [], None, None, defaults, varargs, kwargs)

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

        str_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'str', 'Construct an empty 3 dimensional vector for use in Barnes-Hut algorithm.')
        
        # Assigning a Num to a Attribute (line 63):
        float_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'float')
        # Getting the type of 'self' (line 63)
        self_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_75, 'd0', float_74)
        
        # Assigning a Num to a Attribute (line 64):
        float_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'float')
        # Getting the type of 'self' (line 64)
        self_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_77, 'd1', float_76)
        
        # Assigning a Num to a Attribute (line 65):
        float_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 18), 'float')
        # Getting the type of 'self' (line 65)
        self_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_79, 'd2', float_78)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        Vec3.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.__getitem__.__dict__.__setitem__('stypy_function_name', 'Vec3.__getitem__')
        Vec3.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['i'])
        Vec3.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.__getitem__', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        str_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', "\n        Return the value at the i'th index of the vector.\n        @param i the vector index\n        @return the value at the i'th index of the vector.\n        ")
        
        # Getting the type of 'i' (line 73)
        i_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'i')
        int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'int')
        # Applying the binary operator '==' (line 73)
        result_eq_83 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '==', i_81, int_82)
        
        # Testing if the type of an if condition is none (line 73)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_83):
            
            # Getting the type of 'i' (line 75)
            i_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'i')
            int_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'int')
            # Applying the binary operator '==' (line 75)
            result_eq_89 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 13), '==', i_87, int_88)
            
            # Testing if the type of an if condition is none (line 75)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 13), result_eq_89):
                # Getting the type of 'self' (line 78)
                self_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'self')
                # Obtaining the member 'd2' of a type (line 78)
                d2_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), self_93, 'd2')
                # Assigning a type to the variable 'stypy_return_type' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', d2_94)
            else:
                
                # Testing the type of an if condition (line 75)
                if_condition_90 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 13), result_eq_89)
                # Assigning a type to the variable 'if_condition_90' (line 75)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'if_condition_90', if_condition_90)
                # SSA begins for if statement (line 75)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 76)
                self_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'self')
                # Obtaining the member 'd1' of a type (line 76)
                d1_92 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 19), self_91, 'd1')
                # Assigning a type to the variable 'stypy_return_type' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'stypy_return_type', d1_92)
                # SSA branch for the else part of an if statement (line 75)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'self' (line 78)
                self_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'self')
                # Obtaining the member 'd2' of a type (line 78)
                d2_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), self_93, 'd2')
                # Assigning a type to the variable 'stypy_return_type' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', d2_94)
                # SSA join for if statement (line 75)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 73)
            if_condition_84 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_83)
            # Assigning a type to the variable 'if_condition_84' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_84', if_condition_84)
            # SSA begins for if statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 74)
            self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'self')
            # Obtaining the member 'd0' of a type (line 74)
            d0_86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), self_85, 'd0')
            # Assigning a type to the variable 'stypy_return_type' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', d0_86)
            # SSA branch for the else part of an if statement (line 73)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'i' (line 75)
            i_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'i')
            int_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'int')
            # Applying the binary operator '==' (line 75)
            result_eq_89 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 13), '==', i_87, int_88)
            
            # Testing if the type of an if condition is none (line 75)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 13), result_eq_89):
                # Getting the type of 'self' (line 78)
                self_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'self')
                # Obtaining the member 'd2' of a type (line 78)
                d2_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), self_93, 'd2')
                # Assigning a type to the variable 'stypy_return_type' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', d2_94)
            else:
                
                # Testing the type of an if condition (line 75)
                if_condition_90 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 13), result_eq_89)
                # Assigning a type to the variable 'if_condition_90' (line 75)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'if_condition_90', if_condition_90)
                # SSA begins for if statement (line 75)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 76)
                self_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'self')
                # Obtaining the member 'd1' of a type (line 76)
                d1_92 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 19), self_91, 'd1')
                # Assigning a type to the variable 'stypy_return_type' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'stypy_return_type', d1_92)
                # SSA branch for the else part of an if statement (line 75)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'self' (line 78)
                self_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'self')
                # Obtaining the member 'd2' of a type (line 78)
                d2_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), self_93, 'd2')
                # Assigning a type to the variable 'stypy_return_type' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', d2_94)
                # SSA join for if statement (line 75)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_95


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        Vec3.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.__setitem__.__dict__.__setitem__('stypy_function_name', 'Vec3.__setitem__')
        Vec3.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['i', 'v'])
        Vec3.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.__setitem__', ['i', 'v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['i', 'v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        str_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', "\n        Set the value of the i'th index of the vector.\n        @param i the vector index\n        @param v the value to store\n        ")
        
        # Getting the type of 'i' (line 86)
        i_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'i')
        int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'int')
        # Applying the binary operator '==' (line 86)
        result_eq_99 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), '==', i_97, int_98)
        
        # Testing if the type of an if condition is none (line 86)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_99):
            
            # Getting the type of 'i' (line 88)
            i_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'i')
            int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'int')
            # Applying the binary operator '==' (line 88)
            result_eq_105 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '==', i_103, int_104)
            
            # Testing if the type of an if condition is none (line 88)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 13), result_eq_105):
                
                # Assigning a Name to a Attribute (line 91):
                # Getting the type of 'v' (line 91)
                v_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'v')
                # Getting the type of 'self' (line 91)
                self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'self')
                # Setting the type of the member 'd2' of a type (line 91)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), self_110, 'd2', v_109)
            else:
                
                # Testing the type of an if condition (line 88)
                if_condition_106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 13), result_eq_105)
                # Assigning a type to the variable 'if_condition_106' (line 88)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'if_condition_106', if_condition_106)
                # SSA begins for if statement (line 88)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 89):
                # Getting the type of 'v' (line 89)
                v_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'v')
                # Getting the type of 'self' (line 89)
                self_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self')
                # Setting the type of the member 'd1' of a type (line 89)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_108, 'd1', v_107)
                # SSA branch for the else part of an if statement (line 88)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Attribute (line 91):
                # Getting the type of 'v' (line 91)
                v_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'v')
                # Getting the type of 'self' (line 91)
                self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'self')
                # Setting the type of the member 'd2' of a type (line 91)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), self_110, 'd2', v_109)
                # SSA join for if statement (line 88)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 86)
            if_condition_100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_99)
            # Assigning a type to the variable 'if_condition_100' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_100', if_condition_100)
            # SSA begins for if statement (line 86)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 87):
            # Getting the type of 'v' (line 87)
            v_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'v')
            # Getting the type of 'self' (line 87)
            self_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'self')
            # Setting the type of the member 'd0' of a type (line 87)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), self_102, 'd0', v_101)
            # SSA branch for the else part of an if statement (line 86)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'i' (line 88)
            i_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'i')
            int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'int')
            # Applying the binary operator '==' (line 88)
            result_eq_105 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '==', i_103, int_104)
            
            # Testing if the type of an if condition is none (line 88)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 13), result_eq_105):
                
                # Assigning a Name to a Attribute (line 91):
                # Getting the type of 'v' (line 91)
                v_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'v')
                # Getting the type of 'self' (line 91)
                self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'self')
                # Setting the type of the member 'd2' of a type (line 91)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), self_110, 'd2', v_109)
            else:
                
                # Testing the type of an if condition (line 88)
                if_condition_106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 13), result_eq_105)
                # Assigning a type to the variable 'if_condition_106' (line 88)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'if_condition_106', if_condition_106)
                # SSA begins for if statement (line 88)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 89):
                # Getting the type of 'v' (line 89)
                v_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'v')
                # Getting the type of 'self' (line 89)
                self_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self')
                # Setting the type of the member 'd1' of a type (line 89)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_108, 'd1', v_107)
                # SSA branch for the else part of an if statement (line 88)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Attribute (line 91):
                # Getting the type of 'v' (line 91)
                v_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'v')
                # Getting the type of 'self' (line 91)
                self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'self')
                # Setting the type of the member 'd2' of a type (line 91)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), self_110, 'd2', v_109)
                # SSA join for if statement (line 88)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 86)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_111


    @norecursion
    def __iadd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iadd__'
        module_type_store = module_type_store.open_function_context('__iadd__', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.__iadd__.__dict__.__setitem__('stypy_localization', localization)
        Vec3.__iadd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.__iadd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.__iadd__.__dict__.__setitem__('stypy_function_name', 'Vec3.__iadd__')
        Vec3.__iadd__.__dict__.__setitem__('stypy_param_names_list', ['u'])
        Vec3.__iadd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.__iadd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.__iadd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.__iadd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.__iadd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.__iadd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.__iadd__', ['u'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iadd__', localization, ['u'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iadd__(...)' code ##################

        str_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', '\n        Add two vectors and the result is placed in self vector.\n        @param u the other operand of the addition\n        ')
        
        # Getting the type of 'self' (line 98)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Obtaining the member 'd0' of a type (line 98)
        d0_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_113, 'd0')
        # Getting the type of 'u' (line 98)
        u_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 19), 'u')
        # Obtaining the member 'd0' of a type (line 98)
        d0_116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 19), u_115, 'd0')
        # Applying the binary operator '+=' (line 98)
        result_iadd_117 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 8), '+=', d0_114, d0_116)
        # Getting the type of 'self' (line 98)
        self_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_118, 'd0', result_iadd_117)
        
        
        # Getting the type of 'self' (line 99)
        self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Obtaining the member 'd1' of a type (line 99)
        d1_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_119, 'd1')
        # Getting the type of 'u' (line 99)
        u_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'u')
        # Obtaining the member 'd1' of a type (line 99)
        d1_122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), u_121, 'd1')
        # Applying the binary operator '+=' (line 99)
        result_iadd_123 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 8), '+=', d1_120, d1_122)
        # Getting the type of 'self' (line 99)
        self_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_124, 'd1', result_iadd_123)
        
        
        # Getting the type of 'self' (line 100)
        self_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Obtaining the member 'd2' of a type (line 100)
        d2_126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_125, 'd2')
        # Getting the type of 'u' (line 100)
        u_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'u')
        # Obtaining the member 'd2' of a type (line 100)
        d2_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 19), u_127, 'd2')
        # Applying the binary operator '+=' (line 100)
        result_iadd_129 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 8), '+=', d2_126, d2_128)
        # Getting the type of 'self' (line 100)
        self_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_130, 'd2', result_iadd_129)
        
        # Getting the type of 'self' (line 101)
        self_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', self_131)
        
        # ################# End of '__iadd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iadd__' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iadd__'
        return stypy_return_type_132


    @norecursion
    def __isub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__isub__'
        module_type_store = module_type_store.open_function_context('__isub__', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.__isub__.__dict__.__setitem__('stypy_localization', localization)
        Vec3.__isub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.__isub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.__isub__.__dict__.__setitem__('stypy_function_name', 'Vec3.__isub__')
        Vec3.__isub__.__dict__.__setitem__('stypy_param_names_list', ['u'])
        Vec3.__isub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.__isub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.__isub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.__isub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.__isub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.__isub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.__isub__', ['u'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__isub__', localization, ['u'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__isub__(...)' code ##################

        str_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '\n        Subtract two vectors and the result is placed in self vector.\n        This vector contain the first operand.\n        @param u the other operand of the subtraction.\n        ')
        
        # Getting the type of 'self' (line 109)
        self_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Obtaining the member 'd0' of a type (line 109)
        d0_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_134, 'd0')
        # Getting the type of 'u' (line 109)
        u_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'u')
        # Obtaining the member 'd0' of a type (line 109)
        d0_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), u_136, 'd0')
        # Applying the binary operator '-=' (line 109)
        result_isub_138 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 8), '-=', d0_135, d0_137)
        # Getting the type of 'self' (line 109)
        self_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_139, 'd0', result_isub_138)
        
        
        # Getting the type of 'self' (line 110)
        self_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Obtaining the member 'd1' of a type (line 110)
        d1_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_140, 'd1')
        # Getting the type of 'u' (line 110)
        u_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'u')
        # Obtaining the member 'd1' of a type (line 110)
        d1_143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 19), u_142, 'd1')
        # Applying the binary operator '-=' (line 110)
        result_isub_144 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 8), '-=', d1_141, d1_143)
        # Getting the type of 'self' (line 110)
        self_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_145, 'd1', result_isub_144)
        
        
        # Getting the type of 'self' (line 111)
        self_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Obtaining the member 'd2' of a type (line 111)
        d2_147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_146, 'd2')
        # Getting the type of 'u' (line 111)
        u_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'u')
        # Obtaining the member 'd2' of a type (line 111)
        d2_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), u_148, 'd2')
        # Applying the binary operator '-=' (line 111)
        result_isub_150 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 8), '-=', d2_147, d2_149)
        # Getting the type of 'self' (line 111)
        self_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_151, 'd2', result_isub_150)
        
        # Getting the type of 'self' (line 112)
        self_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'stypy_return_type', self_152)
        
        # ################# End of '__isub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__isub__' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__isub__'
        return stypy_return_type_153


    @norecursion
    def __imul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imul__'
        module_type_store = module_type_store.open_function_context('__imul__', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.__imul__.__dict__.__setitem__('stypy_localization', localization)
        Vec3.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.__imul__.__dict__.__setitem__('stypy_function_name', 'Vec3.__imul__')
        Vec3.__imul__.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Vec3.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.__imul__', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__imul__', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__imul__(...)' code ##################

        str_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', '\n        Multiply the vector times a scalar.\n        @param s the scalar value\n        ')
        
        # Getting the type of 'self' (line 119)
        self_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Obtaining the member 'd0' of a type (line 119)
        d0_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_155, 'd0')
        # Getting the type of 's' (line 119)
        s_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 's')
        # Applying the binary operator '*=' (line 119)
        result_imul_158 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 8), '*=', d0_156, s_157)
        # Getting the type of 'self' (line 119)
        self_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_159, 'd0', result_imul_158)
        
        
        # Getting the type of 'self' (line 120)
        self_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Obtaining the member 'd1' of a type (line 120)
        d1_161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_160, 'd1')
        # Getting the type of 's' (line 120)
        s_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 's')
        # Applying the binary operator '*=' (line 120)
        result_imul_163 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 8), '*=', d1_161, s_162)
        # Getting the type of 'self' (line 120)
        self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_164, 'd1', result_imul_163)
        
        
        # Getting the type of 'self' (line 121)
        self_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Obtaining the member 'd2' of a type (line 121)
        d2_166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_165, 'd2')
        # Getting the type of 's' (line 121)
        s_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 's')
        # Applying the binary operator '*=' (line 121)
        result_imul_168 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 8), '*=', d2_166, s_167)
        # Getting the type of 'self' (line 121)
        self_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_169, 'd2', result_imul_168)
        
        # Getting the type of 'self' (line 122)
        self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', self_170)
        
        # ################# End of '__imul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imul__' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imul__'
        return stypy_return_type_171


    @norecursion
    def __idiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__idiv__'
        module_type_store = module_type_store.open_function_context('__idiv__', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.__idiv__.__dict__.__setitem__('stypy_localization', localization)
        Vec3.__idiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.__idiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.__idiv__.__dict__.__setitem__('stypy_function_name', 'Vec3.__idiv__')
        Vec3.__idiv__.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Vec3.__idiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.__idiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.__idiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.__idiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.__idiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.__idiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.__idiv__', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__idiv__', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__idiv__(...)' code ##################

        str_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', '\n        Divide each element of the vector by a scalar value.\n        @param s the scalar value.\n        ')
        
        # Getting the type of 'self' (line 129)
        self_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Obtaining the member 'd0' of a type (line 129)
        d0_174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_173, 'd0')
        # Getting the type of 's' (line 129)
        s_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 's')
        # Applying the binary operator 'div=' (line 129)
        result_div_176 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 8), 'div=', d0_174, s_175)
        # Getting the type of 'self' (line 129)
        self_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_177, 'd0', result_div_176)
        
        
        # Getting the type of 'self' (line 130)
        self_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Obtaining the member 'd1' of a type (line 130)
        d1_179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_178, 'd1')
        # Getting the type of 's' (line 130)
        s_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 's')
        # Applying the binary operator 'div=' (line 130)
        result_div_181 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 8), 'div=', d1_179, s_180)
        # Getting the type of 'self' (line 130)
        self_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_182, 'd1', result_div_181)
        
        
        # Getting the type of 'self' (line 131)
        self_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self')
        # Obtaining the member 'd2' of a type (line 131)
        d2_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_183, 'd2')
        # Getting the type of 's' (line 131)
        s_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 's')
        # Applying the binary operator 'div=' (line 131)
        result_div_186 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 8), 'div=', d2_184, s_185)
        # Getting the type of 'self' (line 131)
        self_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_187, 'd2', result_div_186)
        
        # Getting the type of 'self' (line 132)
        self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', self_188)
        
        # ################# End of '__idiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__idiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__idiv__'
        return stypy_return_type_189


    @norecursion
    def add_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_scalar'
        module_type_store = module_type_store.open_function_context('add_scalar', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.add_scalar.__dict__.__setitem__('stypy_localization', localization)
        Vec3.add_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.add_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.add_scalar.__dict__.__setitem__('stypy_function_name', 'Vec3.add_scalar')
        Vec3.add_scalar.__dict__.__setitem__('stypy_param_names_list', ['u', 's'])
        Vec3.add_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.add_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.add_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.add_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.add_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.add_scalar.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.add_scalar', ['u', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_scalar', localization, ['u', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_scalar(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 135):
        # Getting the type of 'u' (line 135)
        u_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'u')
        # Obtaining the member 'd0' of a type (line 135)
        d0_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 18), u_190, 'd0')
        # Getting the type of 's' (line 135)
        s_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 's')
        # Applying the binary operator '+' (line 135)
        result_add_193 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 18), '+', d0_191, s_192)
        
        # Getting the type of 'self' (line 135)
        self_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_194, 'd0', result_add_193)
        
        # Assigning a BinOp to a Attribute (line 136):
        # Getting the type of 'u' (line 136)
        u_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'u')
        # Obtaining the member 'd1' of a type (line 136)
        d1_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 18), u_195, 'd1')
        # Getting the type of 's' (line 136)
        s_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 's')
        # Applying the binary operator '+' (line 136)
        result_add_198 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 18), '+', d1_196, s_197)
        
        # Getting the type of 'self' (line 136)
        self_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 136)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_199, 'd1', result_add_198)
        
        # Assigning a BinOp to a Attribute (line 137):
        # Getting the type of 'u' (line 137)
        u_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 18), 'u')
        # Obtaining the member 'd2' of a type (line 137)
        d2_201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 18), u_200, 'd2')
        # Getting the type of 's' (line 137)
        s_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 's')
        # Applying the binary operator '+' (line 137)
        result_add_203 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 18), '+', d2_201, s_202)
        
        # Getting the type of 'self' (line 137)
        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_204, 'd2', result_add_203)
        
        # ################# End of 'add_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_scalar'
        return stypy_return_type_205


    @norecursion
    def subtraction2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'subtraction2'
        module_type_store = module_type_store.open_function_context('subtraction2', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.subtraction2.__dict__.__setitem__('stypy_localization', localization)
        Vec3.subtraction2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.subtraction2.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.subtraction2.__dict__.__setitem__('stypy_function_name', 'Vec3.subtraction2')
        Vec3.subtraction2.__dict__.__setitem__('stypy_param_names_list', ['u', 'v'])
        Vec3.subtraction2.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.subtraction2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.subtraction2.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.subtraction2.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.subtraction2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.subtraction2.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.subtraction2', ['u', 'v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'subtraction2', localization, ['u', 'v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'subtraction2(...)' code ##################

        str_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', '\n        Subtract two vectors and the result is placed in self vector.\n        @param u the first operand of the subtraction.\n        @param v the second opernd of the subtraction\n        ')
        
        # Assigning a BinOp to a Attribute (line 145):
        # Getting the type of 'u' (line 145)
        u_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'u')
        # Obtaining the member 'd0' of a type (line 145)
        d0_208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), u_207, 'd0')
        # Getting the type of 'v' (line 145)
        v_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'v')
        # Obtaining the member 'd0' of a type (line 145)
        d0_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), v_209, 'd0')
        # Applying the binary operator '-' (line 145)
        result_sub_211 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 18), '-', d0_208, d0_210)
        
        # Getting the type of 'self' (line 145)
        self_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_212, 'd0', result_sub_211)
        
        # Assigning a BinOp to a Attribute (line 146):
        # Getting the type of 'u' (line 146)
        u_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'u')
        # Obtaining the member 'd1' of a type (line 146)
        d1_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 18), u_213, 'd1')
        # Getting the type of 'v' (line 146)
        v_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'v')
        # Obtaining the member 'd1' of a type (line 146)
        d1_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 25), v_215, 'd1')
        # Applying the binary operator '-' (line 146)
        result_sub_217 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 18), '-', d1_214, d1_216)
        
        # Getting the type of 'self' (line 146)
        self_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_218, 'd1', result_sub_217)
        
        # Assigning a BinOp to a Attribute (line 147):
        # Getting the type of 'u' (line 147)
        u_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'u')
        # Obtaining the member 'd2' of a type (line 147)
        d2_220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 18), u_219, 'd2')
        # Getting the type of 'v' (line 147)
        v_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'v')
        # Obtaining the member 'd2' of a type (line 147)
        d2_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), v_221, 'd2')
        # Applying the binary operator '-' (line 147)
        result_sub_223 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 18), '-', d2_220, d2_222)
        
        # Getting the type of 'self' (line 147)
        self_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_224, 'd2', result_sub_223)
        
        # ################# End of 'subtraction2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'subtraction2' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'subtraction2'
        return stypy_return_type_225


    @norecursion
    def mult_scalar2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mult_scalar2'
        module_type_store = module_type_store.open_function_context('mult_scalar2', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_localization', localization)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_function_name', 'Vec3.mult_scalar2')
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_param_names_list', ['u', 's'])
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.mult_scalar2.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.mult_scalar2', ['u', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mult_scalar2', localization, ['u', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mult_scalar2(...)' code ##################

        str_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, (-1)), 'str', '\n        Multiply the vector times a scalar and place the result in self vector.\n        @param u the vector\n        @param s the scalar value\n        ')
        
        # Assigning a BinOp to a Attribute (line 155):
        # Getting the type of 'u' (line 155)
        u_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'u')
        # Obtaining the member 'd0' of a type (line 155)
        d0_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 18), u_227, 'd0')
        # Getting the type of 's' (line 155)
        s_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 's')
        # Applying the binary operator '*' (line 155)
        result_mul_230 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 18), '*', d0_228, s_229)
        
        # Getting the type of 'self' (line 155)
        self_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self')
        # Setting the type of the member 'd0' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_231, 'd0', result_mul_230)
        
        # Assigning a BinOp to a Attribute (line 156):
        # Getting the type of 'u' (line 156)
        u_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 18), 'u')
        # Obtaining the member 'd1' of a type (line 156)
        d1_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 18), u_232, 'd1')
        # Getting the type of 's' (line 156)
        s_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 's')
        # Applying the binary operator '*' (line 156)
        result_mul_235 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 18), '*', d1_233, s_234)
        
        # Getting the type of 'self' (line 156)
        self_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self')
        # Setting the type of the member 'd1' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_236, 'd1', result_mul_235)
        
        # Assigning a BinOp to a Attribute (line 157):
        # Getting the type of 'u' (line 157)
        u_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'u')
        # Obtaining the member 'd2' of a type (line 157)
        d2_238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 18), u_237, 'd2')
        # Getting the type of 's' (line 157)
        s_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 's')
        # Applying the binary operator '*' (line 157)
        result_mul_240 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 18), '*', d2_238, s_239)
        
        # Getting the type of 'self' (line 157)
        self_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'd2' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_241, 'd2', result_mul_240)
        
        # ################# End of 'mult_scalar2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mult_scalar2' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mult_scalar2'
        return stypy_return_type_242


    @norecursion
    def dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dot'
        module_type_store = module_type_store.open_function_context('dot', 159, 4, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.dot.__dict__.__setitem__('stypy_localization', localization)
        Vec3.dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.dot.__dict__.__setitem__('stypy_function_name', 'Vec3.dot')
        Vec3.dot.__dict__.__setitem__('stypy_param_names_list', [])
        Vec3.dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.dot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.dot.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.dot', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dot', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dot(...)' code ##################

        str_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n        Return the dot product of a vector.\n        @return the dot product of a vector.\n        ')
        # Getting the type of 'self' (line 164)
        self_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'self')
        # Obtaining the member 'd0' of a type (line 164)
        d0_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 15), self_244, 'd0')
        # Getting the type of 'self' (line 164)
        self_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'self')
        # Obtaining the member 'd0' of a type (line 164)
        d0_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), self_246, 'd0')
        # Applying the binary operator '*' (line 164)
        result_mul_248 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), '*', d0_245, d0_247)
        
        # Getting the type of 'self' (line 164)
        self_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 35), 'self')
        # Obtaining the member 'd1' of a type (line 164)
        d1_250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 35), self_249, 'd1')
        # Getting the type of 'self' (line 164)
        self_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 45), 'self')
        # Obtaining the member 'd1' of a type (line 164)
        d1_252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 45), self_251, 'd1')
        # Applying the binary operator '*' (line 164)
        result_mul_253 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 35), '*', d1_250, d1_252)
        
        # Applying the binary operator '+' (line 164)
        result_add_254 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), '+', result_mul_248, result_mul_253)
        
        # Getting the type of 'self' (line 164)
        self_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 55), 'self')
        # Obtaining the member 'd2' of a type (line 164)
        d2_256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 55), self_255, 'd2')
        # Getting the type of 'self' (line 164)
        self_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 65), 'self')
        # Obtaining the member 'd2' of a type (line 164)
        d2_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 65), self_257, 'd2')
        # Applying the binary operator '*' (line 164)
        result_mul_259 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 55), '*', d2_256, d2_258)
        
        # Applying the binary operator '+' (line 164)
        result_add_260 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 53), '+', result_add_254, result_mul_259)
        
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', result_add_260)
        
        # ################# End of 'dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dot' in the type store
        # Getting the type of 'stypy_return_type' (line 159)
        stypy_return_type_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dot'
        return stypy_return_type_261


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Vec3.stypy__repr__')
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vec3.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vec3.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'str', '%.17f %.17f %.17f ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        # Getting the type of 'self' (line 167)
        self_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'self')
        # Obtaining the member 'd0' of a type (line 167)
        d0_265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 39), self_264, 'd0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 39), tuple_263, d0_265)
        # Adding element type (line 167)
        # Getting the type of 'self' (line 167)
        self_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 48), 'self')
        # Obtaining the member 'd1' of a type (line 167)
        d1_267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 48), self_266, 'd1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 39), tuple_263, d1_267)
        # Adding element type (line 167)
        # Getting the type of 'self' (line 167)
        self_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 57), 'self')
        # Obtaining the member 'd2' of a type (line 167)
        d2_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 57), self_268, 'd2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 39), tuple_263, d2_269)
        
        # Applying the binary operator '%' (line 167)
        result_mod_270 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 15), '%', str_262, tuple_263)
        
        # Assigning a type to the variable 'stypy_return_type' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stypy_return_type', result_mod_270)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 166)
        stypy_return_type_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_271


# Assigning a type to the variable 'Vec3' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'Vec3', Vec3)

# Assigning a List to a Name (line 57):

# Obtaining an instance of the builtin type 'list' (line 57)
list_272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
str_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'str', 'd0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), list_272, str_273)
# Adding element type (line 57)
str_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'str', 'd1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), list_272, str_274)
# Adding element type (line 57)
str_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'str', 'd2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), list_272, str_275)

# Getting the type of 'Vec3'
Vec3_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Vec3')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Vec3_276, '__slots__', list_272)

# Assigning a Num to a Name (line 59):
int_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'int')
# Getting the type of 'Vec3'
Vec3_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Vec3')
# Setting the type of the member 'NDIM' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Vec3_278, 'NDIM', int_277)
# Declaration of the 'HG' class

class HG(object, ):
    str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'str', '\n    A sub class which is used to compute and save information during the\n    gravity computation phase.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HG.__init__', ['b', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['b', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, (-1)), 'str', '\n        Create a  object.\n        @param b the body object\n        @param p a vector that represents the body\n        ')
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of 'b' (line 183)
        b_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'b')
        # Getting the type of 'self' (line 183)
        self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self')
        # Setting the type of the member 'pskip' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_282, 'pskip', b_281)
        
        # Assigning a Call to a Attribute (line 186):
        
        # Call to copy(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'p' (line 186)
        p_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'p', False)
        # Processing the call keyword arguments (line 186)
        kwargs_285 = {}
        # Getting the type of 'copy' (line 186)
        copy_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'copy', False)
        # Calling copy(args, kwargs) (line 186)
        copy_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 186, 20), copy_283, *[p_284], **kwargs_285)
        
        # Getting the type of 'self' (line 186)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self')
        # Setting the type of the member 'pos0' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_287, 'pos0', copy_call_result_286)
        
        # Assigning a Num to a Attribute (line 189):
        float_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 20), 'float')
        # Getting the type of 'self' (line 189)
        self_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self')
        # Setting the type of the member 'phi0' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_289, 'phi0', float_288)
        
        # Assigning a Call to a Attribute (line 192):
        
        # Call to Vec3(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_291 = {}
        # Getting the type of 'Vec3' (line 192)
        Vec3_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 192)
        Vec3_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), Vec3_290, *[], **kwargs_291)
        
        # Getting the type of 'self' (line 192)
        self_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member 'acc0' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_293, 'acc0', Vec3_call_result_292)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'HG' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'HG', HG)

# Assigning a List to a Name (line 175):

# Obtaining an instance of the builtin type 'list' (line 175)
list_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 175)
# Adding element type (line 175)
str_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 17), 'str', 'pskip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 16), list_294, str_295)
# Adding element type (line 175)
str_296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 26), 'str', 'pos0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 16), list_294, str_296)
# Adding element type (line 175)
str_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'phi0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 16), list_294, str_297)
# Adding element type (line 175)
str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 42), 'str', 'acc0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 16), list_294, str_298)

# Getting the type of 'HG'
HG_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HG')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HG_299, '__slots__', list_294)
# Declaration of the 'Node' class

class Node(object, ):
    str_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 4), 'str', 'A class that represents the common fields of a cell or body data structure.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Node.__init__', [], None, None, defaults, varargs, kwargs)

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

        str_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'str', 'Construct an empty node')
        
        # Assigning a Num to a Attribute (line 205):
        float_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 20), 'float')
        # Getting the type of 'self' (line 205)
        self_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'mass' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_303, 'mass', float_302)
        
        # Assigning a Call to a Attribute (line 206):
        
        # Call to Vec3(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_305 = {}
        # Getting the type of 'Vec3' (line 206)
        Vec3_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 206)
        Vec3_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 206, 19), Vec3_304, *[], **kwargs_305)
        
        # Getting the type of 'self' (line 206)
        self_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_307, 'pos', Vec3_call_result_306)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def load_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_tree'
        module_type_store = module_type_store.open_function_context('load_tree', 208, 4, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Node.load_tree.__dict__.__setitem__('stypy_localization', localization)
        Node.load_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Node.load_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Node.load_tree.__dict__.__setitem__('stypy_function_name', 'Node.load_tree')
        Node.load_tree.__dict__.__setitem__('stypy_param_names_list', ['p', 'xpic', 'l', 'root'])
        Node.load_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Node.load_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Node.load_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Node.load_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Node.load_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Node.load_tree.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Node.load_tree', ['p', 'xpic', 'l', 'root'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'load_tree', localization, ['p', 'xpic', 'l', 'root'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'load_tree(...)' code ##################

        
        # Call to NotImplementedError(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_309 = {}
        # Getting the type of 'NotImplementedError' (line 209)
        NotImplementedError_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 209)
        NotImplementedError_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 209, 14), NotImplementedError_308, *[], **kwargs_309)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 8), NotImplementedError_call_result_310, 'raise parameter', BaseException)
        
        # ################# End of 'load_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_tree'
        return stypy_return_type_311


    @norecursion
    def hack_cofm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hack_cofm'
        module_type_store = module_type_store.open_function_context('hack_cofm', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Node.hack_cofm.__dict__.__setitem__('stypy_localization', localization)
        Node.hack_cofm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Node.hack_cofm.__dict__.__setitem__('stypy_type_store', module_type_store)
        Node.hack_cofm.__dict__.__setitem__('stypy_function_name', 'Node.hack_cofm')
        Node.hack_cofm.__dict__.__setitem__('stypy_param_names_list', [])
        Node.hack_cofm.__dict__.__setitem__('stypy_varargs_param_name', None)
        Node.hack_cofm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Node.hack_cofm.__dict__.__setitem__('stypy_call_defaults', defaults)
        Node.hack_cofm.__dict__.__setitem__('stypy_call_varargs', varargs)
        Node.hack_cofm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Node.hack_cofm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Node.hack_cofm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hack_cofm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hack_cofm(...)' code ##################

        
        # Call to NotImplementedError(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_313 = {}
        # Getting the type of 'NotImplementedError' (line 212)
        NotImplementedError_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 212)
        NotImplementedError_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 212, 14), NotImplementedError_312, *[], **kwargs_313)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 212, 8), NotImplementedError_call_result_314, 'raise parameter', BaseException)
        
        # ################# End of 'hack_cofm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hack_cofm' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hack_cofm'
        return stypy_return_type_315


    @norecursion
    def walk_sub_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'walk_sub_tree'
        module_type_store = module_type_store.open_function_context('walk_sub_tree', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Node.walk_sub_tree.__dict__.__setitem__('stypy_localization', localization)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_function_name', 'Node.walk_sub_tree')
        Node.walk_sub_tree.__dict__.__setitem__('stypy_param_names_list', ['dsq', 'hg'])
        Node.walk_sub_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Node.walk_sub_tree.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Node.walk_sub_tree', ['dsq', 'hg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'walk_sub_tree', localization, ['dsq', 'hg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'walk_sub_tree(...)' code ##################

        
        # Call to NotImplementedError(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_317 = {}
        # Getting the type of 'NotImplementedError' (line 215)
        NotImplementedError_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 215)
        NotImplementedError_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 215, 14), NotImplementedError_316, *[], **kwargs_317)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 8), NotImplementedError_call_result_318, 'raise parameter', BaseException)
        
        # ################# End of 'walk_sub_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'walk_sub_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'walk_sub_tree'
        return stypy_return_type_319


    @staticmethod
    @norecursion
    def old_sub_index(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'old_sub_index'
        module_type_store = module_type_store.open_function_context('old_sub_index', 217, 4, False)
        
        # Passed parameters checking function
        Node.old_sub_index.__dict__.__setitem__('stypy_localization', localization)
        Node.old_sub_index.__dict__.__setitem__('stypy_type_of_self', None)
        Node.old_sub_index.__dict__.__setitem__('stypy_type_store', module_type_store)
        Node.old_sub_index.__dict__.__setitem__('stypy_function_name', 'old_sub_index')
        Node.old_sub_index.__dict__.__setitem__('stypy_param_names_list', ['ic', 'l'])
        Node.old_sub_index.__dict__.__setitem__('stypy_varargs_param_name', None)
        Node.old_sub_index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Node.old_sub_index.__dict__.__setitem__('stypy_call_defaults', defaults)
        Node.old_sub_index.__dict__.__setitem__('stypy_call_varargs', varargs)
        Node.old_sub_index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Node.old_sub_index.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'old_sub_index', ['ic', 'l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'old_sub_index', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'old_sub_index(...)' code ##################

        
        # Assigning a Num to a Name (line 219):
        int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 12), 'int')
        # Assigning a type to the variable 'i' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'i', int_320)
        
        
        # Call to xrange(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'Vec3' (line 220)
        Vec3_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'Vec3', False)
        # Obtaining the member 'NDIM' of a type (line 220)
        NDIM_323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 24), Vec3_322, 'NDIM')
        # Processing the call keyword arguments (line 220)
        kwargs_324 = {}
        # Getting the type of 'xrange' (line 220)
        xrange_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 220)
        xrange_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 220, 17), xrange_321, *[NDIM_323], **kwargs_324)
        
        # Assigning a type to the variable 'xrange_call_result_325' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'xrange_call_result_325', xrange_call_result_325)
        # Testing if the for loop is going to be iterated (line 220)
        # Testing the type of a for loop iterable (line 220)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 220, 8), xrange_call_result_325)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 220, 8), xrange_call_result_325):
            # Getting the type of the for loop variable (line 220)
            for_loop_var_326 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 220, 8), xrange_call_result_325)
            # Assigning a type to the variable 'k' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'k', for_loop_var_326)
            # SSA begins for a for statement (line 220)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to int(...): (line 221)
            # Processing the call arguments (line 221)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 221)
            k_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'k', False)
            # Getting the type of 'ic' (line 221)
            ic_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'ic', False)
            # Obtaining the member '__getitem__' of a type (line 221)
            getitem___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), ic_329, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 221)
            subscript_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), getitem___330, k_328)
            
            # Processing the call keyword arguments (line 221)
            kwargs_332 = {}
            # Getting the type of 'int' (line 221)
            int_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'int', False)
            # Calling int(args, kwargs) (line 221)
            int_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), int_327, *[subscript_call_result_331], **kwargs_332)
            
            # Getting the type of 'l' (line 221)
            l_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'l')
            # Applying the binary operator '&' (line 221)
            result_and__335 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 16), '&', int_call_result_333, l_334)
            
            int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 35), 'int')
            # Applying the binary operator '!=' (line 221)
            result_ne_337 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 15), '!=', result_and__335, int_336)
            
            # Testing if the type of an if condition is none (line 221)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 221, 12), result_ne_337):
                pass
            else:
                
                # Testing the type of an if condition (line 221)
                if_condition_338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 12), result_ne_337)
                # Assigning a type to the variable 'if_condition_338' (line 221)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'if_condition_338', if_condition_338)
                # SSA begins for if statement (line 221)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'i' (line 222)
                i_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'i')
                # Getting the type of 'Cell' (line 222)
                Cell_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'Cell')
                # Obtaining the member 'NSUB' of a type (line 222)
                NSUB_341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 21), Cell_340, 'NSUB')
                # Getting the type of 'k' (line 222)
                k_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 35), 'k')
                int_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 39), 'int')
                # Applying the binary operator '+' (line 222)
                result_add_344 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 35), '+', k_342, int_343)
                
                # Applying the binary operator '>>' (line 222)
                result_rshift_345 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 21), '>>', NSUB_341, result_add_344)
                
                # Applying the binary operator '+=' (line 222)
                result_iadd_346 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 16), '+=', i_339, result_rshift_345)
                # Assigning a type to the variable 'i' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'i', result_iadd_346)
                
                # SSA join for if statement (line 221)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'i' (line 223)
        i_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'i')
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', i_347)
        
        # ################# End of 'old_sub_index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'old_sub_index' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'old_sub_index'
        return stypy_return_type_348


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Node.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Node.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Node.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Node.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Node.stypy__repr__')
        Node.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Node.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Node.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Node.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Node.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Node.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Node.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Node.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 15), 'str', '%f : %f')
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'self' (line 226)
        self_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'self')
        # Obtaining the member 'mass' of a type (line 226)
        mass_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 28), self_351, 'mass')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 28), tuple_350, mass_352)
        # Adding element type (line 226)
        # Getting the type of 'self' (line 226)
        self_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 39), 'self')
        # Obtaining the member 'pos' of a type (line 226)
        pos_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 39), self_353, 'pos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 28), tuple_350, pos_354)
        
        # Applying the binary operator '%' (line 226)
        result_mod_355 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 15), '%', str_349, tuple_350)
        
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', result_mod_355)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_356


    @norecursion
    def grav_sub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'grav_sub'
        module_type_store = module_type_store.open_function_context('grav_sub', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Node.grav_sub.__dict__.__setitem__('stypy_localization', localization)
        Node.grav_sub.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Node.grav_sub.__dict__.__setitem__('stypy_type_store', module_type_store)
        Node.grav_sub.__dict__.__setitem__('stypy_function_name', 'Node.grav_sub')
        Node.grav_sub.__dict__.__setitem__('stypy_param_names_list', ['hg'])
        Node.grav_sub.__dict__.__setitem__('stypy_varargs_param_name', None)
        Node.grav_sub.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Node.grav_sub.__dict__.__setitem__('stypy_call_defaults', defaults)
        Node.grav_sub.__dict__.__setitem__('stypy_call_varargs', varargs)
        Node.grav_sub.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Node.grav_sub.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Node.grav_sub', ['hg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'grav_sub', localization, ['hg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'grav_sub(...)' code ##################

        str_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'str', 'Compute a single body-body or body-cell interaction')
        
        # Assigning a Call to a Name (line 230):
        
        # Call to Vec3(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_359 = {}
        # Getting the type of 'Vec3' (line 230)
        Vec3_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 13), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 230)
        Vec3_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 230, 13), Vec3_358, *[], **kwargs_359)
        
        # Assigning a type to the variable 'dr' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'dr', Vec3_call_result_360)
        
        # Call to subtraction2(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'self', False)
        # Obtaining the member 'pos' of a type (line 231)
        pos_364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 24), self_363, 'pos')
        # Getting the type of 'hg' (line 231)
        hg_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'hg', False)
        # Obtaining the member 'pos0' of a type (line 231)
        pos0_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 34), hg_365, 'pos0')
        # Processing the call keyword arguments (line 231)
        kwargs_367 = {}
        # Getting the type of 'dr' (line 231)
        dr_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'dr', False)
        # Obtaining the member 'subtraction2' of a type (line 231)
        subtraction2_362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), dr_361, 'subtraction2')
        # Calling subtraction2(args, kwargs) (line 231)
        subtraction2_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), subtraction2_362, *[pos_364, pos0_366], **kwargs_367)
        
        
        # Assigning a BinOp to a Name (line 233):
        
        # Call to dot(...): (line 233)
        # Processing the call keyword arguments (line 233)
        kwargs_371 = {}
        # Getting the type of 'dr' (line 233)
        dr_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'dr', False)
        # Obtaining the member 'dot' of a type (line 233)
        dot_370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 15), dr_369, 'dot')
        # Calling dot(args, kwargs) (line 233)
        dot_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 233, 15), dot_370, *[], **kwargs_371)
        
        # Getting the type of 'Node' (line 233)
        Node_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'Node')
        # Obtaining the member 'EPS' of a type (line 233)
        EPS_374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 27), Node_373, 'EPS')
        # Getting the type of 'Node' (line 233)
        Node_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 38), 'Node')
        # Obtaining the member 'EPS' of a type (line 233)
        EPS_376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 38), Node_375, 'EPS')
        # Applying the binary operator '*' (line 233)
        result_mul_377 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 27), '*', EPS_374, EPS_376)
        
        # Applying the binary operator '+' (line 233)
        result_add_378 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), '+', dot_call_result_372, result_mul_377)
        
        # Assigning a type to the variable 'drsq' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'drsq', result_add_378)
        
        # Assigning a Call to a Name (line 234):
        
        # Call to sqrt(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'drsq' (line 234)
        drsq_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'drsq', False)
        # Processing the call keyword arguments (line 234)
        kwargs_381 = {}
        # Getting the type of 'sqrt' (line 234)
        sqrt_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 234)
        sqrt_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), sqrt_379, *[drsq_380], **kwargs_381)
        
        # Assigning a type to the variable 'drabs' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'drabs', sqrt_call_result_382)
        
        # Assigning a BinOp to a Name (line 236):
        # Getting the type of 'self' (line 236)
        self_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'self')
        # Obtaining the member 'mass' of a type (line 236)
        mass_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), self_383, 'mass')
        # Getting the type of 'drabs' (line 236)
        drabs_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), 'drabs')
        # Applying the binary operator 'div' (line 236)
        result_div_386 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 15), 'div', mass_384, drabs_385)
        
        # Assigning a type to the variable 'phii' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'phii', result_div_386)
        
        # Getting the type of 'hg' (line 237)
        hg_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'hg')
        # Obtaining the member 'phi0' of a type (line 237)
        phi0_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), hg_387, 'phi0')
        # Getting the type of 'phii' (line 237)
        phii_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'phii')
        # Applying the binary operator '-=' (line 237)
        result_isub_390 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 8), '-=', phi0_388, phii_389)
        # Getting the type of 'hg' (line 237)
        hg_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'hg')
        # Setting the type of the member 'phi0' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), hg_391, 'phi0', result_isub_390)
        
        
        # Assigning a BinOp to a Name (line 238):
        # Getting the type of 'phii' (line 238)
        phii_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'phii')
        # Getting the type of 'drsq' (line 238)
        drsq_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'drsq')
        # Applying the binary operator 'div' (line 238)
        result_div_394 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 15), 'div', phii_392, drsq_393)
        
        # Assigning a type to the variable 'mor3' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'mor3', result_div_394)
        
        # Getting the type of 'dr' (line 239)
        dr_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'dr')
        # Getting the type of 'mor3' (line 239)
        mor3_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'mor3')
        # Applying the binary operator '*=' (line 239)
        result_imul_397 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 8), '*=', dr_395, mor3_396)
        # Assigning a type to the variable 'dr' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'dr', result_imul_397)
        
        
        # Getting the type of 'hg' (line 240)
        hg_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'hg')
        # Obtaining the member 'acc0' of a type (line 240)
        acc0_399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), hg_398, 'acc0')
        # Getting the type of 'dr' (line 240)
        dr_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'dr')
        # Applying the binary operator '+=' (line 240)
        result_iadd_401 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 8), '+=', acc0_399, dr_400)
        # Getting the type of 'hg' (line 240)
        hg_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'hg')
        # Setting the type of the member 'acc0' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), hg_402, 'acc0', result_iadd_401)
        
        # Getting the type of 'hg' (line 241)
        hg_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'hg')
        # Assigning a type to the variable 'stypy_return_type' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'stypy_return_type', hg_403)
        
        # ################# End of 'grav_sub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'grav_sub' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'grav_sub'
        return stypy_return_type_404


# Assigning a type to the variable 'Node' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'Node', Node)

# Assigning a Num to a Name (line 198):
int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 11), 'int')
# Getting the type of 'Node'
Node_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Node')
# Setting the type of the member 'IMAX' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Node_406, 'IMAX', int_405)

# Assigning a Num to a Name (line 201):
float_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 10), 'float')
# Getting the type of 'Node'
Node_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Node')
# Setting the type of the member 'EPS' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Node_408, 'EPS', float_407)
# Declaration of the 'Body' class
# Getting the type of 'Node' (line 244)
Node_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 11), 'Node')

class Body(Node_409, ):
    str_410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 4), 'str', 'A class used to representing particles in the N-body simulation.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.__init__', [], None, None, defaults, varargs, kwargs)

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

        str_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'str', 'Create an empty body.')
        
        # Call to __init__(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'self' (line 248)
        self_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'self', False)
        # Processing the call keyword arguments (line 248)
        kwargs_415 = {}
        # Getting the type of 'Node' (line 248)
        Node_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'Node', False)
        # Obtaining the member '__init__' of a type (line 248)
        init___413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), Node_412, '__init__')
        # Calling __init__(args, kwargs) (line 248)
        init___call_result_416 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), init___413, *[self_414], **kwargs_415)
        
        
        # Assigning a Call to a Attribute (line 249):
        
        # Call to Vec3(...): (line 249)
        # Processing the call keyword arguments (line 249)
        kwargs_418 = {}
        # Getting the type of 'Vec3' (line 249)
        Vec3_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 249)
        Vec3_call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 249, 19), Vec3_417, *[], **kwargs_418)
        
        # Getting the type of 'self' (line 249)
        self_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'vel' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_420, 'vel', Vec3_call_result_419)
        
        # Assigning a Call to a Attribute (line 250):
        
        # Call to Vec3(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_422 = {}
        # Getting the type of 'Vec3' (line 250)
        Vec3_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 250)
        Vec3_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 250, 19), Vec3_421, *[], **kwargs_422)
        
        # Getting the type of 'self' (line 250)
        self_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'acc' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_424, 'acc', Vec3_call_result_423)
        
        # Assigning a Call to a Attribute (line 251):
        
        # Call to Vec3(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_426 = {}
        # Getting the type of 'Vec3' (line 251)
        Vec3_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 251)
        Vec3_call_result_427 = invoke(stypy.reporting.localization.Localization(__file__, 251, 23), Vec3_425, *[], **kwargs_426)
        
        # Getting the type of 'self' (line 251)
        self_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'new_acc' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_428, 'new_acc', Vec3_call_result_427)
        
        # Assigning a Num to a Attribute (line 252):
        float_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 19), 'float')
        # Getting the type of 'self' (line 252)
        self_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self')
        # Setting the type of the member 'phi' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_430, 'phi', float_429)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def expand_box(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'expand_box'
        module_type_store = module_type_store.open_function_context('expand_box', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.expand_box.__dict__.__setitem__('stypy_localization', localization)
        Body.expand_box.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.expand_box.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.expand_box.__dict__.__setitem__('stypy_function_name', 'Body.expand_box')
        Body.expand_box.__dict__.__setitem__('stypy_param_names_list', ['tree', 'nsteps'])
        Body.expand_box.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.expand_box.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.expand_box.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.expand_box.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.expand_box.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.expand_box.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.expand_box', ['tree', 'nsteps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'expand_box', localization, ['tree', 'nsteps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'expand_box(...)' code ##################

        str_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', '\n        Enlarge cubical "box", salvaging existing tree structure.\n        @param tree the root of the tree.\n        @param nsteps the current time step\n        ')
        
        # Assigning a Call to a Name (line 260):
        
        # Call to Vec3(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_433 = {}
        # Getting the type of 'Vec3' (line 260)
        Vec3_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 260)
        Vec3_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), Vec3_432, *[], **kwargs_433)
        
        # Assigning a type to the variable 'rmid' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'rmid', Vec3_call_result_434)
        
        # Assigning a Call to a Name (line 262):
        
        # Call to ic_test(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'tree' (line 262)
        tree_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 29), 'tree', False)
        # Processing the call keyword arguments (line 262)
        kwargs_438 = {}
        # Getting the type of 'self' (line 262)
        self_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'self', False)
        # Obtaining the member 'ic_test' of a type (line 262)
        ic_test_436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 16), self_435, 'ic_test')
        # Calling ic_test(args, kwargs) (line 262)
        ic_test_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 262, 16), ic_test_436, *[tree_437], **kwargs_438)
        
        # Assigning a type to the variable 'inbox' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'inbox', ic_test_call_result_439)
        
        
        # Getting the type of 'inbox' (line 263)
        inbox_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 18), 'inbox')
        # Applying the 'not' unary operator (line 263)
        result_not__441 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 14), 'not', inbox_440)
        
        # Assigning a type to the variable 'result_not__441' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'result_not__441', result_not__441)
        # Testing if the while is going to be iterated (line 263)
        # Testing the type of an if condition (line 263)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 8), result_not__441)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 263, 8), result_not__441):
            # SSA begins for while statement (line 263)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Attribute to a Name (line 264):
            # Getting the type of 'tree' (line 264)
            tree_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'tree')
            # Obtaining the member 'rsize' of a type (line 264)
            rsize_443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 20), tree_442, 'rsize')
            # Assigning a type to the variable 'rsize' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'rsize', rsize_443)
            
            # Call to add_scalar(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'tree' (line 265)
            tree_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 28), 'tree', False)
            # Obtaining the member 'rmin' of a type (line 265)
            rmin_447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 28), tree_446, 'rmin')
            float_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 39), 'float')
            # Getting the type of 'rsize' (line 265)
            rsize_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 45), 'rsize', False)
            # Applying the binary operator '*' (line 265)
            result_mul_450 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 39), '*', float_448, rsize_449)
            
            # Processing the call keyword arguments (line 265)
            kwargs_451 = {}
            # Getting the type of 'rmid' (line 265)
            rmid_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'rmid', False)
            # Obtaining the member 'add_scalar' of a type (line 265)
            add_scalar_445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), rmid_444, 'add_scalar')
            # Calling add_scalar(args, kwargs) (line 265)
            add_scalar_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), add_scalar_445, *[rmin_447, result_mul_450], **kwargs_451)
            
            
            
            # Call to xrange(...): (line 267)
            # Processing the call arguments (line 267)
            # Getting the type of 'Vec3' (line 267)
            Vec3_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 28), 'Vec3', False)
            # Obtaining the member 'NDIM' of a type (line 267)
            NDIM_455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 28), Vec3_454, 'NDIM')
            # Processing the call keyword arguments (line 267)
            kwargs_456 = {}
            # Getting the type of 'xrange' (line 267)
            xrange_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 267)
            xrange_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 267, 21), xrange_453, *[NDIM_455], **kwargs_456)
            
            # Assigning a type to the variable 'xrange_call_result_457' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'xrange_call_result_457', xrange_call_result_457)
            # Testing if the for loop is going to be iterated (line 267)
            # Testing the type of a for loop iterable (line 267)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 267, 12), xrange_call_result_457)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 267, 12), xrange_call_result_457):
                # Getting the type of the for loop variable (line 267)
                for_loop_var_458 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 267, 12), xrange_call_result_457)
                # Assigning a type to the variable 'k' (line 267)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'k', for_loop_var_458)
                # SSA begins for a for statement (line 267)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 268)
                k_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'k')
                # Getting the type of 'self' (line 268)
                self_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'self')
                # Obtaining the member 'pos' of a type (line 268)
                pos_461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), self_460, 'pos')
                # Obtaining the member '__getitem__' of a type (line 268)
                getitem___462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), pos_461, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 268)
                subscript_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), getitem___462, k_459)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 268)
                k_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 38), 'k')
                # Getting the type of 'rmid' (line 268)
                rmid_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 33), 'rmid')
                # Obtaining the member '__getitem__' of a type (line 268)
                getitem___466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 33), rmid_465, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 268)
                subscript_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 268, 33), getitem___466, k_464)
                
                # Applying the binary operator '<' (line 268)
                result_lt_468 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 19), '<', subscript_call_result_463, subscript_call_result_467)
                
                # Testing if the type of an if condition is none (line 268)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 268, 16), result_lt_468):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 268)
                    if_condition_469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 16), result_lt_468)
                    # Assigning a type to the variable 'if_condition_469' (line 268)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'if_condition_469', if_condition_469)
                    # SSA begins for if statement (line 268)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Name (line 269):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k' (line 269)
                    k_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 37), 'k')
                    # Getting the type of 'tree' (line 269)
                    tree_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'tree')
                    # Obtaining the member 'rmin' of a type (line 269)
                    rmin_472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 27), tree_471, 'rmin')
                    # Obtaining the member '__getitem__' of a type (line 269)
                    getitem___473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 27), rmin_472, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 269)
                    subscript_call_result_474 = invoke(stypy.reporting.localization.Localization(__file__, 269, 27), getitem___473, k_470)
                    
                    # Assigning a type to the variable 'rmin' (line 269)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'rmin', subscript_call_result_474)
                    
                    # Assigning a BinOp to a Subscript (line 270):
                    # Getting the type of 'rmin' (line 270)
                    rmin_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 35), 'rmin')
                    # Getting the type of 'rsize' (line 270)
                    rsize_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'rsize')
                    # Applying the binary operator '-' (line 270)
                    result_sub_477 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 35), '-', rmin_475, rsize_476)
                    
                    # Getting the type of 'tree' (line 270)
                    tree_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'tree')
                    # Obtaining the member 'rmin' of a type (line 270)
                    rmin_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), tree_478, 'rmin')
                    # Getting the type of 'k' (line 270)
                    k_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 30), 'k')
                    # Storing an element on a container (line 270)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 20), rmin_479, (k_480, result_sub_477))
                    # SSA join for if statement (line 268)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Attribute (line 272):
            float_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 25), 'float')
            # Getting the type of 'rsize' (line 272)
            rsize_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 31), 'rsize')
            # Applying the binary operator '*' (line 272)
            result_mul_483 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 25), '*', float_481, rsize_482)
            
            # Getting the type of 'tree' (line 272)
            tree_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'tree')
            # Setting the type of the member 'rsize' of a type (line 272)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), tree_484, 'rsize', result_mul_483)
            
            # Getting the type of 'tree' (line 273)
            tree_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'tree')
            # Obtaining the member 'root' of a type (line 273)
            root_486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 15), tree_485, 'root')
            # Getting the type of 'None' (line 273)
            None_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 32), 'None')
            # Applying the binary operator 'isnot' (line 273)
            result_is_not_488 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 15), 'isnot', root_486, None_487)
            
            # Testing if the type of an if condition is none (line 273)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 273, 12), result_is_not_488):
                pass
            else:
                
                # Testing the type of an if condition (line 273)
                if_condition_489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 12), result_is_not_488)
                # Assigning a type to the variable 'if_condition_489' (line 273)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'if_condition_489', if_condition_489)
                # SSA begins for if statement (line 273)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 274):
                
                # Call to int_coord(...): (line 274)
                # Processing the call arguments (line 274)
                # Getting the type of 'rmid' (line 274)
                rmid_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'rmid', False)
                # Processing the call keyword arguments (line 274)
                kwargs_493 = {}
                # Getting the type of 'tree' (line 274)
                tree_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 21), 'tree', False)
                # Obtaining the member 'int_coord' of a type (line 274)
                int_coord_491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 21), tree_490, 'int_coord')
                # Calling int_coord(args, kwargs) (line 274)
                int_coord_call_result_494 = invoke(stypy.reporting.localization.Localization(__file__, 274, 21), int_coord_491, *[rmid_492], **kwargs_493)
                
                # Assigning a type to the variable 'ic' (line 274)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'ic', int_coord_call_result_494)
                
                # Type idiom detected: calculating its left and rigth part (line 275)
                # Getting the type of 'ic' (line 275)
                ic_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'ic')
                # Getting the type of 'None' (line 275)
                None_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 25), 'None')
                
                (may_be_497, more_types_in_union_498) = may_be_none(ic_495, None_496)

                if may_be_497:

                    if more_types_in_union_498:
                        # Runtime conditional SSA (line 275)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Call to Exception(...): (line 276)
                    # Processing the call arguments (line 276)
                    str_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 36), 'str', 'Value is out of bounds')
                    # Processing the call keyword arguments (line 276)
                    kwargs_501 = {}
                    # Getting the type of 'Exception' (line 276)
                    Exception_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'Exception', False)
                    # Calling Exception(args, kwargs) (line 276)
                    Exception_call_result_502 = invoke(stypy.reporting.localization.Localization(__file__, 276, 26), Exception_499, *[str_500], **kwargs_501)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 276, 20), Exception_call_result_502, 'raise parameter', BaseException)

                    if more_types_in_union_498:
                        # SSA join for if statement (line 275)
                        module_type_store = module_type_store.join_ssa_context()


                
                # Getting the type of 'ic' (line 275)
                ic_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'ic')
                # Assigning a type to the variable 'ic' (line 275)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'ic', remove_type_from_union(ic_503, types.NoneType))
                
                # Assigning a Call to a Name (line 277):
                
                # Call to old_sub_index(...): (line 277)
                # Processing the call arguments (line 277)
                # Getting the type of 'ic' (line 277)
                ic_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 39), 'ic', False)
                # Getting the type of 'Node' (line 277)
                Node_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 43), 'Node', False)
                # Obtaining the member 'IMAX' of a type (line 277)
                IMAX_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 43), Node_507, 'IMAX')
                int_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 56), 'int')
                # Applying the binary operator '>>' (line 277)
                result_rshift_510 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 43), '>>', IMAX_508, int_509)
                
                # Processing the call keyword arguments (line 277)
                kwargs_511 = {}
                # Getting the type of 'Node' (line 277)
                Node_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'Node', False)
                # Obtaining the member 'old_sub_index' of a type (line 277)
                old_sub_index_505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 20), Node_504, 'old_sub_index')
                # Calling old_sub_index(args, kwargs) (line 277)
                old_sub_index_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 277, 20), old_sub_index_505, *[ic_506, result_rshift_510], **kwargs_511)
                
                # Assigning a type to the variable 'k' (line 277)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'k', old_sub_index_call_result_512)
                
                # Assigning a Call to a Name (line 278):
                
                # Call to Cell(...): (line 278)
                # Processing the call keyword arguments (line 278)
                kwargs_514 = {}
                # Getting the type of 'Cell' (line 278)
                Cell_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 23), 'Cell', False)
                # Calling Cell(args, kwargs) (line 278)
                Cell_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 278, 23), Cell_513, *[], **kwargs_514)
                
                # Assigning a type to the variable 'newt' (line 278)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'newt', Cell_call_result_515)
                
                # Assigning a Attribute to a Subscript (line 279):
                # Getting the type of 'tree' (line 279)
                tree_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'tree')
                # Obtaining the member 'root' of a type (line 279)
                root_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 31), tree_516, 'root')
                # Getting the type of 'newt' (line 279)
                newt_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'newt')
                # Obtaining the member 'subp' of a type (line 279)
                subp_519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 16), newt_518, 'subp')
                # Getting the type of 'k' (line 279)
                k_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 26), 'k')
                # Storing an element on a container (line 279)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 16), subp_519, (k_520, root_517))
                
                # Assigning a Name to a Attribute (line 280):
                # Getting the type of 'newt' (line 280)
                newt_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'newt')
                # Getting the type of 'tree' (line 280)
                tree_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'tree')
                # Setting the type of the member 'root' of a type (line 280)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), tree_522, 'root', newt_521)
                
                # Assigning a Call to a Name (line 281):
                
                # Call to ic_test(...): (line 281)
                # Processing the call arguments (line 281)
                # Getting the type of 'tree' (line 281)
                tree_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 37), 'tree', False)
                # Processing the call keyword arguments (line 281)
                kwargs_526 = {}
                # Getting the type of 'self' (line 281)
                self_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'self', False)
                # Obtaining the member 'ic_test' of a type (line 281)
                ic_test_524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), self_523, 'ic_test')
                # Calling ic_test(args, kwargs) (line 281)
                ic_test_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 281, 24), ic_test_524, *[tree_525], **kwargs_526)
                
                # Assigning a type to the variable 'inbox' (line 281)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'inbox', ic_test_call_result_527)
                # SSA join for if statement (line 273)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for while statement (line 263)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'expand_box(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'expand_box' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'expand_box'
        return stypy_return_type_528


    @norecursion
    def ic_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ic_test'
        module_type_store = module_type_store.open_function_context('ic_test', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.ic_test.__dict__.__setitem__('stypy_localization', localization)
        Body.ic_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.ic_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.ic_test.__dict__.__setitem__('stypy_function_name', 'Body.ic_test')
        Body.ic_test.__dict__.__setitem__('stypy_param_names_list', ['tree'])
        Body.ic_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.ic_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.ic_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.ic_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.ic_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.ic_test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.ic_test', ['tree'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ic_test', localization, ['tree'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ic_test(...)' code ##################

        str_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 8), 'str', "Check the bounds of the body and return True if it isn't in the correct bounds.")
        
        # Assigning a Subscript to a Name (line 285):
        
        # Obtaining the type of the subscript
        int_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'int')
        # Getting the type of 'self' (line 285)
        self_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'self')
        # Obtaining the member 'pos' of a type (line 285)
        pos_532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), self_531, 'pos')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), pos_532, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_534 = invoke(stypy.reporting.localization.Localization(__file__, 285, 15), getitem___533, int_530)
        
        # Assigning a type to the variable 'pos0' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'pos0', subscript_call_result_534)
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 24), 'int')
        # Getting the type of 'self' (line 286)
        self_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'self')
        # Obtaining the member 'pos' of a type (line 286)
        pos_537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), self_536, 'pos')
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), pos_537, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 286, 15), getitem___538, int_535)
        
        # Assigning a type to the variable 'pos1' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'pos1', subscript_call_result_539)
        
        # Assigning a Subscript to a Name (line 287):
        
        # Obtaining the type of the subscript
        int_540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 24), 'int')
        # Getting the type of 'self' (line 287)
        self_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'self')
        # Obtaining the member 'pos' of a type (line 287)
        pos_542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 15), self_541, 'pos')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 15), pos_542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 287, 15), getitem___543, int_540)
        
        # Assigning a type to the variable 'pos2' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'pos2', subscript_call_result_544)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'True' (line 290)
        True_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'True')
        # Assigning a type to the variable 'result' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'result', True_545)
        
        # Assigning a BinOp to a Name (line 292):
        # Getting the type of 'pos0' (line 292)
        pos0_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'pos0')
        
        # Obtaining the type of the subscript
        int_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 32), 'int')
        # Getting the type of 'tree' (line 292)
        tree_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 22), 'tree')
        # Obtaining the member 'rmin' of a type (line 292)
        rmin_549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 22), tree_548, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 22), rmin_549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 292, 22), getitem___550, int_547)
        
        # Applying the binary operator '-' (line 292)
        result_sub_552 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), '-', pos0_546, subscript_call_result_551)
        
        # Getting the type of 'tree' (line 292)
        tree_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 38), 'tree')
        # Obtaining the member 'rsize' of a type (line 292)
        rsize_554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 38), tree_553, 'rsize')
        # Applying the binary operator 'div' (line 292)
        result_div_555 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 14), 'div', result_sub_552, rsize_554)
        
        # Assigning a type to the variable 'xsc' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'xsc', result_div_555)
        
        
        # Evaluating a boolean operation
        
        float_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 16), 'float')
        # Getting the type of 'xsc' (line 293)
        xsc_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'xsc')
        # Applying the binary operator '<' (line 293)
        result_lt_558 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 16), '<', float_556, xsc_557)
        
        
        # Getting the type of 'xsc' (line 293)
        xsc_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'xsc')
        float_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 36), 'float')
        # Applying the binary operator '<' (line 293)
        result_lt_561 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 30), '<', xsc_559, float_560)
        
        # Applying the binary operator 'and' (line 293)
        result_and_keyword_562 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 16), 'and', result_lt_558, result_lt_561)
        
        # Applying the 'not' unary operator (line 293)
        result_not__563 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 11), 'not', result_and_keyword_562)
        
        # Testing if the type of an if condition is none (line 293)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 293, 8), result_not__563):
            pass
        else:
            
            # Testing the type of an if condition (line 293)
            if_condition_564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), result_not__563)
            # Assigning a type to the variable 'if_condition_564' (line 293)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_564', if_condition_564)
            # SSA begins for if statement (line 293)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 294):
            # Getting the type of 'False' (line 294)
            False_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'False')
            # Assigning a type to the variable 'result' (line 294)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'result', False_565)
            # SSA join for if statement (line 293)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 296):
        # Getting the type of 'pos1' (line 296)
        pos1_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'pos1')
        
        # Obtaining the type of the subscript
        int_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 32), 'int')
        # Getting the type of 'tree' (line 296)
        tree_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'tree')
        # Obtaining the member 'rmin' of a type (line 296)
        rmin_569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 22), tree_568, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 22), rmin_569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_571 = invoke(stypy.reporting.localization.Localization(__file__, 296, 22), getitem___570, int_567)
        
        # Applying the binary operator '-' (line 296)
        result_sub_572 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 15), '-', pos1_566, subscript_call_result_571)
        
        # Getting the type of 'tree' (line 296)
        tree_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'tree')
        # Obtaining the member 'rsize' of a type (line 296)
        rsize_574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 38), tree_573, 'rsize')
        # Applying the binary operator 'div' (line 296)
        result_div_575 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 14), 'div', result_sub_572, rsize_574)
        
        # Assigning a type to the variable 'xsc' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'xsc', result_div_575)
        
        
        # Evaluating a boolean operation
        
        float_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 16), 'float')
        # Getting the type of 'xsc' (line 297)
        xsc_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'xsc')
        # Applying the binary operator '<' (line 297)
        result_lt_578 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 16), '<', float_576, xsc_577)
        
        
        # Getting the type of 'xsc' (line 297)
        xsc_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'xsc')
        float_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 36), 'float')
        # Applying the binary operator '<' (line 297)
        result_lt_581 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 30), '<', xsc_579, float_580)
        
        # Applying the binary operator 'and' (line 297)
        result_and_keyword_582 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 16), 'and', result_lt_578, result_lt_581)
        
        # Applying the 'not' unary operator (line 297)
        result_not__583 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 11), 'not', result_and_keyword_582)
        
        # Testing if the type of an if condition is none (line 297)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 297, 8), result_not__583):
            pass
        else:
            
            # Testing the type of an if condition (line 297)
            if_condition_584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 8), result_not__583)
            # Assigning a type to the variable 'if_condition_584' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'if_condition_584', if_condition_584)
            # SSA begins for if statement (line 297)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 298):
            # Getting the type of 'False' (line 298)
            False_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'False')
            # Assigning a type to the variable 'result' (line 298)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'result', False_585)
            # SSA join for if statement (line 297)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 300):
        # Getting the type of 'pos2' (line 300)
        pos2_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'pos2')
        
        # Obtaining the type of the subscript
        int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 32), 'int')
        # Getting the type of 'tree' (line 300)
        tree_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 22), 'tree')
        # Obtaining the member 'rmin' of a type (line 300)
        rmin_589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 22), tree_588, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 22), rmin_589, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 300, 22), getitem___590, int_587)
        
        # Applying the binary operator '-' (line 300)
        result_sub_592 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 15), '-', pos2_586, subscript_call_result_591)
        
        # Getting the type of 'tree' (line 300)
        tree_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 38), 'tree')
        # Obtaining the member 'rsize' of a type (line 300)
        rsize_594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 38), tree_593, 'rsize')
        # Applying the binary operator 'div' (line 300)
        result_div_595 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 14), 'div', result_sub_592, rsize_594)
        
        # Assigning a type to the variable 'xsc' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'xsc', result_div_595)
        
        
        # Evaluating a boolean operation
        
        float_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 16), 'float')
        # Getting the type of 'xsc' (line 301)
        xsc_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 22), 'xsc')
        # Applying the binary operator '<' (line 301)
        result_lt_598 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), '<', float_596, xsc_597)
        
        
        # Getting the type of 'xsc' (line 301)
        xsc_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'xsc')
        float_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 36), 'float')
        # Applying the binary operator '<' (line 301)
        result_lt_601 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 30), '<', xsc_599, float_600)
        
        # Applying the binary operator 'and' (line 301)
        result_and_keyword_602 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), 'and', result_lt_598, result_lt_601)
        
        # Applying the 'not' unary operator (line 301)
        result_not__603 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 11), 'not', result_and_keyword_602)
        
        # Testing if the type of an if condition is none (line 301)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 301, 8), result_not__603):
            pass
        else:
            
            # Testing the type of an if condition (line 301)
            if_condition_604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 8), result_not__603)
            # Assigning a type to the variable 'if_condition_604' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'if_condition_604', if_condition_604)
            # SSA begins for if statement (line 301)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 302):
            # Getting the type of 'False' (line 302)
            False_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 21), 'False')
            # Assigning a type to the variable 'result' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'result', False_605)
            # SSA join for if statement (line 301)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'result' (line 304)
        result_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', result_606)
        
        # ################# End of 'ic_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ic_test' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ic_test'
        return stypy_return_type_607


    @norecursion
    def load_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_tree'
        module_type_store = module_type_store.open_function_context('load_tree', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.load_tree.__dict__.__setitem__('stypy_localization', localization)
        Body.load_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.load_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.load_tree.__dict__.__setitem__('stypy_function_name', 'Body.load_tree')
        Body.load_tree.__dict__.__setitem__('stypy_param_names_list', ['p', 'xpic', 'l', 'tree'])
        Body.load_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.load_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.load_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.load_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.load_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.load_tree.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.load_tree', ['p', 'xpic', 'l', 'tree'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'load_tree', localization, ['p', 'xpic', 'l', 'tree'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'load_tree(...)' code ##################

        str_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, (-1)), 'str', "\n        Descend and insert particle.  We're at a body so we need to\n        create a cell and attach self body to the cell.\n        @param p the body to insert\n        @param xpic\n        @param l\n        @param tree the root of the data structure\n        @return the subtree with the body inserted\n        ")
        
        # Assigning a Call to a Name (line 317):
        
        # Call to Cell(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_610 = {}
        # Getting the type of 'Cell' (line 317)
        Cell_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'Cell', False)
        # Calling Cell(args, kwargs) (line 317)
        Cell_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 317, 17), Cell_609, *[], **kwargs_610)
        
        # Assigning a type to the variable 'retval' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'retval', Cell_call_result_611)
        
        # Assigning a Call to a Name (line 318):
        
        # Call to sub_index(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'tree' (line 318)
        tree_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'tree', False)
        # Getting the type of 'l' (line 318)
        l_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 34), 'l', False)
        # Processing the call keyword arguments (line 318)
        kwargs_616 = {}
        # Getting the type of 'self' (line 318)
        self_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'self', False)
        # Obtaining the member 'sub_index' of a type (line 318)
        sub_index_613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 13), self_612, 'sub_index')
        # Calling sub_index(args, kwargs) (line 318)
        sub_index_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 318, 13), sub_index_613, *[tree_614, l_615], **kwargs_616)
        
        # Assigning a type to the variable 'si' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'si', sub_index_call_result_617)
        
        # Assigning a Name to a Subscript (line 320):
        # Getting the type of 'self' (line 320)
        self_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'self')
        # Getting the type of 'retval' (line 320)
        retval_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'retval')
        # Obtaining the member 'subp' of a type (line 320)
        subp_620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), retval_619, 'subp')
        # Getting the type of 'si' (line 320)
        si_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'si')
        # Storing an element on a container (line 320)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 8), subp_620, (si_621, self_618))
        
        # Assigning a Call to a Name (line 323):
        
        # Call to old_sub_index(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'xpic' (line 323)
        xpic_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 32), 'xpic', False)
        # Getting the type of 'l' (line 323)
        l_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 38), 'l', False)
        # Processing the call keyword arguments (line 323)
        kwargs_626 = {}
        # Getting the type of 'Node' (line 323)
        Node_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'Node', False)
        # Obtaining the member 'old_sub_index' of a type (line 323)
        old_sub_index_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 13), Node_622, 'old_sub_index')
        # Calling old_sub_index(args, kwargs) (line 323)
        old_sub_index_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 323, 13), old_sub_index_623, *[xpic_624, l_625], **kwargs_626)
        
        # Assigning a type to the variable 'si' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'si', old_sub_index_call_result_627)
        
        # Assigning a Subscript to a Name (line 324):
        
        # Obtaining the type of the subscript
        # Getting the type of 'si' (line 324)
        si_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 25), 'si')
        # Getting the type of 'retval' (line 324)
        retval_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 13), 'retval')
        # Obtaining the member 'subp' of a type (line 324)
        subp_630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 13), retval_629, 'subp')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 13), subp_630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 324, 13), getitem___631, si_628)
        
        # Assigning a type to the variable 'rt' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'rt', subscript_call_result_632)
        
        # Type idiom detected: calculating its left and rigth part (line 325)
        # Getting the type of 'rt' (line 325)
        rt_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'rt')
        # Getting the type of 'None' (line 325)
        None_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 21), 'None')
        
        (may_be_635, more_types_in_union_636) = may_not_be_none(rt_633, None_634)

        if may_be_635:

            if more_types_in_union_636:
                # Runtime conditional SSA (line 325)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Subscript (line 326):
            
            # Call to load_tree(...): (line 326)
            # Processing the call arguments (line 326)
            # Getting the type of 'p' (line 326)
            p_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 43), 'p', False)
            # Getting the type of 'xpic' (line 326)
            xpic_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 46), 'xpic', False)
            # Getting the type of 'l' (line 326)
            l_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 52), 'l', False)
            int_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 57), 'int')
            # Applying the binary operator '>>' (line 326)
            result_rshift_643 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 52), '>>', l_641, int_642)
            
            # Getting the type of 'tree' (line 326)
            tree_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 60), 'tree', False)
            # Processing the call keyword arguments (line 326)
            kwargs_645 = {}
            # Getting the type of 'rt' (line 326)
            rt_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 30), 'rt', False)
            # Obtaining the member 'load_tree' of a type (line 326)
            load_tree_638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 30), rt_637, 'load_tree')
            # Calling load_tree(args, kwargs) (line 326)
            load_tree_call_result_646 = invoke(stypy.reporting.localization.Localization(__file__, 326, 30), load_tree_638, *[p_639, xpic_640, result_rshift_643, tree_644], **kwargs_645)
            
            # Getting the type of 'retval' (line 326)
            retval_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'retval')
            # Obtaining the member 'subp' of a type (line 326)
            subp_648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), retval_647, 'subp')
            # Getting the type of 'si' (line 326)
            si_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 24), 'si')
            # Storing an element on a container (line 326)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 12), subp_648, (si_649, load_tree_call_result_646))

            if more_types_in_union_636:
                # Runtime conditional SSA for else branch (line 325)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_635) or more_types_in_union_636):
            
            # Assigning a Name to a Subscript (line 328):
            # Getting the type of 'p' (line 328)
            p_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 30), 'p')
            # Getting the type of 'retval' (line 328)
            retval_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'retval')
            # Obtaining the member 'subp' of a type (line 328)
            subp_652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), retval_651, 'subp')
            # Getting the type of 'si' (line 328)
            si_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'si')
            # Storing an element on a container (line 328)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 12), subp_652, (si_653, p_650))

            if (may_be_635 and more_types_in_union_636):
                # SSA join for if statement (line 325)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'retval' (line 329)
        retval_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'retval')
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'stypy_return_type', retval_654)
        
        # ################# End of 'load_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 306)
        stypy_return_type_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_655)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_tree'
        return stypy_return_type_655


    @norecursion
    def hack_cofm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hack_cofm'
        module_type_store = module_type_store.open_function_context('hack_cofm', 331, 4, False)
        # Assigning a type to the variable 'self' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.hack_cofm.__dict__.__setitem__('stypy_localization', localization)
        Body.hack_cofm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.hack_cofm.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.hack_cofm.__dict__.__setitem__('stypy_function_name', 'Body.hack_cofm')
        Body.hack_cofm.__dict__.__setitem__('stypy_param_names_list', [])
        Body.hack_cofm.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.hack_cofm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.hack_cofm.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.hack_cofm.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.hack_cofm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.hack_cofm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.hack_cofm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hack_cofm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hack_cofm(...)' code ##################

        str_656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, (-1)), 'str', '\n        Descend tree finding center of mass coordinates\n        @return the mass of self node\n        ')
        # Getting the type of 'self' (line 336)
        self_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'self')
        # Obtaining the member 'mass' of a type (line 336)
        mass_658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 15), self_657, 'mass')
        # Assigning a type to the variable 'stypy_return_type' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'stypy_return_type', mass_658)
        
        # ################# End of 'hack_cofm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hack_cofm' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_659)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hack_cofm'
        return stypy_return_type_659


    @norecursion
    def sub_index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sub_index'
        module_type_store = module_type_store.open_function_context('sub_index', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.sub_index.__dict__.__setitem__('stypy_localization', localization)
        Body.sub_index.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.sub_index.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.sub_index.__dict__.__setitem__('stypy_function_name', 'Body.sub_index')
        Body.sub_index.__dict__.__setitem__('stypy_param_names_list', ['tree', 'l'])
        Body.sub_index.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.sub_index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.sub_index.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.sub_index.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.sub_index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.sub_index.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.sub_index', ['tree', 'l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sub_index', localization, ['tree', 'l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sub_index(...)' code ##################

        str_660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, (-1)), 'str', '\n        Determine which subcell to select.\n        Combination of int_coord and old_sub_index.\n        @param t the root of the tree\n        ')
        
        # Assigning a Call to a Name (line 344):
        
        # Call to Vec3(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_662 = {}
        # Getting the type of 'Vec3' (line 344)
        Vec3_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 13), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 344)
        Vec3_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 344, 13), Vec3_661, *[], **kwargs_662)
        
        # Assigning a type to the variable 'xp' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'xp', Vec3_call_result_663)
        
        # Assigning a BinOp to a Name (line 346):
        
        # Obtaining the type of the subscript
        int_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 24), 'int')
        # Getting the type of 'self' (line 346)
        self_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'self')
        # Obtaining the member 'pos' of a type (line 346)
        pos_666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), self_665, 'pos')
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), pos_666, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_668 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), getitem___667, int_664)
        
        
        # Obtaining the type of the subscript
        int_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 39), 'int')
        # Getting the type of 'tree' (line 346)
        tree_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 29), 'tree')
        # Obtaining the member 'rmin' of a type (line 346)
        rmin_671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 29), tree_670, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 29), rmin_671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 346, 29), getitem___672, int_669)
        
        # Applying the binary operator '-' (line 346)
        result_sub_674 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 15), '-', subscript_call_result_668, subscript_call_result_673)
        
        # Getting the type of 'tree' (line 346)
        tree_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 45), 'tree')
        # Obtaining the member 'rsize' of a type (line 346)
        rsize_676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 45), tree_675, 'rsize')
        # Applying the binary operator 'div' (line 346)
        result_div_677 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 14), 'div', result_sub_674, rsize_676)
        
        # Assigning a type to the variable 'xsc' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'xsc', result_div_677)
        
        # Assigning a Call to a Subscript (line 347):
        
        # Call to floor(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'Node' (line 347)
        Node_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'Node', False)
        # Obtaining the member 'IMAX' of a type (line 347)
        IMAX_680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 22), Node_679, 'IMAX')
        # Getting the type of 'xsc' (line 347)
        xsc_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 34), 'xsc', False)
        # Applying the binary operator '*' (line 347)
        result_mul_682 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 22), '*', IMAX_680, xsc_681)
        
        # Processing the call keyword arguments (line 347)
        kwargs_683 = {}
        # Getting the type of 'floor' (line 347)
        floor_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'floor', False)
        # Calling floor(args, kwargs) (line 347)
        floor_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), floor_678, *[result_mul_682], **kwargs_683)
        
        # Getting the type of 'xp' (line 347)
        xp_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'xp')
        int_686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 11), 'int')
        # Storing an element on a container (line 347)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 8), xp_685, (int_686, floor_call_result_684))
        
        # Assigning a BinOp to a Name (line 349):
        
        # Obtaining the type of the subscript
        int_687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 24), 'int')
        # Getting the type of 'self' (line 349)
        self_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'self')
        # Obtaining the member 'pos' of a type (line 349)
        pos_689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), self_688, 'pos')
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), pos_689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), getitem___690, int_687)
        
        
        # Obtaining the type of the subscript
        int_692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 39), 'int')
        # Getting the type of 'tree' (line 349)
        tree_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'tree')
        # Obtaining the member 'rmin' of a type (line 349)
        rmin_694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), tree_693, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), rmin_694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 349, 29), getitem___695, int_692)
        
        # Applying the binary operator '-' (line 349)
        result_sub_697 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 15), '-', subscript_call_result_691, subscript_call_result_696)
        
        # Getting the type of 'tree' (line 349)
        tree_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 45), 'tree')
        # Obtaining the member 'rsize' of a type (line 349)
        rsize_699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 45), tree_698, 'rsize')
        # Applying the binary operator 'div' (line 349)
        result_div_700 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 14), 'div', result_sub_697, rsize_699)
        
        # Assigning a type to the variable 'xsc' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'xsc', result_div_700)
        
        # Assigning a Call to a Subscript (line 350):
        
        # Call to floor(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'Node' (line 350)
        Node_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'Node', False)
        # Obtaining the member 'IMAX' of a type (line 350)
        IMAX_703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 22), Node_702, 'IMAX')
        # Getting the type of 'xsc' (line 350)
        xsc_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 34), 'xsc', False)
        # Applying the binary operator '*' (line 350)
        result_mul_705 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 22), '*', IMAX_703, xsc_704)
        
        # Processing the call keyword arguments (line 350)
        kwargs_706 = {}
        # Getting the type of 'floor' (line 350)
        floor_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'floor', False)
        # Calling floor(args, kwargs) (line 350)
        floor_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 350, 16), floor_701, *[result_mul_705], **kwargs_706)
        
        # Getting the type of 'xp' (line 350)
        xp_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'xp')
        int_709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 11), 'int')
        # Storing an element on a container (line 350)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 8), xp_708, (int_709, floor_call_result_707))
        
        # Assigning a BinOp to a Name (line 352):
        
        # Obtaining the type of the subscript
        int_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 24), 'int')
        # Getting the type of 'self' (line 352)
        self_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'self')
        # Obtaining the member 'pos' of a type (line 352)
        pos_712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), self_711, 'pos')
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), pos_712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_714 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), getitem___713, int_710)
        
        
        # Obtaining the type of the subscript
        int_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 39), 'int')
        # Getting the type of 'tree' (line 352)
        tree_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 29), 'tree')
        # Obtaining the member 'rmin' of a type (line 352)
        rmin_717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), tree_716, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), rmin_717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 352, 29), getitem___718, int_715)
        
        # Applying the binary operator '-' (line 352)
        result_sub_720 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 15), '-', subscript_call_result_714, subscript_call_result_719)
        
        # Getting the type of 'tree' (line 352)
        tree_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 45), 'tree')
        # Obtaining the member 'rsize' of a type (line 352)
        rsize_722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 45), tree_721, 'rsize')
        # Applying the binary operator 'div' (line 352)
        result_div_723 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 14), 'div', result_sub_720, rsize_722)
        
        # Assigning a type to the variable 'xsc' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'xsc', result_div_723)
        
        # Assigning a Call to a Subscript (line 353):
        
        # Call to floor(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'Node' (line 353)
        Node_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 22), 'Node', False)
        # Obtaining the member 'IMAX' of a type (line 353)
        IMAX_726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 22), Node_725, 'IMAX')
        # Getting the type of 'xsc' (line 353)
        xsc_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 'xsc', False)
        # Applying the binary operator '*' (line 353)
        result_mul_728 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 22), '*', IMAX_726, xsc_727)
        
        # Processing the call keyword arguments (line 353)
        kwargs_729 = {}
        # Getting the type of 'floor' (line 353)
        floor_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'floor', False)
        # Calling floor(args, kwargs) (line 353)
        floor_call_result_730 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), floor_724, *[result_mul_728], **kwargs_729)
        
        # Getting the type of 'xp' (line 353)
        xp_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'xp')
        int_732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 11), 'int')
        # Storing an element on a container (line 353)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 8), xp_731, (int_732, floor_call_result_730))
        
        # Assigning a Num to a Name (line 355):
        int_733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 12), 'int')
        # Assigning a type to the variable 'i' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'i', int_733)
        
        
        # Call to xrange(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'Vec3' (line 356)
        Vec3_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'Vec3', False)
        # Obtaining the member 'NDIM' of a type (line 356)
        NDIM_736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 24), Vec3_735, 'NDIM')
        # Processing the call keyword arguments (line 356)
        kwargs_737 = {}
        # Getting the type of 'xrange' (line 356)
        xrange_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 356)
        xrange_call_result_738 = invoke(stypy.reporting.localization.Localization(__file__, 356, 17), xrange_734, *[NDIM_736], **kwargs_737)
        
        # Assigning a type to the variable 'xrange_call_result_738' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'xrange_call_result_738', xrange_call_result_738)
        # Testing if the for loop is going to be iterated (line 356)
        # Testing the type of a for loop iterable (line 356)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 356, 8), xrange_call_result_738)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 356, 8), xrange_call_result_738):
            # Getting the type of the for loop variable (line 356)
            for_loop_var_739 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 356, 8), xrange_call_result_738)
            # Assigning a type to the variable 'k' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'k', for_loop_var_739)
            # SSA begins for a for statement (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to int(...): (line 357)
            # Processing the call arguments (line 357)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 357)
            k_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'k', False)
            # Getting the type of 'xp' (line 357)
            xp_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'xp', False)
            # Obtaining the member '__getitem__' of a type (line 357)
            getitem___743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), xp_742, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 357)
            subscript_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 357, 20), getitem___743, k_741)
            
            # Processing the call keyword arguments (line 357)
            kwargs_745 = {}
            # Getting the type of 'int' (line 357)
            int_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 16), 'int', False)
            # Calling int(args, kwargs) (line 357)
            int_call_result_746 = invoke(stypy.reporting.localization.Localization(__file__, 357, 16), int_740, *[subscript_call_result_744], **kwargs_745)
            
            # Getting the type of 'l' (line 357)
            l_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 29), 'l')
            # Applying the binary operator '&' (line 357)
            result_and__748 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 16), '&', int_call_result_746, l_747)
            
            int_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 35), 'int')
            # Applying the binary operator '!=' (line 357)
            result_ne_750 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 15), '!=', result_and__748, int_749)
            
            # Testing if the type of an if condition is none (line 357)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 357, 12), result_ne_750):
                pass
            else:
                
                # Testing the type of an if condition (line 357)
                if_condition_751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 12), result_ne_750)
                # Assigning a type to the variable 'if_condition_751' (line 357)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'if_condition_751', if_condition_751)
                # SSA begins for if statement (line 357)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'i' (line 358)
                i_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'i')
                # Getting the type of 'Cell' (line 358)
                Cell_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'Cell')
                # Obtaining the member 'NSUB' of a type (line 358)
                NSUB_754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 21), Cell_753, 'NSUB')
                # Getting the type of 'k' (line 358)
                k_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), 'k')
                int_756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 39), 'int')
                # Applying the binary operator '+' (line 358)
                result_add_757 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 35), '+', k_755, int_756)
                
                # Applying the binary operator '>>' (line 358)
                result_rshift_758 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 21), '>>', NSUB_754, result_add_757)
                
                # Applying the binary operator '+=' (line 358)
                result_iadd_759 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 16), '+=', i_752, result_rshift_758)
                # Assigning a type to the variable 'i' (line 358)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'i', result_iadd_759)
                
                # SSA join for if statement (line 357)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'i' (line 359)
        i_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'i')
        # Assigning a type to the variable 'stypy_return_type' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'stypy_return_type', i_760)
        
        # ################# End of 'sub_index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sub_index' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_761)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sub_index'
        return stypy_return_type_761


    @norecursion
    def hack_gravity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hack_gravity'
        module_type_store = module_type_store.open_function_context('hack_gravity', 361, 4, False)
        # Assigning a type to the variable 'self' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.hack_gravity.__dict__.__setitem__('stypy_localization', localization)
        Body.hack_gravity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.hack_gravity.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.hack_gravity.__dict__.__setitem__('stypy_function_name', 'Body.hack_gravity')
        Body.hack_gravity.__dict__.__setitem__('stypy_param_names_list', ['rsize', 'root'])
        Body.hack_gravity.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.hack_gravity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.hack_gravity.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.hack_gravity.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.hack_gravity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.hack_gravity.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.hack_gravity', ['rsize', 'root'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hack_gravity', localization, ['rsize', 'root'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hack_gravity(...)' code ##################

        str_762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, (-1)), 'str', '\n        Evaluate gravitational field on the body.\n        The original olden version calls a routine named "walkscan",\n        but we use the same name that is in the Barnes code.\n        ')
        
        # Assigning a Call to a Name (line 367):
        
        # Call to HG(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'self' (line 367)
        self_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'self', False)
        # Getting the type of 'self' (line 367)
        self_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'self', False)
        # Obtaining the member 'pos' of a type (line 367)
        pos_766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 22), self_765, 'pos')
        # Processing the call keyword arguments (line 367)
        kwargs_767 = {}
        # Getting the type of 'HG' (line 367)
        HG_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'HG', False)
        # Calling HG(args, kwargs) (line 367)
        HG_call_result_768 = invoke(stypy.reporting.localization.Localization(__file__, 367, 13), HG_763, *[self_764, pos_766], **kwargs_767)
        
        # Assigning a type to the variable 'hg' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'hg', HG_call_result_768)
        
        # Assigning a Call to a Name (line 368):
        
        # Call to walk_sub_tree(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'rsize' (line 368)
        rsize_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 32), 'rsize', False)
        # Getting the type of 'rsize' (line 368)
        rsize_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 40), 'rsize', False)
        # Applying the binary operator '*' (line 368)
        result_mul_773 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 32), '*', rsize_771, rsize_772)
        
        # Getting the type of 'hg' (line 368)
        hg_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 47), 'hg', False)
        # Processing the call keyword arguments (line 368)
        kwargs_775 = {}
        # Getting the type of 'root' (line 368)
        root_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'root', False)
        # Obtaining the member 'walk_sub_tree' of a type (line 368)
        walk_sub_tree_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 13), root_769, 'walk_sub_tree')
        # Calling walk_sub_tree(args, kwargs) (line 368)
        walk_sub_tree_call_result_776 = invoke(stypy.reporting.localization.Localization(__file__, 368, 13), walk_sub_tree_770, *[result_mul_773, hg_774], **kwargs_775)
        
        # Assigning a type to the variable 'hg' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'hg', walk_sub_tree_call_result_776)
        
        # Assigning a Attribute to a Attribute (line 369):
        # Getting the type of 'hg' (line 369)
        hg_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'hg')
        # Obtaining the member 'phi0' of a type (line 369)
        phi0_778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 19), hg_777, 'phi0')
        # Getting the type of 'self' (line 369)
        self_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self')
        # Setting the type of the member 'phi' of a type (line 369)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_779, 'phi', phi0_778)
        
        # Assigning a Attribute to a Attribute (line 370):
        # Getting the type of 'hg' (line 370)
        hg_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'hg')
        # Obtaining the member 'acc0' of a type (line 370)
        acc0_781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 23), hg_780, 'acc0')
        # Getting the type of 'self' (line 370)
        self_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'self')
        # Setting the type of the member 'new_acc' of a type (line 370)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), self_782, 'new_acc', acc0_781)
        
        # ################# End of 'hack_gravity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hack_gravity' in the type store
        # Getting the type of 'stypy_return_type' (line 361)
        stypy_return_type_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hack_gravity'
        return stypy_return_type_783


    @norecursion
    def walk_sub_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'walk_sub_tree'
        module_type_store = module_type_store.open_function_context('walk_sub_tree', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.walk_sub_tree.__dict__.__setitem__('stypy_localization', localization)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_function_name', 'Body.walk_sub_tree')
        Body.walk_sub_tree.__dict__.__setitem__('stypy_param_names_list', ['dsq', 'hg'])
        Body.walk_sub_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.walk_sub_tree.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.walk_sub_tree', ['dsq', 'hg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'walk_sub_tree', localization, ['dsq', 'hg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'walk_sub_tree(...)' code ##################

        str_784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 8), 'str', 'Recursively walk the tree to do hackwalk calculation')
        
        # Getting the type of 'self' (line 374)
        self_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 11), 'self')
        # Getting the type of 'hg' (line 374)
        hg_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 19), 'hg')
        # Obtaining the member 'pskip' of a type (line 374)
        pskip_787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 19), hg_786, 'pskip')
        # Applying the binary operator '!=' (line 374)
        result_ne_788 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 11), '!=', self_785, pskip_787)
        
        # Testing if the type of an if condition is none (line 374)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 374, 8), result_ne_788):
            pass
        else:
            
            # Testing the type of an if condition (line 374)
            if_condition_789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 8), result_ne_788)
            # Assigning a type to the variable 'if_condition_789' (line 374)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'if_condition_789', if_condition_789)
            # SSA begins for if statement (line 374)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 375):
            
            # Call to grav_sub(...): (line 375)
            # Processing the call arguments (line 375)
            # Getting the type of 'hg' (line 375)
            hg_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 31), 'hg', False)
            # Processing the call keyword arguments (line 375)
            kwargs_793 = {}
            # Getting the type of 'self' (line 375)
            self_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 17), 'self', False)
            # Obtaining the member 'grav_sub' of a type (line 375)
            grav_sub_791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 17), self_790, 'grav_sub')
            # Calling grav_sub(args, kwargs) (line 375)
            grav_sub_call_result_794 = invoke(stypy.reporting.localization.Localization(__file__, 375, 17), grav_sub_791, *[hg_792], **kwargs_793)
            
            # Assigning a type to the variable 'hg' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'hg', grav_sub_call_result_794)
            # SSA join for if statement (line 374)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'hg' (line 376)
        hg_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'hg')
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'stypy_return_type', hg_795)
        
        # ################# End of 'walk_sub_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'walk_sub_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_796)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'walk_sub_tree'
        return stypy_return_type_796


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Body.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Body.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Body.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Body.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Body.stypy__repr__')
        Body.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Body.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Body.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Body.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Body.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Body.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Body.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, (-1)), 'str', '\n        Return a string represenation of a body.\n        @return a string represenation of a body.\n        ')
        str_798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 15), 'str', 'Body ')
        
        # Call to __repr__(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'self' (line 383)
        self_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 39), 'self', False)
        # Processing the call keyword arguments (line 383)
        kwargs_802 = {}
        # Getting the type of 'Node' (line 383)
        Node_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 25), 'Node', False)
        # Obtaining the member '__repr__' of a type (line 383)
        repr___800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 25), Node_799, '__repr__')
        # Calling __repr__(args, kwargs) (line 383)
        repr___call_result_803 = invoke(stypy.reporting.localization.Localization(__file__, 383, 25), repr___800, *[self_801], **kwargs_802)
        
        # Applying the binary operator '+' (line 383)
        result_add_804 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 15), '+', str_798, repr___call_result_803)
        
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'stypy_return_type', result_add_804)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_805


# Assigning a type to the variable 'Body' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'Body', Body)
# Declaration of the 'Cell' class
# Getting the type of 'Node' (line 386)
Node_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 11), 'Node')

class Cell(Node_806, ):
    str_807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 4), 'str', 'A class used to represent internal nodes in the tree')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'self' (line 394)
        self_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 22), 'self', False)
        # Processing the call keyword arguments (line 394)
        kwargs_811 = {}
        # Getting the type of 'Node' (line 394)
        Node_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'Node', False)
        # Obtaining the member '__init__' of a type (line 394)
        init___809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), Node_808, '__init__')
        # Calling __init__(args, kwargs) (line 394)
        init___call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), init___809, *[self_810], **kwargs_811)
        
        
        # Assigning a BinOp to a Attribute (line 395):
        
        # Obtaining an instance of the builtin type 'list' (line 395)
        list_813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 395)
        # Adding element type (line 395)
        # Getting the type of 'None' (line 395)
        None_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 20), list_813, None_814)
        
        # Getting the type of 'Cell' (line 395)
        Cell_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 29), 'Cell')
        # Obtaining the member 'NSUB' of a type (line 395)
        NSUB_816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 29), Cell_815, 'NSUB')
        # Applying the binary operator '*' (line 395)
        result_mul_817 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 20), '*', list_813, NSUB_816)
        
        # Getting the type of 'self' (line 395)
        self_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'self')
        # Setting the type of the member 'subp' of a type (line 395)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), self_818, 'subp', result_mul_817)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def load_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_tree'
        module_type_store = module_type_store.open_function_context('load_tree', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.load_tree.__dict__.__setitem__('stypy_localization', localization)
        Cell.load_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.load_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.load_tree.__dict__.__setitem__('stypy_function_name', 'Cell.load_tree')
        Cell.load_tree.__dict__.__setitem__('stypy_param_names_list', ['p', 'xpic', 'l', 'tree'])
        Cell.load_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.load_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.load_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.load_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.load_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.load_tree.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.load_tree', ['p', 'xpic', 'l', 'tree'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'load_tree', localization, ['p', 'xpic', 'l', 'tree'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'load_tree(...)' code ##################

        str_819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, (-1)), 'str', "\n        Descend and insert particle.  We're at a cell so\n        we need to move down the tree.\n        @param p the body to insert into the tree\n        @param xpic\n        @param l\n        @param tree the root of the tree\n        @return the subtree with the body inserted\n        ")
        
        # Assigning a Call to a Name (line 408):
        
        # Call to old_sub_index(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'xpic' (line 408)
        xpic_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 32), 'xpic', False)
        # Getting the type of 'l' (line 408)
        l_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 38), 'l', False)
        # Processing the call keyword arguments (line 408)
        kwargs_824 = {}
        # Getting the type of 'Node' (line 408)
        Node_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 13), 'Node', False)
        # Obtaining the member 'old_sub_index' of a type (line 408)
        old_sub_index_821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 13), Node_820, 'old_sub_index')
        # Calling old_sub_index(args, kwargs) (line 408)
        old_sub_index_call_result_825 = invoke(stypy.reporting.localization.Localization(__file__, 408, 13), old_sub_index_821, *[xpic_822, l_823], **kwargs_824)
        
        # Assigning a type to the variable 'si' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'si', old_sub_index_call_result_825)
        
        # Assigning a Subscript to a Name (line 409):
        
        # Obtaining the type of the subscript
        # Getting the type of 'si' (line 409)
        si_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 23), 'si')
        # Getting the type of 'self' (line 409)
        self_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), 'self')
        # Obtaining the member 'subp' of a type (line 409)
        subp_828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 13), self_827, 'subp')
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 13), subp_828, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 409)
        subscript_call_result_830 = invoke(stypy.reporting.localization.Localization(__file__, 409, 13), getitem___829, si_826)
        
        # Assigning a type to the variable 'rt' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'rt', subscript_call_result_830)
        
        # Type idiom detected: calculating its left and rigth part (line 410)
        # Getting the type of 'rt' (line 410)
        rt_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'rt')
        # Getting the type of 'None' (line 410)
        None_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 21), 'None')
        
        (may_be_833, more_types_in_union_834) = may_not_be_none(rt_831, None_832)

        if may_be_833:

            if more_types_in_union_834:
                # Runtime conditional SSA (line 410)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Subscript (line 411):
            
            # Call to load_tree(...): (line 411)
            # Processing the call arguments (line 411)
            # Getting the type of 'p' (line 411)
            p_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 41), 'p', False)
            # Getting the type of 'xpic' (line 411)
            xpic_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 44), 'xpic', False)
            # Getting the type of 'l' (line 411)
            l_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 50), 'l', False)
            int_840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 55), 'int')
            # Applying the binary operator '>>' (line 411)
            result_rshift_841 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 50), '>>', l_839, int_840)
            
            # Getting the type of 'tree' (line 411)
            tree_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 58), 'tree', False)
            # Processing the call keyword arguments (line 411)
            kwargs_843 = {}
            # Getting the type of 'rt' (line 411)
            rt_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'rt', False)
            # Obtaining the member 'load_tree' of a type (line 411)
            load_tree_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), rt_835, 'load_tree')
            # Calling load_tree(args, kwargs) (line 411)
            load_tree_call_result_844 = invoke(stypy.reporting.localization.Localization(__file__, 411, 28), load_tree_836, *[p_837, xpic_838, result_rshift_841, tree_842], **kwargs_843)
            
            # Getting the type of 'self' (line 411)
            self_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'self')
            # Obtaining the member 'subp' of a type (line 411)
            subp_846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 12), self_845, 'subp')
            # Getting the type of 'si' (line 411)
            si_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 22), 'si')
            # Storing an element on a container (line 411)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), subp_846, (si_847, load_tree_call_result_844))

            if more_types_in_union_834:
                # Runtime conditional SSA for else branch (line 410)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_833) or more_types_in_union_834):
            
            # Assigning a Name to a Subscript (line 413):
            # Getting the type of 'p' (line 413)
            p_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 28), 'p')
            # Getting the type of 'self' (line 413)
            self_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'self')
            # Obtaining the member 'subp' of a type (line 413)
            subp_850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), self_849, 'subp')
            # Getting the type of 'si' (line 413)
            si_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 22), 'si')
            # Storing an element on a container (line 413)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), subp_850, (si_851, p_848))

            if (may_be_833 and more_types_in_union_834):
                # SSA join for if statement (line 410)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 414)
        self_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'stypy_return_type', self_852)
        
        # ################# End of 'load_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_853)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_tree'
        return stypy_return_type_853


    @norecursion
    def hack_cofm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hack_cofm'
        module_type_store = module_type_store.open_function_context('hack_cofm', 416, 4, False)
        # Assigning a type to the variable 'self' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.hack_cofm.__dict__.__setitem__('stypy_localization', localization)
        Cell.hack_cofm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.hack_cofm.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.hack_cofm.__dict__.__setitem__('stypy_function_name', 'Cell.hack_cofm')
        Cell.hack_cofm.__dict__.__setitem__('stypy_param_names_list', [])
        Cell.hack_cofm.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.hack_cofm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.hack_cofm.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.hack_cofm.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.hack_cofm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.hack_cofm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.hack_cofm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hack_cofm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hack_cofm(...)' code ##################

        str_854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, (-1)), 'str', '\n        Descend tree finding center of mass coordinates\n        @return the mass of self node\n        ')
        
        # Assigning a Num to a Name (line 421):
        float_855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 13), 'float')
        # Assigning a type to the variable 'mq' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'mq', float_855)
        
        # Assigning a Call to a Name (line 422):
        
        # Call to Vec3(...): (line 422)
        # Processing the call keyword arguments (line 422)
        kwargs_857 = {}
        # Getting the type of 'Vec3' (line 422)
        Vec3_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 18), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 422)
        Vec3_call_result_858 = invoke(stypy.reporting.localization.Localization(__file__, 422, 18), Vec3_856, *[], **kwargs_857)
        
        # Assigning a type to the variable 'tmp_pos' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'tmp_pos', Vec3_call_result_858)
        
        # Assigning a Call to a Name (line 423):
        
        # Call to Vec3(...): (line 423)
        # Processing the call keyword arguments (line 423)
        kwargs_860 = {}
        # Getting the type of 'Vec3' (line 423)
        Vec3_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 423)
        Vec3_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 423, 15), Vec3_859, *[], **kwargs_860)
        
        # Assigning a type to the variable 'tmpv' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tmpv', Vec3_call_result_861)
        
        
        # Call to xrange(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'Cell' (line 424)
        Cell_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'Cell', False)
        # Obtaining the member 'NSUB' of a type (line 424)
        NSUB_864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 24), Cell_863, 'NSUB')
        # Processing the call keyword arguments (line 424)
        kwargs_865 = {}
        # Getting the type of 'xrange' (line 424)
        xrange_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 424)
        xrange_call_result_866 = invoke(stypy.reporting.localization.Localization(__file__, 424, 17), xrange_862, *[NSUB_864], **kwargs_865)
        
        # Assigning a type to the variable 'xrange_call_result_866' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'xrange_call_result_866', xrange_call_result_866)
        # Testing if the for loop is going to be iterated (line 424)
        # Testing the type of a for loop iterable (line 424)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 424, 8), xrange_call_result_866)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 424, 8), xrange_call_result_866):
            # Getting the type of the for loop variable (line 424)
            for_loop_var_867 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 424, 8), xrange_call_result_866)
            # Assigning a type to the variable 'i' (line 424)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'i', for_loop_var_867)
            # SSA begins for a for statement (line 424)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 425):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 425)
            i_868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'i')
            # Getting the type of 'self' (line 425)
            self_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'self')
            # Obtaining the member 'subp' of a type (line 425)
            subp_870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), self_869, 'subp')
            # Obtaining the member '__getitem__' of a type (line 425)
            getitem___871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 16), subp_870, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 425)
            subscript_call_result_872 = invoke(stypy.reporting.localization.Localization(__file__, 425, 16), getitem___871, i_868)
            
            # Assigning a type to the variable 'r' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'r', subscript_call_result_872)
            
            # Type idiom detected: calculating its left and rigth part (line 426)
            # Getting the type of 'r' (line 426)
            r_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'r')
            # Getting the type of 'None' (line 426)
            None_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 24), 'None')
            
            (may_be_875, more_types_in_union_876) = may_not_be_none(r_873, None_874)

            if may_be_875:

                if more_types_in_union_876:
                    # Runtime conditional SSA (line 426)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 427):
                
                # Call to hack_cofm(...): (line 427)
                # Processing the call keyword arguments (line 427)
                kwargs_879 = {}
                # Getting the type of 'r' (line 427)
                r_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'r', False)
                # Obtaining the member 'hack_cofm' of a type (line 427)
                hack_cofm_878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 21), r_877, 'hack_cofm')
                # Calling hack_cofm(args, kwargs) (line 427)
                hack_cofm_call_result_880 = invoke(stypy.reporting.localization.Localization(__file__, 427, 21), hack_cofm_878, *[], **kwargs_879)
                
                # Assigning a type to the variable 'mr' (line 427)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'mr', hack_cofm_call_result_880)
                
                # Assigning a BinOp to a Name (line 428):
                # Getting the type of 'mr' (line 428)
                mr_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'mr')
                # Getting the type of 'mq' (line 428)
                mq_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'mq')
                # Applying the binary operator '+' (line 428)
                result_add_883 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 21), '+', mr_881, mq_882)
                
                # Assigning a type to the variable 'mq' (line 428)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'mq', result_add_883)
                
                # Call to mult_scalar2(...): (line 429)
                # Processing the call arguments (line 429)
                # Getting the type of 'r' (line 429)
                r_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 34), 'r', False)
                # Obtaining the member 'pos' of a type (line 429)
                pos_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 34), r_886, 'pos')
                # Getting the type of 'mr' (line 429)
                mr_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 41), 'mr', False)
                # Processing the call keyword arguments (line 429)
                kwargs_889 = {}
                # Getting the type of 'tmpv' (line 429)
                tmpv_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'tmpv', False)
                # Obtaining the member 'mult_scalar2' of a type (line 429)
                mult_scalar2_885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 16), tmpv_884, 'mult_scalar2')
                # Calling mult_scalar2(args, kwargs) (line 429)
                mult_scalar2_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 429, 16), mult_scalar2_885, *[pos_887, mr_888], **kwargs_889)
                
                
                # Getting the type of 'tmp_pos' (line 430)
                tmp_pos_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tmp_pos')
                # Getting the type of 'tmpv' (line 430)
                tmpv_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 27), 'tmpv')
                # Applying the binary operator '+=' (line 430)
                result_iadd_893 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 16), '+=', tmp_pos_891, tmpv_892)
                # Assigning a type to the variable 'tmp_pos' (line 430)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'tmp_pos', result_iadd_893)
                

                if more_types_in_union_876:
                    # SSA join for if statement (line 426)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 431):
        # Getting the type of 'mq' (line 431)
        mq_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 20), 'mq')
        # Getting the type of 'self' (line 431)
        self_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self')
        # Setting the type of the member 'mass' of a type (line 431)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_895, 'mass', mq_894)
        
        # Assigning a Name to a Attribute (line 432):
        # Getting the type of 'tmp_pos' (line 432)
        tmp_pos_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), 'tmp_pos')
        # Getting the type of 'self' (line 432)
        self_897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 432)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_897, 'pos', tmp_pos_896)
        
        # Getting the type of 'self' (line 433)
        self_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self')
        # Obtaining the member 'pos' of a type (line 433)
        pos_899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_898, 'pos')
        # Getting the type of 'self' (line 433)
        self_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'self')
        # Obtaining the member 'mass' of a type (line 433)
        mass_901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 20), self_900, 'mass')
        # Applying the binary operator 'div=' (line 433)
        result_div_902 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 8), 'div=', pos_899, mass_901)
        # Getting the type of 'self' (line 433)
        self_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 433)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_903, 'pos', result_div_902)
        
        # Getting the type of 'mq' (line 434)
        mq_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'mq')
        # Assigning a type to the variable 'stypy_return_type' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'stypy_return_type', mq_904)
        
        # ################# End of 'hack_cofm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hack_cofm' in the type store
        # Getting the type of 'stypy_return_type' (line 416)
        stypy_return_type_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hack_cofm'
        return stypy_return_type_905


    @norecursion
    def walk_sub_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'walk_sub_tree'
        module_type_store = module_type_store.open_function_context('walk_sub_tree', 437, 4, False)
        # Assigning a type to the variable 'self' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_localization', localization)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_function_name', 'Cell.walk_sub_tree')
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_param_names_list', ['dsq', 'hg'])
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.walk_sub_tree.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.walk_sub_tree', ['dsq', 'hg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'walk_sub_tree', localization, ['dsq', 'hg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'walk_sub_tree(...)' code ##################

        str_906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 8), 'str', 'Recursively walk the tree to do hackwalk calculation')
        
        # Call to subdiv_p(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'dsq' (line 439)
        dsq_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 25), 'dsq', False)
        # Getting the type of 'hg' (line 439)
        hg_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 30), 'hg', False)
        # Processing the call keyword arguments (line 439)
        kwargs_911 = {}
        # Getting the type of 'self' (line 439)
        self_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'self', False)
        # Obtaining the member 'subdiv_p' of a type (line 439)
        subdiv_p_908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 11), self_907, 'subdiv_p')
        # Calling subdiv_p(args, kwargs) (line 439)
        subdiv_p_call_result_912 = invoke(stypy.reporting.localization.Localization(__file__, 439, 11), subdiv_p_908, *[dsq_909, hg_910], **kwargs_911)
        
        # Testing if the type of an if condition is none (line 439)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 439, 8), subdiv_p_call_result_912):
            
            # Assigning a Call to a Name (line 445):
            
            # Call to grav_sub(...): (line 445)
            # Processing the call arguments (line 445)
            # Getting the type of 'hg' (line 445)
            hg_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 31), 'hg', False)
            # Processing the call keyword arguments (line 445)
            kwargs_940 = {}
            # Getting the type of 'self' (line 445)
            self_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 17), 'self', False)
            # Obtaining the member 'grav_sub' of a type (line 445)
            grav_sub_938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 17), self_937, 'grav_sub')
            # Calling grav_sub(args, kwargs) (line 445)
            grav_sub_call_result_941 = invoke(stypy.reporting.localization.Localization(__file__, 445, 17), grav_sub_938, *[hg_939], **kwargs_940)
            
            # Assigning a type to the variable 'hg' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'hg', grav_sub_call_result_941)
        else:
            
            # Testing the type of an if condition (line 439)
            if_condition_913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 8), subdiv_p_call_result_912)
            # Assigning a type to the variable 'if_condition_913' (line 439)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'if_condition_913', if_condition_913)
            # SSA begins for if statement (line 439)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to xrange(...): (line 440)
            # Processing the call arguments (line 440)
            # Getting the type of 'Cell' (line 440)
            Cell_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 28), 'Cell', False)
            # Obtaining the member 'NSUB' of a type (line 440)
            NSUB_916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 28), Cell_915, 'NSUB')
            # Processing the call keyword arguments (line 440)
            kwargs_917 = {}
            # Getting the type of 'xrange' (line 440)
            xrange_914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 440)
            xrange_call_result_918 = invoke(stypy.reporting.localization.Localization(__file__, 440, 21), xrange_914, *[NSUB_916], **kwargs_917)
            
            # Assigning a type to the variable 'xrange_call_result_918' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'xrange_call_result_918', xrange_call_result_918)
            # Testing if the for loop is going to be iterated (line 440)
            # Testing the type of a for loop iterable (line 440)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 12), xrange_call_result_918)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 440, 12), xrange_call_result_918):
                # Getting the type of the for loop variable (line 440)
                for_loop_var_919 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 12), xrange_call_result_918)
                # Assigning a type to the variable 'k' (line 440)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'k', for_loop_var_919)
                # SSA begins for a for statement (line 440)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 441):
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 441)
                k_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 'k')
                # Getting the type of 'self' (line 441)
                self_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'self')
                # Obtaining the member 'subp' of a type (line 441)
                subp_922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), self_921, 'subp')
                # Obtaining the member '__getitem__' of a type (line 441)
                getitem___923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), subp_922, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 441)
                subscript_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 441, 20), getitem___923, k_920)
                
                # Assigning a type to the variable 'r' (line 441)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'r', subscript_call_result_924)
                
                # Type idiom detected: calculating its left and rigth part (line 442)
                # Getting the type of 'r' (line 442)
                r_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'r')
                # Getting the type of 'None' (line 442)
                None_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'None')
                
                (may_be_927, more_types_in_union_928) = may_not_be_none(r_925, None_926)

                if may_be_927:

                    if more_types_in_union_928:
                        # Runtime conditional SSA (line 442)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Call to a Name (line 443):
                    
                    # Call to walk_sub_tree(...): (line 443)
                    # Processing the call arguments (line 443)
                    # Getting the type of 'dsq' (line 443)
                    dsq_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 41), 'dsq', False)
                    float_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 47), 'float')
                    # Applying the binary operator 'div' (line 443)
                    result_div_933 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 41), 'div', dsq_931, float_932)
                    
                    # Getting the type of 'hg' (line 443)
                    hg_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 52), 'hg', False)
                    # Processing the call keyword arguments (line 443)
                    kwargs_935 = {}
                    # Getting the type of 'r' (line 443)
                    r_929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 25), 'r', False)
                    # Obtaining the member 'walk_sub_tree' of a type (line 443)
                    walk_sub_tree_930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 25), r_929, 'walk_sub_tree')
                    # Calling walk_sub_tree(args, kwargs) (line 443)
                    walk_sub_tree_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 443, 25), walk_sub_tree_930, *[result_div_933, hg_934], **kwargs_935)
                    
                    # Assigning a type to the variable 'hg' (line 443)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'hg', walk_sub_tree_call_result_936)

                    if more_types_in_union_928:
                        # SSA join for if statement (line 442)
                        module_type_store = module_type_store.join_ssa_context()


                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 439)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 445):
            
            # Call to grav_sub(...): (line 445)
            # Processing the call arguments (line 445)
            # Getting the type of 'hg' (line 445)
            hg_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 31), 'hg', False)
            # Processing the call keyword arguments (line 445)
            kwargs_940 = {}
            # Getting the type of 'self' (line 445)
            self_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 17), 'self', False)
            # Obtaining the member 'grav_sub' of a type (line 445)
            grav_sub_938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 17), self_937, 'grav_sub')
            # Calling grav_sub(args, kwargs) (line 445)
            grav_sub_call_result_941 = invoke(stypy.reporting.localization.Localization(__file__, 445, 17), grav_sub_938, *[hg_939], **kwargs_940)
            
            # Assigning a type to the variable 'hg' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'hg', grav_sub_call_result_941)
            # SSA join for if statement (line 439)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'hg' (line 446)
        hg_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 15), 'hg')
        # Assigning a type to the variable 'stypy_return_type' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'stypy_return_type', hg_942)
        
        # ################# End of 'walk_sub_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'walk_sub_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 437)
        stypy_return_type_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_943)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'walk_sub_tree'
        return stypy_return_type_943


    @norecursion
    def subdiv_p(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'subdiv_p'
        module_type_store = module_type_store.open_function_context('subdiv_p', 448, 4, False)
        # Assigning a type to the variable 'self' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.subdiv_p.__dict__.__setitem__('stypy_localization', localization)
        Cell.subdiv_p.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.subdiv_p.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.subdiv_p.__dict__.__setitem__('stypy_function_name', 'Cell.subdiv_p')
        Cell.subdiv_p.__dict__.__setitem__('stypy_param_names_list', ['dsq', 'hg'])
        Cell.subdiv_p.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.subdiv_p.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.subdiv_p.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.subdiv_p.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.subdiv_p.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.subdiv_p.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.subdiv_p', ['dsq', 'hg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'subdiv_p', localization, ['dsq', 'hg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'subdiv_p(...)' code ##################

        str_944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, (-1)), 'str', '\n        Decide if the cell is too close to accept as a single term.\n        @return True if the cell is too close.\n        ')
        
        # Assigning a Call to a Name (line 453):
        
        # Call to Vec3(...): (line 453)
        # Processing the call keyword arguments (line 453)
        kwargs_946 = {}
        # Getting the type of 'Vec3' (line 453)
        Vec3_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 13), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 453)
        Vec3_call_result_947 = invoke(stypy.reporting.localization.Localization(__file__, 453, 13), Vec3_945, *[], **kwargs_946)
        
        # Assigning a type to the variable 'dr' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'dr', Vec3_call_result_947)
        
        # Call to subtraction2(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'self' (line 454)
        self_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'self', False)
        # Obtaining the member 'pos' of a type (line 454)
        pos_951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 24), self_950, 'pos')
        # Getting the type of 'hg' (line 454)
        hg_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 34), 'hg', False)
        # Obtaining the member 'pos0' of a type (line 454)
        pos0_953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 34), hg_952, 'pos0')
        # Processing the call keyword arguments (line 454)
        kwargs_954 = {}
        # Getting the type of 'dr' (line 454)
        dr_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'dr', False)
        # Obtaining the member 'subtraction2' of a type (line 454)
        subtraction2_949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), dr_948, 'subtraction2')
        # Calling subtraction2(args, kwargs) (line 454)
        subtraction2_call_result_955 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), subtraction2_949, *[pos_951, pos0_953], **kwargs_954)
        
        
        # Assigning a Call to a Name (line 455):
        
        # Call to dot(...): (line 455)
        # Processing the call keyword arguments (line 455)
        kwargs_958 = {}
        # Getting the type of 'dr' (line 455)
        dr_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 15), 'dr', False)
        # Obtaining the member 'dot' of a type (line 455)
        dot_957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 15), dr_956, 'dot')
        # Calling dot(args, kwargs) (line 455)
        dot_call_result_959 = invoke(stypy.reporting.localization.Localization(__file__, 455, 15), dot_957, *[], **kwargs_958)
        
        # Assigning a type to the variable 'drsq' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'drsq', dot_call_result_959)
        
        # Getting the type of 'drsq' (line 458)
        drsq_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 15), 'drsq')
        # Getting the type of 'dsq' (line 458)
        dsq_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 22), 'dsq')
        # Applying the binary operator '<' (line 458)
        result_lt_962 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 15), '<', drsq_960, dsq_961)
        
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'stypy_return_type', result_lt_962)
        
        # ################# End of 'subdiv_p(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'subdiv_p' in the type store
        # Getting the type of 'stypy_return_type' (line 448)
        stypy_return_type_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'subdiv_p'
        return stypy_return_type_963


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 460, 4, False)
        # Assigning a type to the variable 'self' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Cell.stypy__repr__')
        Cell.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Cell.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, (-1)), 'str', '\n        Return a string represenation of a cell.\n        @return a string represenation of a cell.\n        ')
        str_965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 15), 'str', 'Cell ')
        
        # Call to __repr__(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'self' (line 465)
        self_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 39), 'self', False)
        # Processing the call keyword arguments (line 465)
        kwargs_969 = {}
        # Getting the type of 'Node' (line 465)
        Node_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 25), 'Node', False)
        # Obtaining the member '__repr__' of a type (line 465)
        repr___967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 25), Node_966, '__repr__')
        # Calling __repr__(args, kwargs) (line 465)
        repr___call_result_970 = invoke(stypy.reporting.localization.Localization(__file__, 465, 25), repr___967, *[self_968], **kwargs_969)
        
        # Applying the binary operator '+' (line 465)
        result_add_971 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 15), '+', str_965, repr___call_result_970)
        
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', result_add_971)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 460)
        stypy_return_type_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_972)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_972


# Assigning a type to the variable 'Cell' (line 386)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 0), 'Cell', Cell)

# Assigning a Num to a Name (line 389):
int_973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 11), 'int')
# Getting the type of 'Cell'
Cell_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cell')
# Setting the type of the member 'NSUB' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cell_974, 'NSUB', int_973)
# Declaration of the 'Tree' class

class Tree:
    str_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, (-1)), 'str', '\n    A class that represents the root of the data structure used\n    to represent the N-bodies in the Barnes-Hut algorithm.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 473, 4, False)
        # Assigning a type to the variable 'self' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tree.__init__', [], None, None, defaults, varargs, kwargs)

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

        str_976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 8), 'str', 'Construct the root of the data structure that represents the N-bodies.')
        
        # Assigning a List to a Attribute (line 475):
        
        # Obtaining an instance of the builtin type 'list' (line 475)
        list_977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 475)
        
        # Getting the type of 'self' (line 475)
        self_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'self')
        # Setting the type of the member 'bodies' of a type (line 475)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), self_978, 'bodies', list_977)
        
        # Assigning a Call to a Attribute (line 476):
        
        # Call to Vec3(...): (line 476)
        # Processing the call keyword arguments (line 476)
        kwargs_980 = {}
        # Getting the type of 'Vec3' (line 476)
        Vec3_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 20), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 476)
        Vec3_call_result_981 = invoke(stypy.reporting.localization.Localization(__file__, 476, 20), Vec3_979, *[], **kwargs_980)
        
        # Getting the type of 'self' (line 476)
        self_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'self')
        # Setting the type of the member 'rmin' of a type (line 476)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), self_982, 'rmin', Vec3_call_result_981)
        
        # Assigning a BinOp to a Attribute (line 477):
        float_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 21), 'float')
        float_984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 28), 'float')
        # Applying the binary operator '*' (line 477)
        result_mul_985 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 21), '*', float_983, float_984)
        
        # Getting the type of 'self' (line 477)
        self_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'self')
        # Setting the type of the member 'rsize' of a type (line 477)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), self_986, 'rsize', result_mul_985)
        
        # Assigning a Name to a Attribute (line 478):
        # Getting the type of 'None' (line 478)
        None_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'None')
        # Getting the type of 'self' (line 478)
        self_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'self')
        # Setting the type of the member 'root' of a type (line 478)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 8), self_988, 'root', None_987)
        
        # Assigning a Num to a Subscript (line 479):
        float_989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 23), 'float')
        # Getting the type of 'self' (line 479)
        self_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'self')
        # Obtaining the member 'rmin' of a type (line 479)
        rmin_991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), self_990, 'rmin')
        int_992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 18), 'int')
        # Storing an element on a container (line 479)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 8), rmin_991, (int_992, float_989))
        
        # Assigning a Num to a Subscript (line 480):
        float_993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 23), 'float')
        # Getting the type of 'self' (line 480)
        self_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self')
        # Obtaining the member 'rmin' of a type (line 480)
        rmin_995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_994, 'rmin')
        int_996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 18), 'int')
        # Storing an element on a container (line 480)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 8), rmin_995, (int_996, float_993))
        
        # Assigning a Num to a Subscript (line 481):
        float_997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 23), 'float')
        # Getting the type of 'self' (line 481)
        self_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'self')
        # Obtaining the member 'rmin' of a type (line 481)
        rmin_999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), self_998, 'rmin')
        int_1000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 18), 'int')
        # Storing an element on a container (line 481)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 8), rmin_999, (int_1000, float_997))
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def create_test_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_test_data'
        module_type_store = module_type_store.open_function_context('create_test_data', 483, 4, False)
        # Assigning a type to the variable 'self' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Tree.create_test_data.__dict__.__setitem__('stypy_localization', localization)
        Tree.create_test_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Tree.create_test_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        Tree.create_test_data.__dict__.__setitem__('stypy_function_name', 'Tree.create_test_data')
        Tree.create_test_data.__dict__.__setitem__('stypy_param_names_list', ['nbody'])
        Tree.create_test_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        Tree.create_test_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Tree.create_test_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        Tree.create_test_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        Tree.create_test_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Tree.create_test_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tree.create_test_data', ['nbody'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_test_data', localization, ['nbody'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_test_data(...)' code ##################

        str_1001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, (-1)), 'str', '\n        Create the testdata used in the benchmark.\n        @param nbody the number of bodies to create\n        ')
        
        # Assigning a Call to a Name (line 488):
        
        # Call to Vec3(...): (line 488)
        # Processing the call keyword arguments (line 488)
        kwargs_1003 = {}
        # Getting the type of 'Vec3' (line 488)
        Vec3_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 14), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 488)
        Vec3_call_result_1004 = invoke(stypy.reporting.localization.Localization(__file__, 488, 14), Vec3_1002, *[], **kwargs_1003)
        
        # Assigning a type to the variable 'cmr' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'cmr', Vec3_call_result_1004)
        
        # Assigning a Call to a Name (line 489):
        
        # Call to Vec3(...): (line 489)
        # Processing the call keyword arguments (line 489)
        kwargs_1006 = {}
        # Getting the type of 'Vec3' (line 489)
        Vec3_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 14), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 489)
        Vec3_call_result_1007 = invoke(stypy.reporting.localization.Localization(__file__, 489, 14), Vec3_1005, *[], **kwargs_1006)
        
        # Assigning a type to the variable 'cmv' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'cmv', Vec3_call_result_1007)
        
        # Assigning a BinOp to a Name (line 491):
        float_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 14), 'float')
        # Getting the type of 'pi' (line 491)
        pi_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'pi')
        # Applying the binary operator '*' (line 491)
        result_mul_1010 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 14), '*', float_1008, pi_1009)
        
        float_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 25), 'float')
        # Applying the binary operator 'div' (line 491)
        result_div_1012 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 23), 'div', result_mul_1010, float_1011)
        
        # Assigning a type to the variable 'rsc' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'rsc', result_div_1012)
        
        # Assigning a Call to a Name (line 492):
        
        # Call to sqrt(...): (line 492)
        # Processing the call arguments (line 492)
        float_1014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 19), 'float')
        # Getting the type of 'rsc' (line 492)
        rsc_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 25), 'rsc', False)
        # Applying the binary operator 'div' (line 492)
        result_div_1016 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 19), 'div', float_1014, rsc_1015)
        
        # Processing the call keyword arguments (line 492)
        kwargs_1017 = {}
        # Getting the type of 'sqrt' (line 492)
        sqrt_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 14), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 492)
        sqrt_call_result_1018 = invoke(stypy.reporting.localization.Localization(__file__, 492, 14), sqrt_1013, *[result_div_1016], **kwargs_1017)
        
        # Assigning a type to the variable 'vsc' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'vsc', sqrt_call_result_1018)
        
        # Assigning a Num to a Name (line 493):
        int_1019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 15), 'int')
        # Assigning a type to the variable 'seed' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'seed', int_1019)
        
        # Assigning a Call to a Name (line 494):
        
        # Call to Random(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'seed' (line 494)
        seed_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 21), 'seed', False)
        # Processing the call keyword arguments (line 494)
        kwargs_1022 = {}
        # Getting the type of 'Random' (line 494)
        Random_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 14), 'Random', False)
        # Calling Random(args, kwargs) (line 494)
        Random_call_result_1023 = invoke(stypy.reporting.localization.Localization(__file__, 494, 14), Random_1020, *[seed_1021], **kwargs_1022)
        
        # Assigning a type to the variable 'rnd' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'rnd', Random_call_result_1023)
        
        # Assigning a BinOp to a Attribute (line 495):
        
        # Obtaining an instance of the builtin type 'list' (line 495)
        list_1024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 495)
        # Adding element type (line 495)
        # Getting the type of 'None' (line 495)
        None_1025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 23), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 22), list_1024, None_1025)
        
        # Getting the type of 'nbody' (line 495)
        nbody_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 31), 'nbody')
        # Applying the binary operator '*' (line 495)
        result_mul_1027 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 22), '*', list_1024, nbody_1026)
        
        # Getting the type of 'self' (line 495)
        self_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'self')
        # Setting the type of the member 'bodies' of a type (line 495)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 8), self_1028, 'bodies', result_mul_1027)
        
        # Assigning a BinOp to a Name (line 496):
        float_1029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 19), 'float')
        
        # Call to float(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'nbody' (line 496)
        nbody_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 31), 'nbody', False)
        # Processing the call keyword arguments (line 496)
        kwargs_1032 = {}
        # Getting the type of 'float' (line 496)
        float_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 25), 'float', False)
        # Calling float(args, kwargs) (line 496)
        float_call_result_1033 = invoke(stypy.reporting.localization.Localization(__file__, 496, 25), float_1030, *[nbody_1031], **kwargs_1032)
        
        # Applying the binary operator 'div' (line 496)
        result_div_1034 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 19), 'div', float_1029, float_call_result_1033)
        
        # Assigning a type to the variable 'aux_mass' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'aux_mass', result_div_1034)
        
        
        # Call to xrange(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'nbody' (line 498)
        nbody_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'nbody', False)
        # Processing the call keyword arguments (line 498)
        kwargs_1037 = {}
        # Getting the type of 'xrange' (line 498)
        xrange_1035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 498)
        xrange_call_result_1038 = invoke(stypy.reporting.localization.Localization(__file__, 498, 17), xrange_1035, *[nbody_1036], **kwargs_1037)
        
        # Assigning a type to the variable 'xrange_call_result_1038' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'xrange_call_result_1038', xrange_call_result_1038)
        # Testing if the for loop is going to be iterated (line 498)
        # Testing the type of a for loop iterable (line 498)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 498, 8), xrange_call_result_1038)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 498, 8), xrange_call_result_1038):
            # Getting the type of the for loop variable (line 498)
            for_loop_var_1039 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 498, 8), xrange_call_result_1038)
            # Assigning a type to the variable 'i' (line 498)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'i', for_loop_var_1039)
            # SSA begins for a for statement (line 498)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 499):
            
            # Call to Body(...): (line 499)
            # Processing the call keyword arguments (line 499)
            kwargs_1041 = {}
            # Getting the type of 'Body' (line 499)
            Body_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'Body', False)
            # Calling Body(args, kwargs) (line 499)
            Body_call_result_1042 = invoke(stypy.reporting.localization.Localization(__file__, 499, 16), Body_1040, *[], **kwargs_1041)
            
            # Assigning a type to the variable 'p' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'p', Body_call_result_1042)
            
            # Assigning a Name to a Subscript (line 500):
            # Getting the type of 'p' (line 500)
            p_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'p')
            # Getting the type of 'self' (line 500)
            self_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'self')
            # Obtaining the member 'bodies' of a type (line 500)
            bodies_1045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), self_1044, 'bodies')
            # Getting the type of 'i' (line 500)
            i_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'i')
            # Storing an element on a container (line 500)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 12), bodies_1045, (i_1046, p_1043))
            
            # Assigning a Name to a Attribute (line 501):
            # Getting the type of 'aux_mass' (line 501)
            aux_mass_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 21), 'aux_mass')
            # Getting the type of 'p' (line 501)
            p_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'p')
            # Setting the type of the member 'mass' of a type (line 501)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), p_1048, 'mass', aux_mass_1047)
            
            # Assigning a Call to a Name (line 503):
            
            # Call to uniform(...): (line 503)
            # Processing the call arguments (line 503)
            float_1051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 29), 'float')
            float_1052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 34), 'float')
            # Processing the call keyword arguments (line 503)
            kwargs_1053 = {}
            # Getting the type of 'rnd' (line 503)
            rnd_1049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 17), 'rnd', False)
            # Obtaining the member 'uniform' of a type (line 503)
            uniform_1050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 17), rnd_1049, 'uniform')
            # Calling uniform(args, kwargs) (line 503)
            uniform_call_result_1054 = invoke(stypy.reporting.localization.Localization(__file__, 503, 17), uniform_1050, *[float_1051, float_1052], **kwargs_1053)
            
            # Assigning a type to the variable 't1' (line 503)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 't1', uniform_call_result_1054)
            
            # Assigning a BinOp to a Name (line 504):
            
            # Call to pow(...): (line 504)
            # Processing the call arguments (line 504)
            # Getting the type of 't1' (line 504)
            t1_1056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 21), 't1', False)
            float_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 26), 'float')
            float_1058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 33), 'float')
            # Applying the binary operator 'div' (line 504)
            result_div_1059 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 26), 'div', float_1057, float_1058)
            
            # Processing the call keyword arguments (line 504)
            kwargs_1060 = {}
            # Getting the type of 'pow' (line 504)
            pow_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'pow', False)
            # Calling pow(args, kwargs) (line 504)
            pow_call_result_1061 = invoke(stypy.reporting.localization.Localization(__file__, 504, 17), pow_1055, *[t1_1056, result_div_1059], **kwargs_1060)
            
            float_1062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 41), 'float')
            # Applying the binary operator '-' (line 504)
            result_sub_1063 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 17), '-', pow_call_result_1061, float_1062)
            
            # Assigning a type to the variable 't1' (line 504)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 't1', result_sub_1063)
            
            # Assigning a BinOp to a Name (line 505):
            float_1064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 16), 'float')
            
            # Call to sqrt(...): (line 505)
            # Processing the call arguments (line 505)
            # Getting the type of 't1' (line 505)
            t1_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 27), 't1', False)
            # Processing the call keyword arguments (line 505)
            kwargs_1067 = {}
            # Getting the type of 'sqrt' (line 505)
            sqrt_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 22), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 505)
            sqrt_call_result_1068 = invoke(stypy.reporting.localization.Localization(__file__, 505, 22), sqrt_1065, *[t1_1066], **kwargs_1067)
            
            # Applying the binary operator 'div' (line 505)
            result_div_1069 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 16), 'div', float_1064, sqrt_call_result_1068)
            
            # Assigning a type to the variable 'r' (line 505)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'r', result_div_1069)
            
            # Assigning a Num to a Name (line 507):
            float_1070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 20), 'float')
            # Assigning a type to the variable 'coeff' (line 507)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'coeff', float_1070)
            
            
            # Call to xrange(...): (line 508)
            # Processing the call arguments (line 508)
            # Getting the type of 'Vec3' (line 508)
            Vec3_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 28), 'Vec3', False)
            # Obtaining the member 'NDIM' of a type (line 508)
            NDIM_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 28), Vec3_1072, 'NDIM')
            # Processing the call keyword arguments (line 508)
            kwargs_1074 = {}
            # Getting the type of 'xrange' (line 508)
            xrange_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 508)
            xrange_call_result_1075 = invoke(stypy.reporting.localization.Localization(__file__, 508, 21), xrange_1071, *[NDIM_1073], **kwargs_1074)
            
            # Assigning a type to the variable 'xrange_call_result_1075' (line 508)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'xrange_call_result_1075', xrange_call_result_1075)
            # Testing if the for loop is going to be iterated (line 508)
            # Testing the type of a for loop iterable (line 508)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 508, 12), xrange_call_result_1075)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 508, 12), xrange_call_result_1075):
                # Getting the type of the for loop variable (line 508)
                for_loop_var_1076 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 508, 12), xrange_call_result_1075)
                # Assigning a type to the variable 'k' (line 508)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'k', for_loop_var_1076)
                # SSA begins for a for statement (line 508)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 509):
                
                # Call to uniform(...): (line 509)
                # Processing the call arguments (line 509)
                float_1079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 32), 'float')
                float_1080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 37), 'float')
                # Processing the call keyword arguments (line 509)
                kwargs_1081 = {}
                # Getting the type of 'rnd' (line 509)
                rnd_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 'rnd', False)
                # Obtaining the member 'uniform' of a type (line 509)
                uniform_1078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 20), rnd_1077, 'uniform')
                # Calling uniform(args, kwargs) (line 509)
                uniform_call_result_1082 = invoke(stypy.reporting.localization.Localization(__file__, 509, 20), uniform_1078, *[float_1079, float_1080], **kwargs_1081)
                
                # Assigning a type to the variable 'r' (line 509)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'r', uniform_call_result_1082)
                
                # Assigning a BinOp to a Subscript (line 510):
                # Getting the type of 'coeff' (line 510)
                coeff_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 27), 'coeff')
                # Getting the type of 'r' (line 510)
                r_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 35), 'r')
                # Applying the binary operator '*' (line 510)
                result_mul_1085 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 27), '*', coeff_1083, r_1084)
                
                # Getting the type of 'p' (line 510)
                p_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'p')
                # Obtaining the member 'pos' of a type (line 510)
                pos_1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), p_1086, 'pos')
                # Getting the type of 'k' (line 510)
                k_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'k')
                # Storing an element on a container (line 510)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 16), pos_1087, (k_1088, result_mul_1085))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'cmr' (line 512)
            cmr_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'cmr')
            # Getting the type of 'p' (line 512)
            p_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 'p')
            # Obtaining the member 'pos' of a type (line 512)
            pos_1091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 19), p_1090, 'pos')
            # Applying the binary operator '+=' (line 512)
            result_iadd_1092 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 12), '+=', cmr_1089, pos_1091)
            # Assigning a type to the variable 'cmr' (line 512)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'cmr', result_iadd_1092)
            
            
            # Getting the type of 'True' (line 514)
            True_1093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 18), 'True')
            # Assigning a type to the variable 'True_1093' (line 514)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'True_1093', True_1093)
            # Testing if the while is going to be iterated (line 514)
            # Testing the type of an if condition (line 514)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 12), True_1093)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 514, 12), True_1093):
                
                # Assigning a Call to a Name (line 515):
                
                # Call to uniform(...): (line 515)
                # Processing the call arguments (line 515)
                float_1096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 32), 'float')
                float_1097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 37), 'float')
                # Processing the call keyword arguments (line 515)
                kwargs_1098 = {}
                # Getting the type of 'rnd' (line 515)
                rnd_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 20), 'rnd', False)
                # Obtaining the member 'uniform' of a type (line 515)
                uniform_1095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 20), rnd_1094, 'uniform')
                # Calling uniform(args, kwargs) (line 515)
                uniform_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 515, 20), uniform_1095, *[float_1096, float_1097], **kwargs_1098)
                
                # Assigning a type to the variable 'x' (line 515)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 16), 'x', uniform_call_result_1099)
                
                # Assigning a Call to a Name (line 516):
                
                # Call to uniform(...): (line 516)
                # Processing the call arguments (line 516)
                float_1102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 32), 'float')
                float_1103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 37), 'float')
                # Processing the call keyword arguments (line 516)
                kwargs_1104 = {}
                # Getting the type of 'rnd' (line 516)
                rnd_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'rnd', False)
                # Obtaining the member 'uniform' of a type (line 516)
                uniform_1101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 20), rnd_1100, 'uniform')
                # Calling uniform(args, kwargs) (line 516)
                uniform_call_result_1105 = invoke(stypy.reporting.localization.Localization(__file__, 516, 20), uniform_1101, *[float_1102, float_1103], **kwargs_1104)
                
                # Assigning a type to the variable 'y' (line 516)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'y', uniform_call_result_1105)
                
                # Getting the type of 'y' (line 517)
                y_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'y')
                # Getting the type of 'x' (line 517)
                x_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 25), 'x')
                # Getting the type of 'x' (line 517)
                x_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 29), 'x')
                # Applying the binary operator '*' (line 517)
                result_mul_1109 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 25), '*', x_1107, x_1108)
                
                
                # Call to pow(...): (line 517)
                # Processing the call arguments (line 517)
                float_1111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 37), 'float')
                # Getting the type of 'x' (line 517)
                x_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 43), 'x', False)
                # Getting the type of 'x' (line 517)
                x_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 47), 'x', False)
                # Applying the binary operator '*' (line 517)
                result_mul_1114 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 43), '*', x_1112, x_1113)
                
                # Applying the binary operator '-' (line 517)
                result_sub_1115 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 37), '-', float_1111, result_mul_1114)
                
                float_1116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 50), 'float')
                # Processing the call keyword arguments (line 517)
                kwargs_1117 = {}
                # Getting the type of 'pow' (line 517)
                pow_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 33), 'pow', False)
                # Calling pow(args, kwargs) (line 517)
                pow_call_result_1118 = invoke(stypy.reporting.localization.Localization(__file__, 517, 33), pow_1110, *[result_sub_1115, float_1116], **kwargs_1117)
                
                # Applying the binary operator '*' (line 517)
                result_mul_1119 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 31), '*', result_mul_1109, pow_call_result_1118)
                
                # Applying the binary operator '<=' (line 517)
                result_le_1120 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 19), '<=', y_1106, result_mul_1119)
                
                # Testing if the type of an if condition is none (line 517)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 517, 16), result_le_1120):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 517)
                    if_condition_1121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 16), result_le_1120)
                    # Assigning a type to the variable 'if_condition_1121' (line 517)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 16), 'if_condition_1121', if_condition_1121)
                    # SSA begins for if statement (line 517)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 517)
                    module_type_store = module_type_store.join_ssa_context()
                    


            
            
            # Assigning a BinOp to a Name (line 519):
            
            # Call to sqrt(...): (line 519)
            # Processing the call arguments (line 519)
            float_1123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 21), 'float')
            # Processing the call keyword arguments (line 519)
            kwargs_1124 = {}
            # Getting the type of 'sqrt' (line 519)
            sqrt_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 519)
            sqrt_call_result_1125 = invoke(stypy.reporting.localization.Localization(__file__, 519, 16), sqrt_1122, *[float_1123], **kwargs_1124)
            
            # Getting the type of 'x' (line 519)
            x_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 28), 'x')
            # Applying the binary operator '*' (line 519)
            result_mul_1127 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 16), '*', sqrt_call_result_1125, x_1126)
            
            
            # Call to pow(...): (line 519)
            # Processing the call arguments (line 519)
            int_1129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 36), 'int')
            # Getting the type of 'r' (line 519)
            r_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 40), 'r', False)
            # Getting the type of 'r' (line 519)
            r_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 44), 'r', False)
            # Applying the binary operator '*' (line 519)
            result_mul_1132 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 40), '*', r_1130, r_1131)
            
            # Applying the binary operator '+' (line 519)
            result_add_1133 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 36), '+', int_1129, result_mul_1132)
            
            float_1134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 47), 'float')
            # Processing the call keyword arguments (line 519)
            kwargs_1135 = {}
            # Getting the type of 'pow' (line 519)
            pow_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 32), 'pow', False)
            # Calling pow(args, kwargs) (line 519)
            pow_call_result_1136 = invoke(stypy.reporting.localization.Localization(__file__, 519, 32), pow_1128, *[result_add_1133, float_1134], **kwargs_1135)
            
            # Applying the binary operator 'div' (line 519)
            result_div_1137 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 30), 'div', result_mul_1127, pow_call_result_1136)
            
            # Assigning a type to the variable 'v' (line 519)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'v', result_div_1137)
            
            # Assigning a BinOp to a Name (line 521):
            # Getting the type of 'vsc' (line 521)
            vsc_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 18), 'vsc')
            # Getting the type of 'v' (line 521)
            v_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 'v')
            # Applying the binary operator '*' (line 521)
            result_mul_1140 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 18), '*', vsc_1138, v_1139)
            
            # Assigning a type to the variable 'rad' (line 521)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'rad', result_mul_1140)
            
            # Getting the type of 'True' (line 522)
            True_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 18), 'True')
            # Assigning a type to the variable 'True_1141' (line 522)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'True_1141', True_1141)
            # Testing if the while is going to be iterated (line 522)
            # Testing the type of an if condition (line 522)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 12), True_1141)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 522, 12), True_1141):
                
                
                # Call to xrange(...): (line 523)
                # Processing the call arguments (line 523)
                # Getting the type of 'Vec3' (line 523)
                Vec3_1143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 32), 'Vec3', False)
                # Obtaining the member 'NDIM' of a type (line 523)
                NDIM_1144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 32), Vec3_1143, 'NDIM')
                # Processing the call keyword arguments (line 523)
                kwargs_1145 = {}
                # Getting the type of 'xrange' (line 523)
                xrange_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 25), 'xrange', False)
                # Calling xrange(args, kwargs) (line 523)
                xrange_call_result_1146 = invoke(stypy.reporting.localization.Localization(__file__, 523, 25), xrange_1142, *[NDIM_1144], **kwargs_1145)
                
                # Assigning a type to the variable 'xrange_call_result_1146' (line 523)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'xrange_call_result_1146', xrange_call_result_1146)
                # Testing if the for loop is going to be iterated (line 523)
                # Testing the type of a for loop iterable (line 523)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 523, 16), xrange_call_result_1146)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 523, 16), xrange_call_result_1146):
                    # Getting the type of the for loop variable (line 523)
                    for_loop_var_1147 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 523, 16), xrange_call_result_1146)
                    # Assigning a type to the variable 'k' (line 523)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'k', for_loop_var_1147)
                    # SSA begins for a for statement (line 523)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Call to a Subscript (line 524):
                    
                    # Call to uniform(...): (line 524)
                    # Processing the call arguments (line 524)
                    float_1150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 43), 'float')
                    float_1151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 49), 'float')
                    # Processing the call keyword arguments (line 524)
                    kwargs_1152 = {}
                    # Getting the type of 'rnd' (line 524)
                    rnd_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 31), 'rnd', False)
                    # Obtaining the member 'uniform' of a type (line 524)
                    uniform_1149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 31), rnd_1148, 'uniform')
                    # Calling uniform(args, kwargs) (line 524)
                    uniform_call_result_1153 = invoke(stypy.reporting.localization.Localization(__file__, 524, 31), uniform_1149, *[float_1150, float_1151], **kwargs_1152)
                    
                    # Getting the type of 'p' (line 524)
                    p_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 20), 'p')
                    # Obtaining the member 'vel' of a type (line 524)
                    vel_1155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 20), p_1154, 'vel')
                    # Getting the type of 'k' (line 524)
                    k_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 26), 'k')
                    # Storing an element on a container (line 524)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 20), vel_1155, (k_1156, uniform_call_result_1153))
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Name (line 525):
                
                # Call to dot(...): (line 525)
                # Processing the call keyword arguments (line 525)
                kwargs_1160 = {}
                # Getting the type of 'p' (line 525)
                p_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 22), 'p', False)
                # Obtaining the member 'vel' of a type (line 525)
                vel_1158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 22), p_1157, 'vel')
                # Obtaining the member 'dot' of a type (line 525)
                dot_1159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 22), vel_1158, 'dot')
                # Calling dot(args, kwargs) (line 525)
                dot_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 525, 22), dot_1159, *[], **kwargs_1160)
                
                # Assigning a type to the variable 'rsq' (line 525)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'rsq', dot_call_result_1161)
                
                # Getting the type of 'rsq' (line 526)
                rsq_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 19), 'rsq')
                float_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 26), 'float')
                # Applying the binary operator '<=' (line 526)
                result_le_1164 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 19), '<=', rsq_1162, float_1163)
                
                # Testing if the type of an if condition is none (line 526)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 526, 16), result_le_1164):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 526)
                    if_condition_1165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 526, 16), result_le_1164)
                    # Assigning a type to the variable 'if_condition_1165' (line 526)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'if_condition_1165', if_condition_1165)
                    # SSA begins for if statement (line 526)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 526)
                    module_type_store = module_type_store.join_ssa_context()
                    


            
            
            # Assigning a BinOp to a Name (line 528):
            # Getting the type of 'rad' (line 528)
            rad_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 19), 'rad')
            
            # Call to sqrt(...): (line 528)
            # Processing the call arguments (line 528)
            # Getting the type of 'rsq' (line 528)
            rsq_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 30), 'rsq', False)
            # Processing the call keyword arguments (line 528)
            kwargs_1169 = {}
            # Getting the type of 'sqrt' (line 528)
            sqrt_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 25), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 528)
            sqrt_call_result_1170 = invoke(stypy.reporting.localization.Localization(__file__, 528, 25), sqrt_1167, *[rsq_1168], **kwargs_1169)
            
            # Applying the binary operator 'div' (line 528)
            result_div_1171 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 19), 'div', rad_1166, sqrt_call_result_1170)
            
            # Assigning a type to the variable 'rsc1' (line 528)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'rsc1', result_div_1171)
            
            # Getting the type of 'p' (line 529)
            p_1172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'p')
            # Obtaining the member 'vel' of a type (line 529)
            vel_1173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), p_1172, 'vel')
            # Getting the type of 'rsc1' (line 529)
            rsc1_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 21), 'rsc1')
            # Applying the binary operator '*=' (line 529)
            result_imul_1175 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 12), '*=', vel_1173, rsc1_1174)
            # Getting the type of 'p' (line 529)
            p_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'p')
            # Setting the type of the member 'vel' of a type (line 529)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), p_1176, 'vel', result_imul_1175)
            
            
            # Getting the type of 'cmv' (line 530)
            cmv_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'cmv')
            # Getting the type of 'p' (line 530)
            p_1178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 19), 'p')
            # Obtaining the member 'vel' of a type (line 530)
            vel_1179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 19), p_1178, 'vel')
            # Applying the binary operator '+=' (line 530)
            result_iadd_1180 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 12), '+=', cmv_1177, vel_1179)
            # Assigning a type to the variable 'cmv' (line 530)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'cmv', result_iadd_1180)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'cmr' (line 532)
        cmr_1181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'cmr')
        
        # Call to float(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'nbody' (line 532)
        nbody_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 21), 'nbody', False)
        # Processing the call keyword arguments (line 532)
        kwargs_1184 = {}
        # Getting the type of 'float' (line 532)
        float_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 15), 'float', False)
        # Calling float(args, kwargs) (line 532)
        float_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 532, 15), float_1182, *[nbody_1183], **kwargs_1184)
        
        # Applying the binary operator 'div=' (line 532)
        result_div_1186 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 8), 'div=', cmr_1181, float_call_result_1185)
        # Assigning a type to the variable 'cmr' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'cmr', result_div_1186)
        
        
        # Getting the type of 'cmv' (line 533)
        cmv_1187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'cmv')
        
        # Call to float(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'nbody' (line 533)
        nbody_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 21), 'nbody', False)
        # Processing the call keyword arguments (line 533)
        kwargs_1190 = {}
        # Getting the type of 'float' (line 533)
        float_1188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'float', False)
        # Calling float(args, kwargs) (line 533)
        float_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 533, 15), float_1188, *[nbody_1189], **kwargs_1190)
        
        # Applying the binary operator 'div=' (line 533)
        result_div_1192 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 8), 'div=', cmv_1187, float_call_result_1191)
        # Assigning a type to the variable 'cmv' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'cmv', result_div_1192)
        
        
        # Getting the type of 'self' (line 535)
        self_1193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 17), 'self')
        # Obtaining the member 'bodies' of a type (line 535)
        bodies_1194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 17), self_1193, 'bodies')
        # Assigning a type to the variable 'bodies_1194' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'bodies_1194', bodies_1194)
        # Testing if the for loop is going to be iterated (line 535)
        # Testing the type of a for loop iterable (line 535)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 535, 8), bodies_1194)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 535, 8), bodies_1194):
            # Getting the type of the for loop variable (line 535)
            for_loop_var_1195 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 535, 8), bodies_1194)
            # Assigning a type to the variable 'b' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'b', for_loop_var_1195)
            # SSA begins for a for statement (line 535)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'b' (line 536)
            b_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'b')
            # Obtaining the member 'pos' of a type (line 536)
            pos_1197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 12), b_1196, 'pos')
            # Getting the type of 'cmr' (line 536)
            cmr_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 21), 'cmr')
            # Applying the binary operator '-=' (line 536)
            result_isub_1199 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 12), '-=', pos_1197, cmr_1198)
            # Getting the type of 'b' (line 536)
            b_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'b')
            # Setting the type of the member 'pos' of a type (line 536)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 12), b_1200, 'pos', result_isub_1199)
            
            
            # Getting the type of 'b' (line 537)
            b_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'b')
            # Obtaining the member 'vel' of a type (line 537)
            vel_1202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), b_1201, 'vel')
            # Getting the type of 'cmv' (line 537)
            cmv_1203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 21), 'cmv')
            # Applying the binary operator '-=' (line 537)
            result_isub_1204 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 12), '-=', vel_1202, cmv_1203)
            # Getting the type of 'b' (line 537)
            b_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'b')
            # Setting the type of the member 'vel' of a type (line 537)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), b_1205, 'vel', result_isub_1204)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'create_test_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_test_data' in the type store
        # Getting the type of 'stypy_return_type' (line 483)
        stypy_return_type_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1206)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_test_data'
        return stypy_return_type_1206


    @norecursion
    def step_system(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'step_system'
        module_type_store = module_type_store.open_function_context('step_system', 539, 4, False)
        # Assigning a type to the variable 'self' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Tree.step_system.__dict__.__setitem__('stypy_localization', localization)
        Tree.step_system.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Tree.step_system.__dict__.__setitem__('stypy_type_store', module_type_store)
        Tree.step_system.__dict__.__setitem__('stypy_function_name', 'Tree.step_system')
        Tree.step_system.__dict__.__setitem__('stypy_param_names_list', ['nstep'])
        Tree.step_system.__dict__.__setitem__('stypy_varargs_param_name', None)
        Tree.step_system.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Tree.step_system.__dict__.__setitem__('stypy_call_defaults', defaults)
        Tree.step_system.__dict__.__setitem__('stypy_call_varargs', varargs)
        Tree.step_system.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Tree.step_system.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tree.step_system', ['nstep'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'step_system', localization, ['nstep'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'step_system(...)' code ##################

        str_1207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, (-1)), 'str', '\n        Advance the N-body system one time-step.\n        @param nstep the current time step\n        ')
        
        # Assigning a Name to a Attribute (line 545):
        # Getting the type of 'None' (line 545)
        None_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 20), 'None')
        # Getting the type of 'self' (line 545)
        self_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'self')
        # Setting the type of the member 'root' of a type (line 545)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 8), self_1209, 'root', None_1208)
        
        # Call to make_tree(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'nstep' (line 547)
        nstep_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 23), 'nstep', False)
        # Processing the call keyword arguments (line 547)
        kwargs_1213 = {}
        # Getting the type of 'self' (line 547)
        self_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'self', False)
        # Obtaining the member 'make_tree' of a type (line 547)
        make_tree_1211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 8), self_1210, 'make_tree')
        # Calling make_tree(args, kwargs) (line 547)
        make_tree_call_result_1214 = invoke(stypy.reporting.localization.Localization(__file__, 547, 8), make_tree_1211, *[nstep_1212], **kwargs_1213)
        
        
        
        # Call to reversed(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'self' (line 550)
        self_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 26), 'self', False)
        # Obtaining the member 'bodies' of a type (line 550)
        bodies_1217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 26), self_1216, 'bodies')
        # Processing the call keyword arguments (line 550)
        kwargs_1218 = {}
        # Getting the type of 'reversed' (line 550)
        reversed_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 17), 'reversed', False)
        # Calling reversed(args, kwargs) (line 550)
        reversed_call_result_1219 = invoke(stypy.reporting.localization.Localization(__file__, 550, 17), reversed_1215, *[bodies_1217], **kwargs_1218)
        
        # Assigning a type to the variable 'reversed_call_result_1219' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'reversed_call_result_1219', reversed_call_result_1219)
        # Testing if the for loop is going to be iterated (line 550)
        # Testing the type of a for loop iterable (line 550)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 550, 8), reversed_call_result_1219)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 550, 8), reversed_call_result_1219):
            # Getting the type of the for loop variable (line 550)
            for_loop_var_1220 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 550, 8), reversed_call_result_1219)
            # Assigning a type to the variable 'b' (line 550)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'b', for_loop_var_1220)
            # SSA begins for a for statement (line 550)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to hack_gravity(...): (line 551)
            # Processing the call arguments (line 551)
            # Getting the type of 'self' (line 551)
            self_1223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 27), 'self', False)
            # Obtaining the member 'rsize' of a type (line 551)
            rsize_1224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 27), self_1223, 'rsize')
            # Getting the type of 'self' (line 551)
            self_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 39), 'self', False)
            # Obtaining the member 'root' of a type (line 551)
            root_1226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 39), self_1225, 'root')
            # Processing the call keyword arguments (line 551)
            kwargs_1227 = {}
            # Getting the type of 'b' (line 551)
            b_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'b', False)
            # Obtaining the member 'hack_gravity' of a type (line 551)
            hack_gravity_1222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 12), b_1221, 'hack_gravity')
            # Calling hack_gravity(args, kwargs) (line 551)
            hack_gravity_call_result_1228 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), hack_gravity_1222, *[rsize_1224, root_1226], **kwargs_1227)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to vp(...): (line 552)
        # Processing the call arguments (line 552)
        # Getting the type of 'self' (line 552)
        self_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'self', False)
        # Obtaining the member 'bodies' of a type (line 552)
        bodies_1232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 16), self_1231, 'bodies')
        # Getting the type of 'nstep' (line 552)
        nstep_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 29), 'nstep', False)
        # Processing the call keyword arguments (line 552)
        kwargs_1234 = {}
        # Getting the type of 'Tree' (line 552)
        Tree_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'Tree', False)
        # Obtaining the member 'vp' of a type (line 552)
        vp_1230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 8), Tree_1229, 'vp')
        # Calling vp(args, kwargs) (line 552)
        vp_call_result_1235 = invoke(stypy.reporting.localization.Localization(__file__, 552, 8), vp_1230, *[bodies_1232, nstep_1233], **kwargs_1234)
        
        
        # ################# End of 'step_system(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'step_system' in the type store
        # Getting the type of 'stypy_return_type' (line 539)
        stypy_return_type_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'step_system'
        return stypy_return_type_1236


    @norecursion
    def make_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_tree'
        module_type_store = module_type_store.open_function_context('make_tree', 554, 4, False)
        # Assigning a type to the variable 'self' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Tree.make_tree.__dict__.__setitem__('stypy_localization', localization)
        Tree.make_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Tree.make_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Tree.make_tree.__dict__.__setitem__('stypy_function_name', 'Tree.make_tree')
        Tree.make_tree.__dict__.__setitem__('stypy_param_names_list', ['nstep'])
        Tree.make_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Tree.make_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Tree.make_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Tree.make_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Tree.make_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Tree.make_tree.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tree.make_tree', ['nstep'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_tree', localization, ['nstep'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_tree(...)' code ##################

        str_1237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, (-1)), 'str', '\n        Initialize the tree structure for hack force calculation.\n        @param nsteps the current time step\n        ')
        
        
        # Call to reversed(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'self' (line 559)
        self_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 26), 'self', False)
        # Obtaining the member 'bodies' of a type (line 559)
        bodies_1240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 26), self_1239, 'bodies')
        # Processing the call keyword arguments (line 559)
        kwargs_1241 = {}
        # Getting the type of 'reversed' (line 559)
        reversed_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 17), 'reversed', False)
        # Calling reversed(args, kwargs) (line 559)
        reversed_call_result_1242 = invoke(stypy.reporting.localization.Localization(__file__, 559, 17), reversed_1238, *[bodies_1240], **kwargs_1241)
        
        # Assigning a type to the variable 'reversed_call_result_1242' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'reversed_call_result_1242', reversed_call_result_1242)
        # Testing if the for loop is going to be iterated (line 559)
        # Testing the type of a for loop iterable (line 559)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 559, 8), reversed_call_result_1242)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 559, 8), reversed_call_result_1242):
            # Getting the type of the for loop variable (line 559)
            for_loop_var_1243 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 559, 8), reversed_call_result_1242)
            # Assigning a type to the variable 'q' (line 559)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'q', for_loop_var_1243)
            # SSA begins for a for statement (line 559)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'q' (line 560)
            q_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'q')
            # Obtaining the member 'mass' of a type (line 560)
            mass_1245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 15), q_1244, 'mass')
            float_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 25), 'float')
            # Applying the binary operator '!=' (line 560)
            result_ne_1247 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 15), '!=', mass_1245, float_1246)
            
            # Testing if the type of an if condition is none (line 560)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 560, 12), result_ne_1247):
                pass
            else:
                
                # Testing the type of an if condition (line 560)
                if_condition_1248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 12), result_ne_1247)
                # Assigning a type to the variable 'if_condition_1248' (line 560)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'if_condition_1248', if_condition_1248)
                # SSA begins for if statement (line 560)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to expand_box(...): (line 561)
                # Processing the call arguments (line 561)
                # Getting the type of 'self' (line 561)
                self_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 29), 'self', False)
                # Getting the type of 'nstep' (line 561)
                nstep_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 35), 'nstep', False)
                # Processing the call keyword arguments (line 561)
                kwargs_1253 = {}
                # Getting the type of 'q' (line 561)
                q_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'q', False)
                # Obtaining the member 'expand_box' of a type (line 561)
                expand_box_1250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), q_1249, 'expand_box')
                # Calling expand_box(args, kwargs) (line 561)
                expand_box_call_result_1254 = invoke(stypy.reporting.localization.Localization(__file__, 561, 16), expand_box_1250, *[self_1251, nstep_1252], **kwargs_1253)
                
                
                # Assigning a Call to a Name (line 562):
                
                # Call to int_coord(...): (line 562)
                # Processing the call arguments (line 562)
                # Getting the type of 'q' (line 562)
                q_1257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 38), 'q', False)
                # Obtaining the member 'pos' of a type (line 562)
                pos_1258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 38), q_1257, 'pos')
                # Processing the call keyword arguments (line 562)
                kwargs_1259 = {}
                # Getting the type of 'self' (line 562)
                self_1255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 23), 'self', False)
                # Obtaining the member 'int_coord' of a type (line 562)
                int_coord_1256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 23), self_1255, 'int_coord')
                # Calling int_coord(args, kwargs) (line 562)
                int_coord_call_result_1260 = invoke(stypy.reporting.localization.Localization(__file__, 562, 23), int_coord_1256, *[pos_1258], **kwargs_1259)
                
                # Assigning a type to the variable 'xqic' (line 562)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'xqic', int_coord_call_result_1260)
                
                # Type idiom detected: calculating its left and rigth part (line 563)
                # Getting the type of 'self' (line 563)
                self_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 19), 'self')
                # Obtaining the member 'root' of a type (line 563)
                root_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 19), self_1261, 'root')
                # Getting the type of 'None' (line 563)
                None_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 32), 'None')
                
                (may_be_1264, more_types_in_union_1265) = may_be_none(root_1262, None_1263)

                if may_be_1264:

                    if more_types_in_union_1265:
                        # Runtime conditional SSA (line 563)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Name to a Attribute (line 564):
                    # Getting the type of 'q' (line 564)
                    q_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 32), 'q')
                    # Getting the type of 'self' (line 564)
                    self_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'self')
                    # Setting the type of the member 'root' of a type (line 564)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 20), self_1267, 'root', q_1266)

                    if more_types_in_union_1265:
                        # Runtime conditional SSA for else branch (line 563)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_1264) or more_types_in_union_1265):
                    
                    # Assigning a Call to a Attribute (line 566):
                    
                    # Call to load_tree(...): (line 566)
                    # Processing the call arguments (line 566)
                    # Getting the type of 'q' (line 566)
                    q_1271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 52), 'q', False)
                    # Getting the type of 'xqic' (line 566)
                    xqic_1272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 55), 'xqic', False)
                    # Getting the type of 'Node' (line 566)
                    Node_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 61), 'Node', False)
                    # Obtaining the member 'IMAX' of a type (line 566)
                    IMAX_1274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 61), Node_1273, 'IMAX')
                    int_1275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 74), 'int')
                    # Applying the binary operator '>>' (line 566)
                    result_rshift_1276 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 61), '>>', IMAX_1274, int_1275)
                    
                    # Getting the type of 'self' (line 566)
                    self_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 77), 'self', False)
                    # Processing the call keyword arguments (line 566)
                    kwargs_1278 = {}
                    # Getting the type of 'self' (line 566)
                    self_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 32), 'self', False)
                    # Obtaining the member 'root' of a type (line 566)
                    root_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 32), self_1268, 'root')
                    # Obtaining the member 'load_tree' of a type (line 566)
                    load_tree_1270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 32), root_1269, 'load_tree')
                    # Calling load_tree(args, kwargs) (line 566)
                    load_tree_call_result_1279 = invoke(stypy.reporting.localization.Localization(__file__, 566, 32), load_tree_1270, *[q_1271, xqic_1272, result_rshift_1276, self_1277], **kwargs_1278)
                    
                    # Getting the type of 'self' (line 566)
                    self_1280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'self')
                    # Setting the type of the member 'root' of a type (line 566)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 20), self_1280, 'root', load_tree_call_result_1279)

                    if (may_be_1264 and more_types_in_union_1265):
                        # SSA join for if statement (line 563)
                        module_type_store = module_type_store.join_ssa_context()


                
                # SSA join for if statement (line 560)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to hack_cofm(...): (line 567)
        # Processing the call keyword arguments (line 567)
        kwargs_1284 = {}
        # Getting the type of 'self' (line 567)
        self_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'self', False)
        # Obtaining the member 'root' of a type (line 567)
        root_1282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), self_1281, 'root')
        # Obtaining the member 'hack_cofm' of a type (line 567)
        hack_cofm_1283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), root_1282, 'hack_cofm')
        # Calling hack_cofm(args, kwargs) (line 567)
        hack_cofm_call_result_1285 = invoke(stypy.reporting.localization.Localization(__file__, 567, 8), hack_cofm_1283, *[], **kwargs_1284)
        
        
        # ################# End of 'make_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 554)
        stypy_return_type_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_tree'
        return stypy_return_type_1286


    @norecursion
    def int_coord(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'int_coord'
        module_type_store = module_type_store.open_function_context('int_coord', 569, 4, False)
        # Assigning a type to the variable 'self' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Tree.int_coord.__dict__.__setitem__('stypy_localization', localization)
        Tree.int_coord.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Tree.int_coord.__dict__.__setitem__('stypy_type_store', module_type_store)
        Tree.int_coord.__dict__.__setitem__('stypy_function_name', 'Tree.int_coord')
        Tree.int_coord.__dict__.__setitem__('stypy_param_names_list', ['vp'])
        Tree.int_coord.__dict__.__setitem__('stypy_varargs_param_name', None)
        Tree.int_coord.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Tree.int_coord.__dict__.__setitem__('stypy_call_defaults', defaults)
        Tree.int_coord.__dict__.__setitem__('stypy_call_varargs', varargs)
        Tree.int_coord.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Tree.int_coord.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tree.int_coord', ['vp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'int_coord', localization, ['vp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'int_coord(...)' code ##################

        str_1287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, (-1)), 'str', '\n        Compute integerized coordinates.\n        @return the coordinates or None if rp is out of bounds\n        ')
        
        # Assigning a Call to a Name (line 574):
        
        # Call to Vec3(...): (line 574)
        # Processing the call keyword arguments (line 574)
        kwargs_1289 = {}
        # Getting the type of 'Vec3' (line 574)
        Vec3_1288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 13), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 574)
        Vec3_call_result_1290 = invoke(stypy.reporting.localization.Localization(__file__, 574, 13), Vec3_1288, *[], **kwargs_1289)
        
        # Assigning a type to the variable 'xp' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'xp', Vec3_call_result_1290)
        
        # Assigning a BinOp to a Name (line 576):
        
        # Obtaining the type of the subscript
        int_1291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 18), 'int')
        # Getting the type of 'vp' (line 576)
        vp_1292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 15), 'vp')
        # Obtaining the member '__getitem__' of a type (line 576)
        getitem___1293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 15), vp_1292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 576)
        subscript_call_result_1294 = invoke(stypy.reporting.localization.Localization(__file__, 576, 15), getitem___1293, int_1291)
        
        
        # Obtaining the type of the subscript
        int_1295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 33), 'int')
        # Getting the type of 'self' (line 576)
        self_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 23), 'self')
        # Obtaining the member 'rmin' of a type (line 576)
        rmin_1297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 23), self_1296, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 576)
        getitem___1298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 23), rmin_1297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 576)
        subscript_call_result_1299 = invoke(stypy.reporting.localization.Localization(__file__, 576, 23), getitem___1298, int_1295)
        
        # Applying the binary operator '-' (line 576)
        result_sub_1300 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 15), '-', subscript_call_result_1294, subscript_call_result_1299)
        
        # Getting the type of 'self' (line 576)
        self_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 39), 'self')
        # Obtaining the member 'rsize' of a type (line 576)
        rsize_1302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 39), self_1301, 'rsize')
        # Applying the binary operator 'div' (line 576)
        result_div_1303 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 14), 'div', result_sub_1300, rsize_1302)
        
        # Assigning a type to the variable 'xsc' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'xsc', result_div_1303)
        
        # Evaluating a boolean operation
        
        float_1304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 11), 'float')
        # Getting the type of 'xsc' (line 577)
        xsc_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 18), 'xsc')
        # Applying the binary operator '<=' (line 577)
        result_le_1306 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 11), '<=', float_1304, xsc_1305)
        
        
        # Getting the type of 'xsc' (line 577)
        xsc_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 26), 'xsc')
        float_1308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 32), 'float')
        # Applying the binary operator '<' (line 577)
        result_lt_1309 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 26), '<', xsc_1307, float_1308)
        
        # Applying the binary operator 'and' (line 577)
        result_and_keyword_1310 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 11), 'and', result_le_1306, result_lt_1309)
        
        # Testing if the type of an if condition is none (line 577)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 577, 8), result_and_keyword_1310):
            # Getting the type of 'None' (line 580)
            None_1321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 580)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'stypy_return_type', None_1321)
        else:
            
            # Testing the type of an if condition (line 577)
            if_condition_1311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 577, 8), result_and_keyword_1310)
            # Assigning a type to the variable 'if_condition_1311' (line 577)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'if_condition_1311', if_condition_1311)
            # SSA begins for if statement (line 577)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 578):
            
            # Call to floor(...): (line 578)
            # Processing the call arguments (line 578)
            # Getting the type of 'Node' (line 578)
            Node_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 26), 'Node', False)
            # Obtaining the member 'IMAX' of a type (line 578)
            IMAX_1314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 26), Node_1313, 'IMAX')
            # Getting the type of 'xsc' (line 578)
            xsc_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 38), 'xsc', False)
            # Applying the binary operator '*' (line 578)
            result_mul_1316 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 26), '*', IMAX_1314, xsc_1315)
            
            # Processing the call keyword arguments (line 578)
            kwargs_1317 = {}
            # Getting the type of 'floor' (line 578)
            floor_1312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 20), 'floor', False)
            # Calling floor(args, kwargs) (line 578)
            floor_call_result_1318 = invoke(stypy.reporting.localization.Localization(__file__, 578, 20), floor_1312, *[result_mul_1316], **kwargs_1317)
            
            # Getting the type of 'xp' (line 578)
            xp_1319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'xp')
            int_1320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 15), 'int')
            # Storing an element on a container (line 578)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 12), xp_1319, (int_1320, floor_call_result_1318))
            # SSA branch for the else part of an if statement (line 577)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'None' (line 580)
            None_1321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 580)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'stypy_return_type', None_1321)
            # SSA join for if statement (line 577)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 582):
        
        # Obtaining the type of the subscript
        int_1322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 18), 'int')
        # Getting the type of 'vp' (line 582)
        vp_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 15), 'vp')
        # Obtaining the member '__getitem__' of a type (line 582)
        getitem___1324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 15), vp_1323, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 582)
        subscript_call_result_1325 = invoke(stypy.reporting.localization.Localization(__file__, 582, 15), getitem___1324, int_1322)
        
        
        # Obtaining the type of the subscript
        int_1326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 33), 'int')
        # Getting the type of 'self' (line 582)
        self_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 23), 'self')
        # Obtaining the member 'rmin' of a type (line 582)
        rmin_1328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 23), self_1327, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 582)
        getitem___1329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 23), rmin_1328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 582)
        subscript_call_result_1330 = invoke(stypy.reporting.localization.Localization(__file__, 582, 23), getitem___1329, int_1326)
        
        # Applying the binary operator '-' (line 582)
        result_sub_1331 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 15), '-', subscript_call_result_1325, subscript_call_result_1330)
        
        # Getting the type of 'self' (line 582)
        self_1332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 39), 'self')
        # Obtaining the member 'rsize' of a type (line 582)
        rsize_1333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 39), self_1332, 'rsize')
        # Applying the binary operator 'div' (line 582)
        result_div_1334 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 14), 'div', result_sub_1331, rsize_1333)
        
        # Assigning a type to the variable 'xsc' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'xsc', result_div_1334)
        
        # Evaluating a boolean operation
        
        float_1335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 11), 'float')
        # Getting the type of 'xsc' (line 583)
        xsc_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 18), 'xsc')
        # Applying the binary operator '<=' (line 583)
        result_le_1337 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 11), '<=', float_1335, xsc_1336)
        
        
        # Getting the type of 'xsc' (line 583)
        xsc_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 26), 'xsc')
        float_1339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 32), 'float')
        # Applying the binary operator '<' (line 583)
        result_lt_1340 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 26), '<', xsc_1338, float_1339)
        
        # Applying the binary operator 'and' (line 583)
        result_and_keyword_1341 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 11), 'and', result_le_1337, result_lt_1340)
        
        # Testing if the type of an if condition is none (line 583)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 583, 8), result_and_keyword_1341):
            # Getting the type of 'None' (line 586)
            None_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 586)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'stypy_return_type', None_1352)
        else:
            
            # Testing the type of an if condition (line 583)
            if_condition_1342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 8), result_and_keyword_1341)
            # Assigning a type to the variable 'if_condition_1342' (line 583)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'if_condition_1342', if_condition_1342)
            # SSA begins for if statement (line 583)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 584):
            
            # Call to floor(...): (line 584)
            # Processing the call arguments (line 584)
            # Getting the type of 'Node' (line 584)
            Node_1344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 26), 'Node', False)
            # Obtaining the member 'IMAX' of a type (line 584)
            IMAX_1345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 26), Node_1344, 'IMAX')
            # Getting the type of 'xsc' (line 584)
            xsc_1346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 38), 'xsc', False)
            # Applying the binary operator '*' (line 584)
            result_mul_1347 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 26), '*', IMAX_1345, xsc_1346)
            
            # Processing the call keyword arguments (line 584)
            kwargs_1348 = {}
            # Getting the type of 'floor' (line 584)
            floor_1343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'floor', False)
            # Calling floor(args, kwargs) (line 584)
            floor_call_result_1349 = invoke(stypy.reporting.localization.Localization(__file__, 584, 20), floor_1343, *[result_mul_1347], **kwargs_1348)
            
            # Getting the type of 'xp' (line 584)
            xp_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'xp')
            int_1351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 15), 'int')
            # Storing an element on a container (line 584)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 12), xp_1350, (int_1351, floor_call_result_1349))
            # SSA branch for the else part of an if statement (line 583)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'None' (line 586)
            None_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 586)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'stypy_return_type', None_1352)
            # SSA join for if statement (line 583)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 588):
        
        # Obtaining the type of the subscript
        int_1353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 18), 'int')
        # Getting the type of 'vp' (line 588)
        vp_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 15), 'vp')
        # Obtaining the member '__getitem__' of a type (line 588)
        getitem___1355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 15), vp_1354, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 588)
        subscript_call_result_1356 = invoke(stypy.reporting.localization.Localization(__file__, 588, 15), getitem___1355, int_1353)
        
        
        # Obtaining the type of the subscript
        int_1357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 33), 'int')
        # Getting the type of 'self' (line 588)
        self_1358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 23), 'self')
        # Obtaining the member 'rmin' of a type (line 588)
        rmin_1359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 23), self_1358, 'rmin')
        # Obtaining the member '__getitem__' of a type (line 588)
        getitem___1360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 23), rmin_1359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 588)
        subscript_call_result_1361 = invoke(stypy.reporting.localization.Localization(__file__, 588, 23), getitem___1360, int_1357)
        
        # Applying the binary operator '-' (line 588)
        result_sub_1362 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 15), '-', subscript_call_result_1356, subscript_call_result_1361)
        
        # Getting the type of 'self' (line 588)
        self_1363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 39), 'self')
        # Obtaining the member 'rsize' of a type (line 588)
        rsize_1364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 39), self_1363, 'rsize')
        # Applying the binary operator 'div' (line 588)
        result_div_1365 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 14), 'div', result_sub_1362, rsize_1364)
        
        # Assigning a type to the variable 'xsc' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'xsc', result_div_1365)
        
        # Evaluating a boolean operation
        
        float_1366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 11), 'float')
        # Getting the type of 'xsc' (line 589)
        xsc_1367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'xsc')
        # Applying the binary operator '<=' (line 589)
        result_le_1368 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 11), '<=', float_1366, xsc_1367)
        
        
        # Getting the type of 'xsc' (line 589)
        xsc_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 26), 'xsc')
        float_1370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 32), 'float')
        # Applying the binary operator '<' (line 589)
        result_lt_1371 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 26), '<', xsc_1369, float_1370)
        
        # Applying the binary operator 'and' (line 589)
        result_and_keyword_1372 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 11), 'and', result_le_1368, result_lt_1371)
        
        # Testing if the type of an if condition is none (line 589)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 589, 8), result_and_keyword_1372):
            # Getting the type of 'None' (line 592)
            None_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 592)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'stypy_return_type', None_1383)
        else:
            
            # Testing the type of an if condition (line 589)
            if_condition_1373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 589, 8), result_and_keyword_1372)
            # Assigning a type to the variable 'if_condition_1373' (line 589)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'if_condition_1373', if_condition_1373)
            # SSA begins for if statement (line 589)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 590):
            
            # Call to floor(...): (line 590)
            # Processing the call arguments (line 590)
            # Getting the type of 'Node' (line 590)
            Node_1375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 26), 'Node', False)
            # Obtaining the member 'IMAX' of a type (line 590)
            IMAX_1376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 26), Node_1375, 'IMAX')
            # Getting the type of 'xsc' (line 590)
            xsc_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 38), 'xsc', False)
            # Applying the binary operator '*' (line 590)
            result_mul_1378 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 26), '*', IMAX_1376, xsc_1377)
            
            # Processing the call keyword arguments (line 590)
            kwargs_1379 = {}
            # Getting the type of 'floor' (line 590)
            floor_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'floor', False)
            # Calling floor(args, kwargs) (line 590)
            floor_call_result_1380 = invoke(stypy.reporting.localization.Localization(__file__, 590, 20), floor_1374, *[result_mul_1378], **kwargs_1379)
            
            # Getting the type of 'xp' (line 590)
            xp_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'xp')
            int_1382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 15), 'int')
            # Storing an element on a container (line 590)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), xp_1381, (int_1382, floor_call_result_1380))
            # SSA branch for the else part of an if statement (line 589)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'None' (line 592)
            None_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 592)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'stypy_return_type', None_1383)
            # SSA join for if statement (line 589)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'xp' (line 594)
        xp_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 15), 'xp')
        # Assigning a type to the variable 'stypy_return_type' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'stypy_return_type', xp_1384)
        
        # ################# End of 'int_coord(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'int_coord' in the type store
        # Getting the type of 'stypy_return_type' (line 569)
        stypy_return_type_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'int_coord'
        return stypy_return_type_1385


    @staticmethod
    @norecursion
    def vp(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'vp'
        module_type_store = module_type_store.open_function_context('vp', 596, 4, False)
        
        # Passed parameters checking function
        Tree.vp.__dict__.__setitem__('stypy_localization', localization)
        Tree.vp.__dict__.__setitem__('stypy_type_of_self', None)
        Tree.vp.__dict__.__setitem__('stypy_type_store', module_type_store)
        Tree.vp.__dict__.__setitem__('stypy_function_name', 'vp')
        Tree.vp.__dict__.__setitem__('stypy_param_names_list', ['bodies', 'nstep'])
        Tree.vp.__dict__.__setitem__('stypy_varargs_param_name', None)
        Tree.vp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Tree.vp.__dict__.__setitem__('stypy_call_defaults', defaults)
        Tree.vp.__dict__.__setitem__('stypy_call_varargs', varargs)
        Tree.vp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Tree.vp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'vp', ['bodies', 'nstep'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'vp', localization, ['nstep'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'vp(...)' code ##################

        
        # Assigning a Call to a Name (line 598):
        
        # Call to Vec3(...): (line 598)
        # Processing the call keyword arguments (line 598)
        kwargs_1387 = {}
        # Getting the type of 'Vec3' (line 598)
        Vec3_1386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 598)
        Vec3_call_result_1388 = invoke(stypy.reporting.localization.Localization(__file__, 598, 15), Vec3_1386, *[], **kwargs_1387)
        
        # Assigning a type to the variable 'dacc' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'dacc', Vec3_call_result_1388)
        
        # Assigning a Call to a Name (line 599):
        
        # Call to Vec3(...): (line 599)
        # Processing the call keyword arguments (line 599)
        kwargs_1390 = {}
        # Getting the type of 'Vec3' (line 599)
        Vec3_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 15), 'Vec3', False)
        # Calling Vec3(args, kwargs) (line 599)
        Vec3_call_result_1391 = invoke(stypy.reporting.localization.Localization(__file__, 599, 15), Vec3_1389, *[], **kwargs_1390)
        
        # Assigning a type to the variable 'dvel' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'dvel', Vec3_call_result_1391)
        
        # Assigning a BinOp to a Name (line 600):
        float_1392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 15), 'float')
        # Getting the type of 'BH' (line 600)
        BH_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 21), 'BH')
        # Obtaining the member 'DTIME' of a type (line 600)
        DTIME_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 21), BH_1393, 'DTIME')
        # Applying the binary operator '*' (line 600)
        result_mul_1395 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 15), '*', float_1392, DTIME_1394)
        
        # Assigning a type to the variable 'dthf' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'dthf', result_mul_1395)
        
        
        # Call to reversed(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'bodies' (line 602)
        bodies_1397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 26), 'bodies', False)
        # Processing the call keyword arguments (line 602)
        kwargs_1398 = {}
        # Getting the type of 'reversed' (line 602)
        reversed_1396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 17), 'reversed', False)
        # Calling reversed(args, kwargs) (line 602)
        reversed_call_result_1399 = invoke(stypy.reporting.localization.Localization(__file__, 602, 17), reversed_1396, *[bodies_1397], **kwargs_1398)
        
        # Assigning a type to the variable 'reversed_call_result_1399' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'reversed_call_result_1399', reversed_call_result_1399)
        # Testing if the for loop is going to be iterated (line 602)
        # Testing the type of a for loop iterable (line 602)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 602, 8), reversed_call_result_1399)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 602, 8), reversed_call_result_1399):
            # Getting the type of the for loop variable (line 602)
            for_loop_var_1400 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 602, 8), reversed_call_result_1399)
            # Assigning a type to the variable 'b' (line 602)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'b', for_loop_var_1400)
            # SSA begins for a for statement (line 602)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 603):
            
            # Call to copy(...): (line 603)
            # Processing the call arguments (line 603)
            # Getting the type of 'b' (line 603)
            b_1402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 24), 'b', False)
            # Obtaining the member 'new_acc' of a type (line 603)
            new_acc_1403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 24), b_1402, 'new_acc')
            # Processing the call keyword arguments (line 603)
            kwargs_1404 = {}
            # Getting the type of 'copy' (line 603)
            copy_1401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 19), 'copy', False)
            # Calling copy(args, kwargs) (line 603)
            copy_call_result_1405 = invoke(stypy.reporting.localization.Localization(__file__, 603, 19), copy_1401, *[new_acc_1403], **kwargs_1404)
            
            # Assigning a type to the variable 'acc1' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'acc1', copy_call_result_1405)
            
            # Getting the type of 'nstep' (line 604)
            nstep_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 15), 'nstep')
            int_1407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 23), 'int')
            # Applying the binary operator '>' (line 604)
            result_gt_1408 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 15), '>', nstep_1406, int_1407)
            
            # Testing if the type of an if condition is none (line 604)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 604, 12), result_gt_1408):
                pass
            else:
                
                # Testing the type of an if condition (line 604)
                if_condition_1409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 12), result_gt_1408)
                # Assigning a type to the variable 'if_condition_1409' (line 604)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'if_condition_1409', if_condition_1409)
                # SSA begins for if statement (line 604)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to subtraction2(...): (line 605)
                # Processing the call arguments (line 605)
                # Getting the type of 'acc1' (line 605)
                acc1_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 34), 'acc1', False)
                # Getting the type of 'b' (line 605)
                b_1413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 40), 'b', False)
                # Obtaining the member 'acc' of a type (line 605)
                acc_1414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 40), b_1413, 'acc')
                # Processing the call keyword arguments (line 605)
                kwargs_1415 = {}
                # Getting the type of 'dacc' (line 605)
                dacc_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 16), 'dacc', False)
                # Obtaining the member 'subtraction2' of a type (line 605)
                subtraction2_1411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 16), dacc_1410, 'subtraction2')
                # Calling subtraction2(args, kwargs) (line 605)
                subtraction2_call_result_1416 = invoke(stypy.reporting.localization.Localization(__file__, 605, 16), subtraction2_1411, *[acc1_1412, acc_1414], **kwargs_1415)
                
                
                # Call to mult_scalar2(...): (line 606)
                # Processing the call arguments (line 606)
                # Getting the type of 'dacc' (line 606)
                dacc_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 34), 'dacc', False)
                # Getting the type of 'dthf' (line 606)
                dthf_1420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 40), 'dthf', False)
                # Processing the call keyword arguments (line 606)
                kwargs_1421 = {}
                # Getting the type of 'dvel' (line 606)
                dvel_1417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'dvel', False)
                # Obtaining the member 'mult_scalar2' of a type (line 606)
                mult_scalar2_1418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 16), dvel_1417, 'mult_scalar2')
                # Calling mult_scalar2(args, kwargs) (line 606)
                mult_scalar2_call_result_1422 = invoke(stypy.reporting.localization.Localization(__file__, 606, 16), mult_scalar2_1418, *[dacc_1419, dthf_1420], **kwargs_1421)
                
                
                # Getting the type of 'dvel' (line 607)
                dvel_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'dvel')
                # Getting the type of 'b' (line 607)
                b_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 24), 'b')
                # Obtaining the member 'vel' of a type (line 607)
                vel_1425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 24), b_1424, 'vel')
                # Applying the binary operator '+=' (line 607)
                result_iadd_1426 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 16), '+=', dvel_1423, vel_1425)
                # Assigning a type to the variable 'dvel' (line 607)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'dvel', result_iadd_1426)
                
                
                # Assigning a Call to a Attribute (line 608):
                
                # Call to copy(...): (line 608)
                # Processing the call arguments (line 608)
                # Getting the type of 'dvel' (line 608)
                dvel_1428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 29), 'dvel', False)
                # Processing the call keyword arguments (line 608)
                kwargs_1429 = {}
                # Getting the type of 'copy' (line 608)
                copy_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 24), 'copy', False)
                # Calling copy(args, kwargs) (line 608)
                copy_call_result_1430 = invoke(stypy.reporting.localization.Localization(__file__, 608, 24), copy_1427, *[dvel_1428], **kwargs_1429)
                
                # Getting the type of 'b' (line 608)
                b_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 16), 'b')
                # Setting the type of the member 'vel' of a type (line 608)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 16), b_1431, 'vel', copy_call_result_1430)
                # SSA join for if statement (line 604)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Attribute (line 610):
            
            # Call to copy(...): (line 610)
            # Processing the call arguments (line 610)
            # Getting the type of 'acc1' (line 610)
            acc1_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 25), 'acc1', False)
            # Processing the call keyword arguments (line 610)
            kwargs_1434 = {}
            # Getting the type of 'copy' (line 610)
            copy_1432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'copy', False)
            # Calling copy(args, kwargs) (line 610)
            copy_call_result_1435 = invoke(stypy.reporting.localization.Localization(__file__, 610, 20), copy_1432, *[acc1_1433], **kwargs_1434)
            
            # Getting the type of 'b' (line 610)
            b_1436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'b')
            # Setting the type of the member 'acc' of a type (line 610)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 12), b_1436, 'acc', copy_call_result_1435)
            
            # Call to mult_scalar2(...): (line 611)
            # Processing the call arguments (line 611)
            # Getting the type of 'b' (line 611)
            b_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 30), 'b', False)
            # Obtaining the member 'acc' of a type (line 611)
            acc_1440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 30), b_1439, 'acc')
            # Getting the type of 'dthf' (line 611)
            dthf_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 37), 'dthf', False)
            # Processing the call keyword arguments (line 611)
            kwargs_1442 = {}
            # Getting the type of 'dvel' (line 611)
            dvel_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'dvel', False)
            # Obtaining the member 'mult_scalar2' of a type (line 611)
            mult_scalar2_1438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 12), dvel_1437, 'mult_scalar2')
            # Calling mult_scalar2(args, kwargs) (line 611)
            mult_scalar2_call_result_1443 = invoke(stypy.reporting.localization.Localization(__file__, 611, 12), mult_scalar2_1438, *[acc_1440, dthf_1441], **kwargs_1442)
            
            
            # Assigning a Call to a Name (line 613):
            
            # Call to copy(...): (line 613)
            # Processing the call arguments (line 613)
            # Getting the type of 'b' (line 613)
            b_1445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 24), 'b', False)
            # Obtaining the member 'vel' of a type (line 613)
            vel_1446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 24), b_1445, 'vel')
            # Processing the call keyword arguments (line 613)
            kwargs_1447 = {}
            # Getting the type of 'copy' (line 613)
            copy_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 19), 'copy', False)
            # Calling copy(args, kwargs) (line 613)
            copy_call_result_1448 = invoke(stypy.reporting.localization.Localization(__file__, 613, 19), copy_1444, *[vel_1446], **kwargs_1447)
            
            # Assigning a type to the variable 'vel1' (line 613)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'vel1', copy_call_result_1448)
            
            # Getting the type of 'vel1' (line 614)
            vel1_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'vel1')
            # Getting the type of 'dvel' (line 614)
            dvel_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'dvel')
            # Applying the binary operator '+=' (line 614)
            result_iadd_1451 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 12), '+=', vel1_1449, dvel_1450)
            # Assigning a type to the variable 'vel1' (line 614)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'vel1', result_iadd_1451)
            
            
            # Assigning a Call to a Name (line 615):
            
            # Call to copy(...): (line 615)
            # Processing the call arguments (line 615)
            # Getting the type of 'vel1' (line 615)
            vel1_1453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 24), 'vel1', False)
            # Processing the call keyword arguments (line 615)
            kwargs_1454 = {}
            # Getting the type of 'copy' (line 615)
            copy_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 19), 'copy', False)
            # Calling copy(args, kwargs) (line 615)
            copy_call_result_1455 = invoke(stypy.reporting.localization.Localization(__file__, 615, 19), copy_1452, *[vel1_1453], **kwargs_1454)
            
            # Assigning a type to the variable 'dpos' (line 615)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'dpos', copy_call_result_1455)
            
            # Getting the type of 'dpos' (line 616)
            dpos_1456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'dpos')
            # Getting the type of 'BH' (line 616)
            BH_1457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'BH')
            # Obtaining the member 'DTIME' of a type (line 616)
            DTIME_1458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 20), BH_1457, 'DTIME')
            # Applying the binary operator '*=' (line 616)
            result_imul_1459 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 12), '*=', dpos_1456, DTIME_1458)
            # Assigning a type to the variable 'dpos' (line 616)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'dpos', result_imul_1459)
            
            
            # Getting the type of 'dpos' (line 617)
            dpos_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'dpos')
            # Getting the type of 'b' (line 617)
            b_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 'b')
            # Obtaining the member 'pos' of a type (line 617)
            pos_1462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 20), b_1461, 'pos')
            # Applying the binary operator '+=' (line 617)
            result_iadd_1463 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 12), '+=', dpos_1460, pos_1462)
            # Assigning a type to the variable 'dpos' (line 617)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'dpos', result_iadd_1463)
            
            
            # Assigning a Call to a Attribute (line 618):
            
            # Call to copy(...): (line 618)
            # Processing the call arguments (line 618)
            # Getting the type of 'dpos' (line 618)
            dpos_1465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 25), 'dpos', False)
            # Processing the call keyword arguments (line 618)
            kwargs_1466 = {}
            # Getting the type of 'copy' (line 618)
            copy_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 20), 'copy', False)
            # Calling copy(args, kwargs) (line 618)
            copy_call_result_1467 = invoke(stypy.reporting.localization.Localization(__file__, 618, 20), copy_1464, *[dpos_1465], **kwargs_1466)
            
            # Getting the type of 'b' (line 618)
            b_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'b')
            # Setting the type of the member 'pos' of a type (line 618)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 12), b_1468, 'pos', copy_call_result_1467)
            
            # Getting the type of 'vel1' (line 619)
            vel1_1469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'vel1')
            # Getting the type of 'dvel' (line 619)
            dvel_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'dvel')
            # Applying the binary operator '+=' (line 619)
            result_iadd_1471 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 12), '+=', vel1_1469, dvel_1470)
            # Assigning a type to the variable 'vel1' (line 619)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'vel1', result_iadd_1471)
            
            
            # Assigning a Call to a Attribute (line 620):
            
            # Call to copy(...): (line 620)
            # Processing the call arguments (line 620)
            # Getting the type of 'vel1' (line 620)
            vel1_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 25), 'vel1', False)
            # Processing the call keyword arguments (line 620)
            kwargs_1474 = {}
            # Getting the type of 'copy' (line 620)
            copy_1472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 20), 'copy', False)
            # Calling copy(args, kwargs) (line 620)
            copy_call_result_1475 = invoke(stypy.reporting.localization.Localization(__file__, 620, 20), copy_1472, *[vel1_1473], **kwargs_1474)
            
            # Getting the type of 'b' (line 620)
            b_1476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'b')
            # Setting the type of the member 'vel' of a type (line 620)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 12), b_1476, 'vel', copy_call_result_1475)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'vp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'vp' in the type store
        # Getting the type of 'stypy_return_type' (line 596)
        stypy_return_type_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1477)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'vp'
        return stypy_return_type_1477


# Assigning a type to the variable 'Tree' (line 468)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 0), 'Tree', Tree)
# Declaration of the 'BH' class

class BH(object, ):

    @staticmethod
    @norecursion
    def main(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'main'
        module_type_store = module_type_store.open_function_context('main', 639, 4, False)
        
        # Passed parameters checking function
        BH.main.__dict__.__setitem__('stypy_localization', localization)
        BH.main.__dict__.__setitem__('stypy_type_of_self', None)
        BH.main.__dict__.__setitem__('stypy_type_store', module_type_store)
        BH.main.__dict__.__setitem__('stypy_function_name', 'main')
        BH.main.__dict__.__setitem__('stypy_param_names_list', ['args'])
        BH.main.__dict__.__setitem__('stypy_varargs_param_name', None)
        BH.main.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BH.main.__dict__.__setitem__('stypy_call_defaults', defaults)
        BH.main.__dict__.__setitem__('stypy_call_varargs', varargs)
        BH.main.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BH.main.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'main', ['args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'main', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'main(...)' code ##################

        
        # Call to parse_cmd_line(...): (line 641)
        # Processing the call arguments (line 641)
        # Getting the type of 'args' (line 641)
        args_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 26), 'args', False)
        # Processing the call keyword arguments (line 641)
        kwargs_1481 = {}
        # Getting the type of 'BH' (line 641)
        BH_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'BH', False)
        # Obtaining the member 'parse_cmd_line' of a type (line 641)
        parse_cmd_line_1479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 8), BH_1478, 'parse_cmd_line')
        # Calling parse_cmd_line(args, kwargs) (line 641)
        parse_cmd_line_call_result_1482 = invoke(stypy.reporting.localization.Localization(__file__, 641, 8), parse_cmd_line_1479, *[args_1480], **kwargs_1481)
        
        # Getting the type of 'BH' (line 643)
        BH_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 11), 'BH')
        # Obtaining the member 'print_msgs' of a type (line 643)
        print_msgs_1484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 11), BH_1483, 'print_msgs')
        # Testing if the type of an if condition is none (line 643)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 643, 8), print_msgs_1484):
            pass
        else:
            
            # Testing the type of an if condition (line 643)
            if_condition_1485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 643, 8), print_msgs_1484)
            # Assigning a type to the variable 'if_condition_1485' (line 643)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'if_condition_1485', if_condition_1485)
            # SSA begins for if statement (line 643)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_1486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 18), 'str', 'nbody =')
            # Getting the type of 'BH' (line 644)
            BH_1487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 29), 'BH')
            # Obtaining the member 'nbody' of a type (line 644)
            nbody_1488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 29), BH_1487, 'nbody')
            # SSA join for if statement (line 643)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 646):
        
        # Call to clock(...): (line 646)
        # Processing the call keyword arguments (line 646)
        kwargs_1490 = {}
        # Getting the type of 'clock' (line 646)
        clock_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 17), 'clock', False)
        # Calling clock(args, kwargs) (line 646)
        clock_call_result_1491 = invoke(stypy.reporting.localization.Localization(__file__, 646, 17), clock_1489, *[], **kwargs_1490)
        
        # Assigning a type to the variable 'start0' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'start0', clock_call_result_1491)
        
        # Assigning a Call to a Name (line 647):
        
        # Call to Tree(...): (line 647)
        # Processing the call keyword arguments (line 647)
        kwargs_1493 = {}
        # Getting the type of 'Tree' (line 647)
        Tree_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'Tree', False)
        # Calling Tree(args, kwargs) (line 647)
        Tree_call_result_1494 = invoke(stypy.reporting.localization.Localization(__file__, 647, 15), Tree_1492, *[], **kwargs_1493)
        
        # Assigning a type to the variable 'root' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'root', Tree_call_result_1494)
        
        # Call to create_test_data(...): (line 648)
        # Processing the call arguments (line 648)
        # Getting the type of 'BH' (line 648)
        BH_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 30), 'BH', False)
        # Obtaining the member 'nbody' of a type (line 648)
        nbody_1498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 30), BH_1497, 'nbody')
        # Processing the call keyword arguments (line 648)
        kwargs_1499 = {}
        # Getting the type of 'root' (line 648)
        root_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'root', False)
        # Obtaining the member 'create_test_data' of a type (line 648)
        create_test_data_1496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 8), root_1495, 'create_test_data')
        # Calling create_test_data(args, kwargs) (line 648)
        create_test_data_call_result_1500 = invoke(stypy.reporting.localization.Localization(__file__, 648, 8), create_test_data_1496, *[nbody_1498], **kwargs_1499)
        
        
        # Assigning a Call to a Name (line 649):
        
        # Call to clock(...): (line 649)
        # Processing the call keyword arguments (line 649)
        kwargs_1502 = {}
        # Getting the type of 'clock' (line 649)
        clock_1501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 15), 'clock', False)
        # Calling clock(args, kwargs) (line 649)
        clock_call_result_1503 = invoke(stypy.reporting.localization.Localization(__file__, 649, 15), clock_1501, *[], **kwargs_1502)
        
        # Assigning a type to the variable 'end0' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'end0', clock_call_result_1503)
        # Getting the type of 'BH' (line 650)
        BH_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 11), 'BH')
        # Obtaining the member 'print_msgs' of a type (line 650)
        print_msgs_1505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 11), BH_1504, 'print_msgs')
        # Testing if the type of an if condition is none (line 650)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 650, 8), print_msgs_1505):
            pass
        else:
            
            # Testing the type of an if condition (line 650)
            if_condition_1506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 8), print_msgs_1505)
            # Assigning a type to the variable 'if_condition_1506' (line 650)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'if_condition_1506', if_condition_1506)
            # SSA begins for if statement (line 650)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_1507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 20), 'str', 'Bodies created')
            # SSA join for if statement (line 650)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 653):
        
        # Call to clock(...): (line 653)
        # Processing the call keyword arguments (line 653)
        kwargs_1509 = {}
        # Getting the type of 'clock' (line 653)
        clock_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 17), 'clock', False)
        # Calling clock(args, kwargs) (line 653)
        clock_call_result_1510 = invoke(stypy.reporting.localization.Localization(__file__, 653, 17), clock_1508, *[], **kwargs_1509)
        
        # Assigning a type to the variable 'start1' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'start1', clock_call_result_1510)
        
        # Assigning a Num to a Name (line 654):
        float_1511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 15), 'float')
        # Assigning a type to the variable 'tnow' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'tnow', float_1511)
        
        # Assigning a Num to a Name (line 655):
        int_1512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 12), 'int')
        # Assigning a type to the variable 'i' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'i', int_1512)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'tnow' (line 656)
        tnow_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 15), 'tnow')
        # Getting the type of 'BH' (line 656)
        BH_1514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 22), 'BH')
        # Obtaining the member 'TSTOP' of a type (line 656)
        TSTOP_1515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 22), BH_1514, 'TSTOP')
        float_1516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 33), 'float')
        # Getting the type of 'BH' (line 656)
        BH_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 39), 'BH')
        # Obtaining the member 'DTIME' of a type (line 656)
        DTIME_1518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 39), BH_1517, 'DTIME')
        # Applying the binary operator '*' (line 656)
        result_mul_1519 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 33), '*', float_1516, DTIME_1518)
        
        # Applying the binary operator '+' (line 656)
        result_add_1520 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 22), '+', TSTOP_1515, result_mul_1519)
        
        # Applying the binary operator '<' (line 656)
        result_lt_1521 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 15), '<', tnow_1513, result_add_1520)
        
        
        # Getting the type of 'i' (line 656)
        i_1522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 53), 'i')
        # Getting the type of 'BH' (line 656)
        BH_1523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 57), 'BH')
        # Obtaining the member 'nsteps' of a type (line 656)
        nsteps_1524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 57), BH_1523, 'nsteps')
        # Applying the binary operator '<' (line 656)
        result_lt_1525 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 53), '<', i_1522, nsteps_1524)
        
        # Applying the binary operator 'and' (line 656)
        result_and_keyword_1526 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 14), 'and', result_lt_1521, result_lt_1525)
        
        # Assigning a type to the variable 'result_and_keyword_1526' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'result_and_keyword_1526', result_and_keyword_1526)
        # Testing if the while is going to be iterated (line 656)
        # Testing the type of an if condition (line 656)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 8), result_and_keyword_1526)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 656, 8), result_and_keyword_1526):
            # SSA begins for while statement (line 656)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Call to step_system(...): (line 657)
            # Processing the call arguments (line 657)
            # Getting the type of 'i' (line 657)
            i_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 29), 'i', False)
            # Processing the call keyword arguments (line 657)
            kwargs_1530 = {}
            # Getting the type of 'root' (line 657)
            root_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'root', False)
            # Obtaining the member 'step_system' of a type (line 657)
            step_system_1528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 12), root_1527, 'step_system')
            # Calling step_system(args, kwargs) (line 657)
            step_system_call_result_1531 = invoke(stypy.reporting.localization.Localization(__file__, 657, 12), step_system_1528, *[i_1529], **kwargs_1530)
            
            
            # Getting the type of 'i' (line 658)
            i_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'i')
            int_1533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 17), 'int')
            # Applying the binary operator '+=' (line 658)
            result_iadd_1534 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 12), '+=', i_1532, int_1533)
            # Assigning a type to the variable 'i' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'i', result_iadd_1534)
            
            
            # Getting the type of 'tnow' (line 659)
            tnow_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'tnow')
            # Getting the type of 'BH' (line 659)
            BH_1536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 20), 'BH')
            # Obtaining the member 'DTIME' of a type (line 659)
            DTIME_1537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 20), BH_1536, 'DTIME')
            # Applying the binary operator '+=' (line 659)
            result_iadd_1538 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 12), '+=', tnow_1535, DTIME_1537)
            # Assigning a type to the variable 'tnow' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'tnow', result_iadd_1538)
            
            # SSA join for while statement (line 656)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 660):
        
        # Call to clock(...): (line 660)
        # Processing the call keyword arguments (line 660)
        kwargs_1540 = {}
        # Getting the type of 'clock' (line 660)
        clock_1539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 15), 'clock', False)
        # Calling clock(args, kwargs) (line 660)
        clock_call_result_1541 = invoke(stypy.reporting.localization.Localization(__file__, 660, 15), clock_1539, *[], **kwargs_1540)
        
        # Assigning a type to the variable 'end1' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'end1', clock_call_result_1541)
        # Getting the type of 'BH' (line 662)
        BH_1542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 11), 'BH')
        # Obtaining the member 'print_results' of a type (line 662)
        print_results_1543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 11), BH_1542, 'print_results')
        # Testing if the type of an if condition is none (line 662)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 662, 8), print_results_1543):
            pass
        else:
            
            # Testing the type of an if condition (line 662)
            if_condition_1544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 8), print_results_1543)
            # Assigning a type to the variable 'if_condition_1544' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'if_condition_1544', if_condition_1544)
            # SSA begins for if statement (line 662)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to enumerate(...): (line 663)
            # Processing the call arguments (line 663)
            # Getting the type of 'root' (line 663)
            root_1546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 34), 'root', False)
            # Obtaining the member 'bodies' of a type (line 663)
            bodies_1547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 34), root_1546, 'bodies')
            # Processing the call keyword arguments (line 663)
            kwargs_1548 = {}
            # Getting the type of 'enumerate' (line 663)
            enumerate_1545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 24), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 663)
            enumerate_call_result_1549 = invoke(stypy.reporting.localization.Localization(__file__, 663, 24), enumerate_1545, *[bodies_1547], **kwargs_1548)
            
            # Assigning a type to the variable 'enumerate_call_result_1549' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'enumerate_call_result_1549', enumerate_call_result_1549)
            # Testing if the for loop is going to be iterated (line 663)
            # Testing the type of a for loop iterable (line 663)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 663, 12), enumerate_call_result_1549)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 663, 12), enumerate_call_result_1549):
                # Getting the type of the for loop variable (line 663)
                for_loop_var_1550 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 663, 12), enumerate_call_result_1549)
                # Assigning a type to the variable 'j' (line 663)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 12), for_loop_var_1550, 2, 0))
                # Assigning a type to the variable 'b' (line 663)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 12), for_loop_var_1550, 2, 1))
                # SSA begins for a for statement (line 663)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                str_1551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 22), 'str', 'body %d: %s')
                
                # Obtaining an instance of the builtin type 'tuple' (line 664)
                tuple_1552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 39), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 664)
                # Adding element type (line 664)
                # Getting the type of 'j' (line 664)
                j_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 39), 'j')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 39), tuple_1552, j_1553)
                # Adding element type (line 664)
                # Getting the type of 'b' (line 664)
                b_1554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 42), 'b')
                # Obtaining the member 'pos' of a type (line 664)
                pos_1555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 42), b_1554, 'pos')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 39), tuple_1552, pos_1555)
                
                # Applying the binary operator '%' (line 664)
                result_mod_1556 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 22), '%', str_1551, tuple_1552)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 662)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'BH' (line 666)
        BH_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 11), 'BH')
        # Obtaining the member 'print_msgs' of a type (line 666)
        print_msgs_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 11), BH_1557, 'print_msgs')
        # Testing if the type of an if condition is none (line 666)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 666, 8), print_msgs_1558):
            pass
        else:
            
            # Testing the type of an if condition (line 666)
            if_condition_1559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 8), print_msgs_1558)
            # Assigning a type to the variable 'if_condition_1559' (line 666)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'if_condition_1559', if_condition_1559)
            # SSA begins for if statement (line 666)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_1560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 18), 'str', 'Build Time %.3f')
            # Getting the type of 'end0' (line 667)
            end0_1561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 39), 'end0')
            # Getting the type of 'start0' (line 667)
            start0_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 46), 'start0')
            # Applying the binary operator '-' (line 667)
            result_sub_1563 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 39), '-', end0_1561, start0_1562)
            
            # Applying the binary operator '%' (line 667)
            result_mod_1564 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 18), '%', str_1560, result_sub_1563)
            
            str_1565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 18), 'str', 'Compute Time %.3f')
            # Getting the type of 'end1' (line 668)
            end1_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 41), 'end1')
            # Getting the type of 'start1' (line 668)
            start1_1567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 48), 'start1')
            # Applying the binary operator '-' (line 668)
            result_sub_1568 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 41), '-', end1_1566, start1_1567)
            
            # Applying the binary operator '%' (line 668)
            result_mod_1569 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 18), '%', str_1565, result_sub_1568)
            
            str_1570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 18), 'str', 'Total Time %.3f')
            # Getting the type of 'end1' (line 669)
            end1_1571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 39), 'end1')
            # Getting the type of 'start0' (line 669)
            start0_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 46), 'start0')
            # Applying the binary operator '-' (line 669)
            result_sub_1573 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 39), '-', end1_1571, start0_1572)
            
            # Applying the binary operator '%' (line 669)
            result_mod_1574 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 18), '%', str_1570, result_sub_1573)
            
            # SSA join for if statement (line 666)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'main(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'main' in the type store
        # Getting the type of 'stypy_return_type' (line 639)
        stypy_return_type_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'main'
        return stypy_return_type_1575


    @staticmethod
    @norecursion
    def parse_cmd_line(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'parse_cmd_line'
        module_type_store = module_type_store.open_function_context('parse_cmd_line', 672, 4, False)
        
        # Passed parameters checking function
        BH.parse_cmd_line.__dict__.__setitem__('stypy_localization', localization)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_type_of_self', None)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_type_store', module_type_store)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_function_name', 'parse_cmd_line')
        BH.parse_cmd_line.__dict__.__setitem__('stypy_param_names_list', ['args'])
        BH.parse_cmd_line.__dict__.__setitem__('stypy_varargs_param_name', None)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_call_defaults', defaults)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_call_varargs', varargs)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BH.parse_cmd_line.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'parse_cmd_line', ['args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse_cmd_line', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse_cmd_line(...)' code ##################

        
        # Assigning a Num to a Name (line 674):
        int_1576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 12), 'int')
        # Assigning a type to the variable 'i' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'i', int_1576)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 675)
        i_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 14), 'i')
        
        # Call to len(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'args' (line 675)
        args_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 22), 'args', False)
        # Processing the call keyword arguments (line 675)
        kwargs_1580 = {}
        # Getting the type of 'len' (line 675)
        len_1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 18), 'len', False)
        # Calling len(args, kwargs) (line 675)
        len_call_result_1581 = invoke(stypy.reporting.localization.Localization(__file__, 675, 18), len_1578, *[args_1579], **kwargs_1580)
        
        # Applying the binary operator '<' (line 675)
        result_lt_1582 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 14), '<', i_1577, len_call_result_1581)
        
        
        # Call to startswith(...): (line 675)
        # Processing the call arguments (line 675)
        str_1588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 51), 'str', '-')
        # Processing the call keyword arguments (line 675)
        kwargs_1589 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 675)
        i_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 37), 'i', False)
        # Getting the type of 'args' (line 675)
        args_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 32), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 675)
        getitem___1585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 32), args_1584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 675)
        subscript_call_result_1586 = invoke(stypy.reporting.localization.Localization(__file__, 675, 32), getitem___1585, i_1583)
        
        # Obtaining the member 'startswith' of a type (line 675)
        startswith_1587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 32), subscript_call_result_1586, 'startswith')
        # Calling startswith(args, kwargs) (line 675)
        startswith_call_result_1590 = invoke(stypy.reporting.localization.Localization(__file__, 675, 32), startswith_1587, *[str_1588], **kwargs_1589)
        
        # Applying the binary operator 'and' (line 675)
        result_and_keyword_1591 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 14), 'and', result_lt_1582, startswith_call_result_1590)
        
        # Assigning a type to the variable 'result_and_keyword_1591' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'result_and_keyword_1591', result_and_keyword_1591)
        # Testing if the while is going to be iterated (line 675)
        # Testing the type of an if condition (line 675)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 8), result_and_keyword_1591)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 675, 8), result_and_keyword_1591):
            # SSA begins for while statement (line 675)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Subscript to a Name (line 676):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 676)
            i_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 23), 'i')
            # Getting the type of 'args' (line 676)
            args_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 18), 'args')
            # Obtaining the member '__getitem__' of a type (line 676)
            getitem___1594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 18), args_1593, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 676)
            subscript_call_result_1595 = invoke(stypy.reporting.localization.Localization(__file__, 676, 18), getitem___1594, i_1592)
            
            # Assigning a type to the variable 'arg' (line 676)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'arg', subscript_call_result_1595)
            
            # Getting the type of 'i' (line 677)
            i_1596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 12), 'i')
            int_1597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 17), 'int')
            # Applying the binary operator '+=' (line 677)
            result_iadd_1598 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 12), '+=', i_1596, int_1597)
            # Assigning a type to the variable 'i' (line 677)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 12), 'i', result_iadd_1598)
            
            
            # Getting the type of 'arg' (line 680)
            arg_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 15), 'arg')
            str_1600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 22), 'str', '-b')
            # Applying the binary operator '==' (line 680)
            result_eq_1601 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 15), '==', arg_1599, str_1600)
            
            # Testing if the type of an if condition is none (line 680)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 680, 12), result_eq_1601):
                
                # Getting the type of 'arg' (line 686)
                arg_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 17), 'arg')
                str_1626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 24), 'str', '-s')
                # Applying the binary operator '==' (line 686)
                result_eq_1627 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 17), '==', arg_1625, str_1626)
                
                # Testing if the type of an if condition is none (line 686)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 686, 17), result_eq_1627):
                    
                    # Getting the type of 'arg' (line 692)
                    arg_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'arg')
                    str_1652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 24), 'str', '-m')
                    # Applying the binary operator '==' (line 692)
                    result_eq_1653 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 17), '==', arg_1651, str_1652)
                    
                    # Testing if the type of an if condition is none (line 692)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653):
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 692)
                        if_condition_1654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653)
                        # Assigning a type to the variable 'if_condition_1654' (line 692)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'if_condition_1654', if_condition_1654)
                        # SSA begins for if statement (line 692)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 693):
                        # Getting the type of 'True' (line 693)
                        True_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 32), 'True')
                        # Getting the type of 'BH' (line 693)
                        BH_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'BH')
                        # Setting the type of the member 'print_msgs' of a type (line 693)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), BH_1656, 'print_msgs', True_1655)
                        # SSA branch for the else part of an if statement (line 692)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 692)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 686)
                    if_condition_1628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 686, 17), result_eq_1627)
                    # Assigning a type to the variable 'if_condition_1628' (line 686)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 17), 'if_condition_1628', if_condition_1628)
                    # SSA begins for if statement (line 686)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'i' (line 687)
                    i_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 19), 'i')
                    
                    # Call to len(...): (line 687)
                    # Processing the call arguments (line 687)
                    # Getting the type of 'args' (line 687)
                    args_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 27), 'args', False)
                    # Processing the call keyword arguments (line 687)
                    kwargs_1632 = {}
                    # Getting the type of 'len' (line 687)
                    len_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 23), 'len', False)
                    # Calling len(args, kwargs) (line 687)
                    len_call_result_1633 = invoke(stypy.reporting.localization.Localization(__file__, 687, 23), len_1630, *[args_1631], **kwargs_1632)
                    
                    # Applying the binary operator '<' (line 687)
                    result_lt_1634 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 19), '<', i_1629, len_call_result_1633)
                    
                    # Testing if the type of an if condition is none (line 687)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 687, 16), result_lt_1634):
                        
                        # Call to Exception(...): (line 691)
                        # Processing the call arguments (line 691)
                        str_1648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 36), 'str', '-l requires the number of levels')
                        # Processing the call keyword arguments (line 691)
                        kwargs_1649 = {}
                        # Getting the type of 'Exception' (line 691)
                        Exception_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 26), 'Exception', False)
                        # Calling Exception(args, kwargs) (line 691)
                        Exception_call_result_1650 = invoke(stypy.reporting.localization.Localization(__file__, 691, 26), Exception_1647, *[str_1648], **kwargs_1649)
                        
                        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 691, 20), Exception_call_result_1650, 'raise parameter', BaseException)
                    else:
                        
                        # Testing the type of an if condition (line 687)
                        if_condition_1635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 687, 16), result_lt_1634)
                        # Assigning a type to the variable 'if_condition_1635' (line 687)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 16), 'if_condition_1635', if_condition_1635)
                        # SSA begins for if statement (line 687)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Attribute (line 688):
                        
                        # Call to int(...): (line 688)
                        # Processing the call arguments (line 688)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 688)
                        i_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 41), 'i', False)
                        # Getting the type of 'args' (line 688)
                        args_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 36), 'args', False)
                        # Obtaining the member '__getitem__' of a type (line 688)
                        getitem___1639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 36), args_1638, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 688)
                        subscript_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 688, 36), getitem___1639, i_1637)
                        
                        # Processing the call keyword arguments (line 688)
                        kwargs_1641 = {}
                        # Getting the type of 'int' (line 688)
                        int_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 32), 'int', False)
                        # Calling int(args, kwargs) (line 688)
                        int_call_result_1642 = invoke(stypy.reporting.localization.Localization(__file__, 688, 32), int_1636, *[subscript_call_result_1640], **kwargs_1641)
                        
                        # Getting the type of 'BH' (line 688)
                        BH_1643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 20), 'BH')
                        # Setting the type of the member 'nsteps' of a type (line 688)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 20), BH_1643, 'nsteps', int_call_result_1642)
                        
                        # Getting the type of 'i' (line 689)
                        i_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 20), 'i')
                        int_1645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 25), 'int')
                        # Applying the binary operator '+=' (line 689)
                        result_iadd_1646 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 20), '+=', i_1644, int_1645)
                        # Assigning a type to the variable 'i' (line 689)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 20), 'i', result_iadd_1646)
                        
                        # SSA branch for the else part of an if statement (line 687)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to Exception(...): (line 691)
                        # Processing the call arguments (line 691)
                        str_1648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 36), 'str', '-l requires the number of levels')
                        # Processing the call keyword arguments (line 691)
                        kwargs_1649 = {}
                        # Getting the type of 'Exception' (line 691)
                        Exception_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 26), 'Exception', False)
                        # Calling Exception(args, kwargs) (line 691)
                        Exception_call_result_1650 = invoke(stypy.reporting.localization.Localization(__file__, 691, 26), Exception_1647, *[str_1648], **kwargs_1649)
                        
                        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 691, 20), Exception_call_result_1650, 'raise parameter', BaseException)
                        # SSA join for if statement (line 687)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 686)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'arg' (line 692)
                    arg_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'arg')
                    str_1652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 24), 'str', '-m')
                    # Applying the binary operator '==' (line 692)
                    result_eq_1653 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 17), '==', arg_1651, str_1652)
                    
                    # Testing if the type of an if condition is none (line 692)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653):
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 692)
                        if_condition_1654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653)
                        # Assigning a type to the variable 'if_condition_1654' (line 692)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'if_condition_1654', if_condition_1654)
                        # SSA begins for if statement (line 692)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 693):
                        # Getting the type of 'True' (line 693)
                        True_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 32), 'True')
                        # Getting the type of 'BH' (line 693)
                        BH_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'BH')
                        # Setting the type of the member 'print_msgs' of a type (line 693)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), BH_1656, 'print_msgs', True_1655)
                        # SSA branch for the else part of an if statement (line 692)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 692)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 686)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 680)
                if_condition_1602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 680, 12), result_eq_1601)
                # Assigning a type to the variable 'if_condition_1602' (line 680)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'if_condition_1602', if_condition_1602)
                # SSA begins for if statement (line 680)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'i' (line 681)
                i_1603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 19), 'i')
                
                # Call to len(...): (line 681)
                # Processing the call arguments (line 681)
                # Getting the type of 'args' (line 681)
                args_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 27), 'args', False)
                # Processing the call keyword arguments (line 681)
                kwargs_1606 = {}
                # Getting the type of 'len' (line 681)
                len_1604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 23), 'len', False)
                # Calling len(args, kwargs) (line 681)
                len_call_result_1607 = invoke(stypy.reporting.localization.Localization(__file__, 681, 23), len_1604, *[args_1605], **kwargs_1606)
                
                # Applying the binary operator '<' (line 681)
                result_lt_1608 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 19), '<', i_1603, len_call_result_1607)
                
                # Testing if the type of an if condition is none (line 681)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 681, 16), result_lt_1608):
                    
                    # Call to Exception(...): (line 685)
                    # Processing the call arguments (line 685)
                    str_1622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 36), 'str', '-l requires the number of levels')
                    # Processing the call keyword arguments (line 685)
                    kwargs_1623 = {}
                    # Getting the type of 'Exception' (line 685)
                    Exception_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 26), 'Exception', False)
                    # Calling Exception(args, kwargs) (line 685)
                    Exception_call_result_1624 = invoke(stypy.reporting.localization.Localization(__file__, 685, 26), Exception_1621, *[str_1622], **kwargs_1623)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 685, 20), Exception_call_result_1624, 'raise parameter', BaseException)
                else:
                    
                    # Testing the type of an if condition (line 681)
                    if_condition_1609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 681, 16), result_lt_1608)
                    # Assigning a type to the variable 'if_condition_1609' (line 681)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'if_condition_1609', if_condition_1609)
                    # SSA begins for if statement (line 681)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Attribute (line 682):
                    
                    # Call to int(...): (line 682)
                    # Processing the call arguments (line 682)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 682)
                    i_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 40), 'i', False)
                    # Getting the type of 'args' (line 682)
                    args_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 35), 'args', False)
                    # Obtaining the member '__getitem__' of a type (line 682)
                    getitem___1613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 35), args_1612, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 682)
                    subscript_call_result_1614 = invoke(stypy.reporting.localization.Localization(__file__, 682, 35), getitem___1613, i_1611)
                    
                    # Processing the call keyword arguments (line 682)
                    kwargs_1615 = {}
                    # Getting the type of 'int' (line 682)
                    int_1610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 31), 'int', False)
                    # Calling int(args, kwargs) (line 682)
                    int_call_result_1616 = invoke(stypy.reporting.localization.Localization(__file__, 682, 31), int_1610, *[subscript_call_result_1614], **kwargs_1615)
                    
                    # Getting the type of 'BH' (line 682)
                    BH_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 20), 'BH')
                    # Setting the type of the member 'nbody' of a type (line 682)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 20), BH_1617, 'nbody', int_call_result_1616)
                    
                    # Getting the type of 'i' (line 683)
                    i_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'i')
                    int_1619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 25), 'int')
                    # Applying the binary operator '+=' (line 683)
                    result_iadd_1620 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 20), '+=', i_1618, int_1619)
                    # Assigning a type to the variable 'i' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'i', result_iadd_1620)
                    
                    # SSA branch for the else part of an if statement (line 681)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to Exception(...): (line 685)
                    # Processing the call arguments (line 685)
                    str_1622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 36), 'str', '-l requires the number of levels')
                    # Processing the call keyword arguments (line 685)
                    kwargs_1623 = {}
                    # Getting the type of 'Exception' (line 685)
                    Exception_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 26), 'Exception', False)
                    # Calling Exception(args, kwargs) (line 685)
                    Exception_call_result_1624 = invoke(stypy.reporting.localization.Localization(__file__, 685, 26), Exception_1621, *[str_1622], **kwargs_1623)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 685, 20), Exception_call_result_1624, 'raise parameter', BaseException)
                    # SSA join for if statement (line 681)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 680)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'arg' (line 686)
                arg_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 17), 'arg')
                str_1626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 24), 'str', '-s')
                # Applying the binary operator '==' (line 686)
                result_eq_1627 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 17), '==', arg_1625, str_1626)
                
                # Testing if the type of an if condition is none (line 686)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 686, 17), result_eq_1627):
                    
                    # Getting the type of 'arg' (line 692)
                    arg_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'arg')
                    str_1652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 24), 'str', '-m')
                    # Applying the binary operator '==' (line 692)
                    result_eq_1653 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 17), '==', arg_1651, str_1652)
                    
                    # Testing if the type of an if condition is none (line 692)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653):
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 692)
                        if_condition_1654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653)
                        # Assigning a type to the variable 'if_condition_1654' (line 692)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'if_condition_1654', if_condition_1654)
                        # SSA begins for if statement (line 692)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 693):
                        # Getting the type of 'True' (line 693)
                        True_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 32), 'True')
                        # Getting the type of 'BH' (line 693)
                        BH_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'BH')
                        # Setting the type of the member 'print_msgs' of a type (line 693)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), BH_1656, 'print_msgs', True_1655)
                        # SSA branch for the else part of an if statement (line 692)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 692)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 686)
                    if_condition_1628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 686, 17), result_eq_1627)
                    # Assigning a type to the variable 'if_condition_1628' (line 686)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 17), 'if_condition_1628', if_condition_1628)
                    # SSA begins for if statement (line 686)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'i' (line 687)
                    i_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 19), 'i')
                    
                    # Call to len(...): (line 687)
                    # Processing the call arguments (line 687)
                    # Getting the type of 'args' (line 687)
                    args_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 27), 'args', False)
                    # Processing the call keyword arguments (line 687)
                    kwargs_1632 = {}
                    # Getting the type of 'len' (line 687)
                    len_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 23), 'len', False)
                    # Calling len(args, kwargs) (line 687)
                    len_call_result_1633 = invoke(stypy.reporting.localization.Localization(__file__, 687, 23), len_1630, *[args_1631], **kwargs_1632)
                    
                    # Applying the binary operator '<' (line 687)
                    result_lt_1634 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 19), '<', i_1629, len_call_result_1633)
                    
                    # Testing if the type of an if condition is none (line 687)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 687, 16), result_lt_1634):
                        
                        # Call to Exception(...): (line 691)
                        # Processing the call arguments (line 691)
                        str_1648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 36), 'str', '-l requires the number of levels')
                        # Processing the call keyword arguments (line 691)
                        kwargs_1649 = {}
                        # Getting the type of 'Exception' (line 691)
                        Exception_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 26), 'Exception', False)
                        # Calling Exception(args, kwargs) (line 691)
                        Exception_call_result_1650 = invoke(stypy.reporting.localization.Localization(__file__, 691, 26), Exception_1647, *[str_1648], **kwargs_1649)
                        
                        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 691, 20), Exception_call_result_1650, 'raise parameter', BaseException)
                    else:
                        
                        # Testing the type of an if condition (line 687)
                        if_condition_1635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 687, 16), result_lt_1634)
                        # Assigning a type to the variable 'if_condition_1635' (line 687)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 16), 'if_condition_1635', if_condition_1635)
                        # SSA begins for if statement (line 687)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Attribute (line 688):
                        
                        # Call to int(...): (line 688)
                        # Processing the call arguments (line 688)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 688)
                        i_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 41), 'i', False)
                        # Getting the type of 'args' (line 688)
                        args_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 36), 'args', False)
                        # Obtaining the member '__getitem__' of a type (line 688)
                        getitem___1639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 36), args_1638, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 688)
                        subscript_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 688, 36), getitem___1639, i_1637)
                        
                        # Processing the call keyword arguments (line 688)
                        kwargs_1641 = {}
                        # Getting the type of 'int' (line 688)
                        int_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 32), 'int', False)
                        # Calling int(args, kwargs) (line 688)
                        int_call_result_1642 = invoke(stypy.reporting.localization.Localization(__file__, 688, 32), int_1636, *[subscript_call_result_1640], **kwargs_1641)
                        
                        # Getting the type of 'BH' (line 688)
                        BH_1643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 20), 'BH')
                        # Setting the type of the member 'nsteps' of a type (line 688)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 20), BH_1643, 'nsteps', int_call_result_1642)
                        
                        # Getting the type of 'i' (line 689)
                        i_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 20), 'i')
                        int_1645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 25), 'int')
                        # Applying the binary operator '+=' (line 689)
                        result_iadd_1646 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 20), '+=', i_1644, int_1645)
                        # Assigning a type to the variable 'i' (line 689)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 20), 'i', result_iadd_1646)
                        
                        # SSA branch for the else part of an if statement (line 687)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to Exception(...): (line 691)
                        # Processing the call arguments (line 691)
                        str_1648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 36), 'str', '-l requires the number of levels')
                        # Processing the call keyword arguments (line 691)
                        kwargs_1649 = {}
                        # Getting the type of 'Exception' (line 691)
                        Exception_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 26), 'Exception', False)
                        # Calling Exception(args, kwargs) (line 691)
                        Exception_call_result_1650 = invoke(stypy.reporting.localization.Localization(__file__, 691, 26), Exception_1647, *[str_1648], **kwargs_1649)
                        
                        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 691, 20), Exception_call_result_1650, 'raise parameter', BaseException)
                        # SSA join for if statement (line 687)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 686)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'arg' (line 692)
                    arg_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'arg')
                    str_1652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 24), 'str', '-m')
                    # Applying the binary operator '==' (line 692)
                    result_eq_1653 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 17), '==', arg_1651, str_1652)
                    
                    # Testing if the type of an if condition is none (line 692)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653):
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 692)
                        if_condition_1654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 17), result_eq_1653)
                        # Assigning a type to the variable 'if_condition_1654' (line 692)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'if_condition_1654', if_condition_1654)
                        # SSA begins for if statement (line 692)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 693):
                        # Getting the type of 'True' (line 693)
                        True_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 32), 'True')
                        # Getting the type of 'BH' (line 693)
                        BH_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'BH')
                        # Setting the type of the member 'print_msgs' of a type (line 693)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), BH_1656, 'print_msgs', True_1655)
                        # SSA branch for the else part of an if statement (line 692)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'arg' (line 694)
                        arg_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'arg')
                        str_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', '-p')
                        # Applying the binary operator '==' (line 694)
                        result_eq_1659 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 17), '==', arg_1657, str_1658)
                        
                        # Testing if the type of an if condition is none (line 694)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659):
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 694)
                            if_condition_1660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 17), result_eq_1659)
                            # Assigning a type to the variable 'if_condition_1660' (line 694)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'if_condition_1660', if_condition_1660)
                            # SSA begins for if statement (line 694)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 695):
                            # Getting the type of 'True' (line 695)
                            True_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'True')
                            # Getting the type of 'BH' (line 695)
                            BH_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'BH')
                            # Setting the type of the member 'print_results' of a type (line 695)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), BH_1662, 'print_results', True_1661)
                            # SSA branch for the else part of an if statement (line 694)
                            module_type_store.open_ssa_branch('else')
                            
                            # Getting the type of 'arg' (line 696)
                            arg_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'arg')
                            str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', '-h')
                            # Applying the binary operator '==' (line 696)
                            result_eq_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 17), '==', arg_1663, str_1664)
                            
                            # Testing if the type of an if condition is none (line 696)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 696)
                                if_condition_1666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 17), result_eq_1665)
                                # Assigning a type to the variable 'if_condition_1666' (line 696)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'if_condition_1666', if_condition_1666)
                                # SSA begins for if statement (line 696)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to usage(...): (line 697)
                                # Processing the call keyword arguments (line 697)
                                kwargs_1669 = {}
                                # Getting the type of 'BH' (line 697)
                                BH_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'BH', False)
                                # Obtaining the member 'usage' of a type (line 697)
                                usage_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), BH_1667, 'usage')
                                # Calling usage(args, kwargs) (line 697)
                                usage_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), usage_1668, *[], **kwargs_1669)
                                
                                # SSA join for if statement (line 696)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 694)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 692)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 686)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 680)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for while statement (line 675)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'BH' (line 699)
        BH_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 11), 'BH')
        # Obtaining the member 'nbody' of a type (line 699)
        nbody_1672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 11), BH_1671, 'nbody')
        int_1673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 23), 'int')
        # Applying the binary operator '==' (line 699)
        result_eq_1674 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 11), '==', nbody_1672, int_1673)
        
        # Testing if the type of an if condition is none (line 699)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 699, 8), result_eq_1674):
            pass
        else:
            
            # Testing the type of an if condition (line 699)
            if_condition_1675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 699, 8), result_eq_1674)
            # Assigning a type to the variable 'if_condition_1675' (line 699)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'if_condition_1675', if_condition_1675)
            # SSA begins for if statement (line 699)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to usage(...): (line 700)
            # Processing the call keyword arguments (line 700)
            kwargs_1678 = {}
            # Getting the type of 'BH' (line 700)
            BH_1676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'BH', False)
            # Obtaining the member 'usage' of a type (line 700)
            usage_1677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 12), BH_1676, 'usage')
            # Calling usage(args, kwargs) (line 700)
            usage_call_result_1679 = invoke(stypy.reporting.localization.Localization(__file__, 700, 12), usage_1677, *[], **kwargs_1678)
            
            # SSA join for if statement (line 699)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'parse_cmd_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse_cmd_line' in the type store
        # Getting the type of 'stypy_return_type' (line 672)
        stypy_return_type_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1680)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse_cmd_line'
        return stypy_return_type_1680


    @staticmethod
    @norecursion
    def usage(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'usage'
        module_type_store = module_type_store.open_function_context('usage', 702, 4, False)
        
        # Passed parameters checking function
        BH.usage.__dict__.__setitem__('stypy_localization', localization)
        BH.usage.__dict__.__setitem__('stypy_type_of_self', None)
        BH.usage.__dict__.__setitem__('stypy_type_store', module_type_store)
        BH.usage.__dict__.__setitem__('stypy_function_name', 'usage')
        BH.usage.__dict__.__setitem__('stypy_param_names_list', [])
        BH.usage.__dict__.__setitem__('stypy_varargs_param_name', None)
        BH.usage.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BH.usage.__dict__.__setitem__('stypy_call_defaults', defaults)
        BH.usage.__dict__.__setitem__('stypy_call_varargs', varargs)
        BH.usage.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BH.usage.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'usage', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'usage', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'usage(...)' code ##################

        str_1681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 8), 'str', 'The usage routine which describes the program options.')
        str_1682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 24), 'str', 'usage: python bh.py -b <size> [-s <steps>] [-p] [-m] [-h]')
        str_1683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 24), 'str', '  -b the number of bodies')
        str_1684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 24), 'str', '  -s the max. number of time steps (default=10)')
        str_1685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 24), 'str', '  -p (print detailed results)')
        str_1686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 24), 'str', '  -m (print information messages')
        str_1687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 24), 'str', '  -h (self message)')
        
        # Call to SystemExit(...): (line 711)
        # Processing the call keyword arguments (line 711)
        kwargs_1689 = {}
        # Getting the type of 'SystemExit' (line 711)
        SystemExit_1688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 14), 'SystemExit', False)
        # Calling SystemExit(args, kwargs) (line 711)
        SystemExit_call_result_1690 = invoke(stypy.reporting.localization.Localization(__file__, 711, 14), SystemExit_1688, *[], **kwargs_1689)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 711, 8), SystemExit_call_result_1690, 'raise parameter', BaseException)
        
        # ################# End of 'usage(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'usage' in the type store
        # Getting the type of 'stypy_return_type' (line 702)
        stypy_return_type_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'usage'
        return stypy_return_type_1691


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 623, 0, False)
        # Assigning a type to the variable 'self' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BH.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BH' (line 623)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 0), 'BH', BH)

# Assigning a Num to a Name (line 624):
float_1692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 12), 'float')
# Getting the type of 'BH'
BH_1693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BH')
# Setting the type of the member 'DTIME' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BH_1693, 'DTIME', float_1692)

# Assigning a Num to a Name (line 625):
float_1694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 12), 'float')
# Getting the type of 'BH'
BH_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BH')
# Setting the type of the member 'TSTOP' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BH_1695, 'TSTOP', float_1694)

# Assigning a Num to a Name (line 628):
int_1696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 12), 'int')
# Getting the type of 'BH'
BH_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BH')
# Setting the type of the member 'nbody' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BH_1697, 'nbody', int_1696)

# Assigning a Num to a Name (line 631):
int_1698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 13), 'int')
# Getting the type of 'BH'
BH_1699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BH')
# Setting the type of the member 'nsteps' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BH_1699, 'nsteps', int_1698)

# Assigning a Name to a Name (line 634):
# Getting the type of 'False' (line 634)
False_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 17), 'False')
# Getting the type of 'BH'
BH_1701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BH')
# Setting the type of the member 'print_msgs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BH_1701, 'print_msgs', False_1700)

# Assigning a Name to a Name (line 637):
# Getting the type of 'False' (line 637)
False_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'False')
# Getting the type of 'BH'
BH_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BH')
# Setting the type of the member 'print_results' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BH_1703, 'print_results', False_1702)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 714, 0, False)
    
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

    
    # Assigning a List to a Name (line 715):
    
    # Obtaining an instance of the builtin type 'list' (line 715)
    list_1704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 715)
    # Adding element type (line 715)
    str_1705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 12), 'str', 'bh.py')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 11), list_1704, str_1705)
    # Adding element type (line 715)
    str_1706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 21), 'str', '-b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 11), list_1704, str_1706)
    # Adding element type (line 715)
    str_1707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 27), 'str', '500')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 11), list_1704, str_1707)
    
    # Assigning a type to the variable 'args' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'args', list_1704)
    
    # Call to main(...): (line 716)
    # Processing the call arguments (line 716)
    # Getting the type of 'args' (line 716)
    args_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'args', False)
    # Processing the call keyword arguments (line 716)
    kwargs_1711 = {}
    # Getting the type of 'BH' (line 716)
    BH_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 4), 'BH', False)
    # Obtaining the member 'main' of a type (line 716)
    main_1709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 4), BH_1708, 'main')
    # Calling main(args, kwargs) (line 716)
    main_call_result_1712 = invoke(stypy.reporting.localization.Localization(__file__, 716, 4), main_1709, *[args_1710], **kwargs_1711)
    
    # Getting the type of 'True' (line 717)
    True_1713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'stypy_return_type', True_1713)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 714)
    stypy_return_type_1714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1714)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1714

# Assigning a type to the variable 'run' (line 714)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 0), 'run', run)

# Call to run(...): (line 719)
# Processing the call keyword arguments (line 719)
kwargs_1716 = {}
# Getting the type of 'run' (line 719)
run_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 0), 'run', False)
# Calling run(args, kwargs) (line 719)
run_call_result_1717 = invoke(stypy.reporting.localization.Localization(__file__, 719, 0), run_1715, *[], **kwargs_1716)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

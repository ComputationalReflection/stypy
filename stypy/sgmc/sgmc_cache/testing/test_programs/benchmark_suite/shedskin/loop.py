
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Copyright 2011 Google Inc.
3: 
4: Licensed under the Apache License, Version 2.0 (the "License")
5: you may not use this file except in compliance with the License.
6: You may obtain a copy of the License at
7: 
8:     http:#www.apache.org/licenses/LICENSE-2.0
9: 
10: Unless required by applicable law or agreed to in writing, software
11: distributed under the License is distributed on an "AS IS" BASIS,
12: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
13: See the License for the specific language governing permissions and
14: limitations under the License.
15: 
16: This version is designed for ShedSkin
17: D code translated to Python by leonardo maffi, v.1.0, Jun 14 2011
18: '''
19: 
20: import sys
21: 
22: sys.setrecursionlimit(100000)
23: from sys import stdout
24: 
25: 
26: class Simple_loop(object):
27:     '''
28:     Basic representation of loops, a loop has an entry point,
29:     one or more exit edges, a set of basic blocks, and potentially
30:     an outer loop - a "parent" loop.
31: 
32:     Furthermore, it can have any set of properties, e.g.,
33:     it can be an irreducible loop, have control flow, be
34:     a candidate for transformations, and what not.
35:     '''
36: 
37:     def __init__(self):
38:         self.basic_blocks_ = set()
39:         self.children_ = set()
40:         self.parent_ = None
41:         self.is_root_ = False
42:         self.counter_ = 0
43:         self.nesting_level_ = 0
44:         self.depth_level_ = 0
45: 
46:     def add_node(self, basic_block):
47:         self.basic_blocks_.add(basic_block)
48: 
49:     def add_child_loop(self, loop):
50:         self.children_.add(loop)
51: 
52:     def dump(self):
53:         pass
54:         # Simplified for readability purposes.
55:         # print "loop-%d, nest: %d, depth: %d" % (self.counter_, self.nesting_level_, self.depth_level_)
56: 
57:     # Getters/Setters
58:     def set_parent(self, parent):
59:         self.parent_ = parent
60:         parent.add_child_loop(self)
61: 
62:     def set_nesting_level(self, level):
63:         self.nesting_level_ = level
64:         if level == 0:
65:             self.is_root_ = True
66: 
67: 
68: class Loop_structure_graph(object):
69:     '''
70:     Maintain loop structure for a given cfg.
71: 
72:     Two values are maintained for this loop graph, depth, and nesting level.
73:     For example:
74: 
75:     loop        nesting level    depth
76:     ---------------------------------------
77:     loop-0      2                0
78:       loop-1    1                1
79:       loop-3    1                1
80:         loop-2  0                2
81:     '''
82: 
83:     def __init__(self):
84:         self.loops_ = []
85:         self.loop_counter_ = 0
86:         self.root_ = Simple_loop()
87:         self.root_.set_nesting_level(0)  # make it the root node
88:         self.root_.counter_ = self.loop_counter_
89:         self.loop_counter_ += 1
90:         self.loops_.append(self.root_)
91: 
92:     def create_new_loop(self):
93:         loop = Simple_loop()
94:         loop.counter_ = self.loop_counter_
95:         self.loop_counter_ += 1
96:         return loop
97: 
98:     def dump(self):
99:         self.dump_rec(self.root_, 0)
100: 
101:     def dump_rec(self, loop, indent):
102:         # Simplified for readability purposes.
103:         loop.dump()
104: 
105:         for liter in loop.children_:
106:             pass  # self.dump_rec(liter, indent + 1)
107: 
108:     def calculate_nesting_level(self):
109:         # link up all 1st level loops to artificial root node.
110:         for loop in self.loops_:
111:             if loop.is_root_:
112:                 continue
113:             if not loop.parent_:
114:                 loop.set_parent(self.root_)
115: 
116:         # recursively traverse the tree and assign levels.
117:         self.calculate_nesting_level_rec(self.root_, 0)
118: 
119:     def calculate_nesting_level_rec(self, loop, depth):
120:         loop.depth_level_ = depth
121:         for ch in loop.children_:
122:             calculate_nesting_level_rec(ch, depth + 1)
123:             loop.nesting_level_ = max(loop.nesting_level_, 1 + ch.nesting_level_)
124: 
125: 
126: # ======================================================
127: # Main Algorithm
128: # ======================================================
129: 
130: 
131: class Union_find_node(object):  # add __slots__ *******************************************
132:     '''
133:     Union/Find algorithm after Tarjan, R.E., 1983, Data Structures
134:     and Network Algorithms.
135:     '''
136: 
137:     def init(self, bb, dfs_number):
138:         self.parent_ = self
139:         self.bb_ = bb
140:         self.dfs_number_ = dfs_number
141:         self.loop_ = None
142: 
143:     def find_set(self):
144:         '''
145:         Union/Find Algorithm - The find routine.
146: 
147:         Implemented with Path Compression (inner loops are only
148:         visited and collapsed once, however, deep nests would still
149:         result in significant traversals).
150:         '''
151:         nodeList = []
152: 
153:         node = self
154:         while node != node.parent_:
155:             if node.parent_ != node.parent_.parent_:
156:                 nodeList.append(node)
157:             node = node.parent_
158: 
159:         # Path Compression, all nodes' parents point to the 1st level parent.
160:         for n in nodeList:
161:             n.parent_ = node.parent_
162: 
163:         return node
164: 
165:     # / Union/Find Algorithm - The Union routine. We rely on path compression.
166:     def do_union(self, B):
167:         self.parent_ = B
168: 
169: 
170: class Basic_block_class(object):
171:     TOP = 0  # uninitialized
172:     NONHEADER = 1  # a regular BB
173:     REDUCIBLE = 2  # reducible loop
174:     SELF = 3  # single BB loop
175:     IRREDUCIBLE = 4  # irreducible loop
176:     DEAD = 5  # a dead BB
177:     LAST = 6  # Sentinel
178: 
179: 
180: class Havlak_loop_finder(object):
181:     '''
182:     Loop Recognition
183: 
184:     based on:
185:       Paul Havlak, Nesting of Reducible and Irreducible Loops,
186:          Rice University.
187: 
188:       We adef doing tree balancing and instead use path compression
189:       to adef traversing parent pointers over and over.
190: 
191:       Most of the variable names and identifiers are taken literally
192:       from_n this paper (and the original Tarjan paper mentioned above).
193:     '''
194: 
195:     def __init__(self, cfg, lsg):
196:         self.cfg_ = cfg  # current control flow graph.
197:         self.lsg_ = lsg  # loop forest.
198: 
199:     # Constants
200:     # / Marker for uninitialized nodes.
201:     K_UNVISITED = -1
202: 
203:     # / Safeguard against pathologic algorithm behavior.
204:     K_MAX_NON_BACK_PREDS = 32 * 1024
205: 
206:     '''
207:     As described in the paper, determine whether a node 'w' is a
208:     "True" ancestor for node 'v'.
209: 
210:     Dominance can be tested quickly using a pre-order trick
211:     for depth-first spanning trees. This is why dfs is the first
212:     thing we run below.
213:     '''
214: 
215:     @staticmethod
216:     def is_ancestor(w, v, last):
217:         return w <= v and v <= last[w]  # improve this ************************************************
218: 
219:     @staticmethod
220:     def dfs(current_node, nodes, number, last, current):
221:         # / Simple depth first traversal along out edges with node numbering.
222:         nodes[current].init(current_node, current)
223:         number[current_node] = current
224: 
225:         lastid = current
226:         for target in current_node.out_edges_:
227:             if number[target] == Havlak_loop_finder.K_UNVISITED:
228:                 lastid = Havlak_loop_finder.dfs(target, nodes, number, last, lastid + 1)
229: 
230:         last[number[current_node]] = lastid
231:         return lastid
232: 
233:     '''
234:     Find loops and build loop forest using Havlak's algorithm, which
235:     is derived from_n Tarjan. Variable names and step numbering has
236:     been chosen to be identical to the nomenclature in Havlak's
237:     paper (which is similar to the one used by Tarjan).
238:     '''
239: 
240:     def find_loops(self):
241:         if not self.cfg_.start_node_:
242:             return
243: 
244:         size = len(self.cfg_.basic_block_map_)
245:         non_back_preds = [set() for _ in xrange(size)]
246:         back_preds = [[] for _ in xrange(size)]
247:         header = [0] * size
248:         type = [0] * size
249:         last = [0] * size
250:         nodes = [Union_find_node() for _ in xrange(size)]
251: 
252:         number = {}
253: 
254:         # Step a:
255:         #   - initialize all nodes as unvisited.
256:         #   - depth-first traversal and numbering.
257:         #   - unreached BB's are marked as dead.
258:         #
259:         for bblock in self.cfg_.basic_block_map_.itervalues():
260:             number[bblock] = Havlak_loop_finder.K_UNVISITED
261: 
262:         Havlak_loop_finder.dfs(self.cfg_.start_node_, nodes, number, last, 0)
263: 
264:         # Step b:
265:         #   - iterate over all nodes.
266:         #
267:         #   A backedge comes from_n a descendant in the dfs tree, and non-backedges
268:         #   from_n non-descendants (following Tarjan).
269:         #
270:         #   - check incoming edges 'v' and add them to either
271:         #     - the list of backedges (back_preds) or
272:         #     - the list of non-backedges (non_back_preds)
273:         for w in xrange(size):
274:             header[w] = 0
275:             type[w] = Basic_block_class.NONHEADER
276: 
277:             node_w = nodes[w].bb_
278:             if not node_w:
279:                 type[w] = Basic_block_class.DEAD
280:                 continue  # dead BB
281: 
282:             if len(node_w.in_edges_):
283:                 for node_v in node_w.in_edges_:
284:                     v = number[node_v]
285:                     if v == Havlak_loop_finder.K_UNVISITED:
286:                         continue  # dead node
287: 
288:                     if Havlak_loop_finder.is_ancestor(w, v, last):
289:                         back_preds[w].append(v)
290:                     else:
291:                         non_back_preds[w].add(v)
292: 
293:         # Start node is root of all other loops.
294:         header[0] = 0
295: 
296:         # Step c:
297:         #
298:         # The outer loop, unchanged from_n Tarjan. It does nothing except
299:         # for those nodes which are the destinations of backedges.
300:         # For a header node w, we chase backward from_n the sources of the
301:         # backedges adding nodes to the set P, representing the body of
302:         # the loop headed by w.
303:         #
304:         # By running through the nodes in reverse of the DFST preorder,
305:         # we ensure that inner loop headers will be processed before the
306:         # headers for surrounding loops.
307:         for w in xrange(size - 1, -1, -1):
308:             node_pool = []  # this is 'P' in Havlak's paper
309:             node_w = nodes[w].bb_
310:             if not node_w:
311:                 continue  # dead BB
312: 
313:             # Step d:
314:             for back_pred in back_preds[w]:
315:                 if back_pred != w:
316:                     node_pool.append(nodes[back_pred].find_set())
317:                 else:
318:                     type[w] = Basic_block_class.SELF
319: 
320:             # Copy node_pool to worklist.
321:             worklist = []
322:             for np in node_pool:
323:                 worklist.append(np)
324: 
325:             if len(node_pool):
326:                 type[w] = Basic_block_class.REDUCIBLE
327: 
328:             # work the list...
329:             #
330:             while len(worklist):
331:                 x = worklist[0]
332:                 worklist = worklist[1:]  # slow? *************************************************
333: 
334:                 # Step e:
335:                 #
336:                 # Step e represents the main difference from_n Tarjan's method.
337:                 # Chasing upwards from_n the sources of a node w's backedges. If
338:                 # there is a node y' that is not a descendant of w, w is marked
339:                 # the header of an irreducible loop, there is another entry
340:                 # into this loop that avoids w.
341: 
342:                 # The algorithm has degenerated. Break and
343:                 # return in this case.
344:                 non_back_size = len(non_back_preds[x.dfs_number_])
345:                 if non_back_size > Havlak_loop_finder.K_MAX_NON_BACK_PREDS:
346:                     return
347: 
348:                 for non_back_pred_iter in non_back_preds[x.dfs_number_]:
349:                     y = nodes[non_back_pred_iter]
350:                     ydash = y.find_set()
351: 
352:                     if not Havlak_loop_finder.is_ancestor(w, ydash.dfs_number_, last):
353:                         type[w] = Basic_block_class.IRREDUCIBLE
354:                         non_back_preds[w].add(ydash.dfs_number_)
355:                     else:
356:                         if ydash.dfs_number_ != w:
357:                             if ydash not in node_pool:
358:                                 worklist.append(ydash)
359:                                 node_pool.append(ydash)
360: 
361:             # Collapse/Unionize nodes in a SCC to a single node
362:             # For every SCC found, create a loop descriptor and link it in.
363:             #
364:             if len(node_pool) or type[w] == Basic_block_class.SELF:
365:                 loop = self.lsg_.create_new_loop()
366: 
367:                 # At this point, one can set attributes to the loop, such as:
368:                 #
369:                 # the bottom node:
370:                 #    int[]::iterator iter  = back_preds[w].begin()
371:                 #    loop bottom is: nodes[*backp_iter].node)
372:                 #
373:                 # the number of backedges:
374:                 #    back_preds[w].length
375:                 #
376:                 # whether this loop is reducible:
377:                 #    type[w] != IRREDUCIBLE
378:                 #
379:                 # TODO(rhundt): Define those interfaces in the Loop Forest.
380:                 #
381:                 nodes[w].loop_ = loop
382: 
383:                 for node in node_pool:
384:                     # Add nodes to loop descriptor.
385:                     header[node.dfs_number_] = w
386:                     node.do_union(nodes[w])
387: 
388:                     # Nested loops are not added, but linked together.
389:                     if node.loop_:
390:                         node.loop_.parent_ = loop
391:                     else:
392:                         loop.add_node(node.bb_)
393: 
394:                 self.lsg_.loops_.append(loop)
395: 
396: 
397: def find_havlak_loops(cfg, lsg):
398:     '''External entry point.'''
399:     finder = Havlak_loop_finder(cfg, lsg)
400:     finder.find_loops()
401:     return len(lsg.loops_)
402: 
403: 
404: def build_diamond(cfg, start):
405:     bb0 = start
406:     Basic_block_edge(cfg, bb0, bb0 + 1)
407:     Basic_block_edge(cfg, bb0, bb0 + 2)
408:     Basic_block_edge(cfg, bb0 + 1, bb0 + 3)
409:     Basic_block_edge(cfg, bb0 + 2, bb0 + 3)
410:     return bb0 + 3
411: 
412: 
413: def build_connect(cfg, start, end):
414:     Basic_block_edge(cfg, start, end)
415: 
416: 
417: def build_straight(cfg, start, n):
418:     for i in xrange(n):
419:         build_connect(cfg, start + i, start + i + 1)
420:     return start + n
421: 
422: 
423: def build_base_loop(cfg, from_n):
424:     header = build_straight(cfg, from_n, 1)
425:     diamond1 = build_diamond(cfg, header)
426:     d11 = build_straight(cfg, diamond1, 1)
427:     diamond2 = build_diamond(cfg, d11)
428:     footer = build_straight(cfg, diamond2, 1)
429:     build_connect(cfg, diamond2, d11)
430:     build_connect(cfg, diamond1, header)
431:     build_connect(cfg, footer, from_n)
432:     footer = build_straight(cfg, footer, 1)
433:     return footer
434: 
435: 
436: # --- MOCKING CODE begin -------------------
437: # These data structures are stubbed out to make the code below easier to review.
438: 
439: class Basic_block_edge(object):
440:     '''Basic_block_edge only maintains two pointers to BasicBlocks.'''
441: 
442:     def __init__(self, cfg, from_name, to_name):
443:         self.from_ = cfg.create_node(from_name)
444:         self.to_ = cfg.create_node(to_name)
445:         self.from_.out_edges_.append(self.to_)
446:         self.to_.in_edges_.append(self.from_)
447:         cfg.edge_list_.append(self)
448: 
449: 
450: class Basic_block(object):
451:     '''Basic_block only maintains a vector of in-edges and a vector of out-edges.'''
452: 
453:     def __init__(self, name):
454:         self.in_edges_ = []
455:         self.out_edges_ = []
456:         self.name_ = name
457: 
458: 
459: class MaoCFG(object):
460:     '''MaoCFG maintains a list of nodes.'''
461: 
462:     def __init__(self):
463:         self.basic_block_map_ = {}
464:         self.start_node_ = None
465:         self.edge_list_ = []
466: 
467:     def create_node(self, name):
468:         if name in self.basic_block_map_:
469:             node = self.basic_block_map_[name]
470:         else:
471:             node = Basic_block(name)
472:             self.basic_block_map_[name] = node
473: 
474:         if len(self.basic_block_map_) == 1:
475:             self.start_node_ = node
476: 
477:         return node
478: 
479: 
480: # --- MOCKING CODE end  -------------------
481: 
482: 
483: def main():
484:     # print "Welcome to LoopTesterApp, Python edition"
485:     # print "Constructing App..."
486:     cfg = MaoCFG()
487:     lsg = Loop_structure_graph()
488: 
489:     # print "Constructing Simple cfg..."
490:     cfg.create_node(0)  # top
491:     build_base_loop(cfg, 0)
492:     cfg.create_node(1)  # bottom
493:     Basic_block_edge(cfg, 0, 2)
494: 
495:     # print "15000 dummy loops"
496:     for dummyLoops in xrange(15000):
497:         lsglocal = Loop_structure_graph()
498:         find_havlak_loops(cfg, lsglocal)
499: 
500:     # print "Constructing cfg..."
501:     n = 2
502: 
503:     for parlooptrees in xrange(10):
504:         cfg.create_node(n + 1)
505:         build_connect(cfg, 2, n + 1)
506:         n += 1
507: 
508:         for i in xrange(100):
509:             top = n
510:             n = build_straight(cfg, n, 1)
511:             for j in xrange(25):
512:                 n = build_base_loop(cfg, n)
513:             bottom = build_straight(cfg, n, 1)
514:             build_connect(cfg, n, top)
515:             n = bottom
516:         build_connect(cfg, n, 1)
517: 
518: 
519: ##    try:
520: ##        print "Performing Loop Recognition\n1 Iteration"
521: ##        numLoops = find_havlak_loops(cfg, lsg)
522: ##        print (numLoops)
523: ##    except Exception, e:
524: ##        print(e)
525: ##        
526: ##    print "Another 50 iterations..."
527: ##    sum = 0
528: ##    for i in xrange(50):
529: ##        lsg2 = Loop_structure_graph()
530: ##        stdout.write(".")
531: ##        sum += find_havlak_loops(cfg, lsg2)
532: ##
533: ##    print "\nFound %d loops (including artificial root node)(%d)" % (numLoops, sum)
534: ##    lsg.dump()
535: 
536: def run():
537:     main()
538:     return True
539: 
540: 
541: run()
542: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\nCopyright 2011 Google Inc.\n\nLicensed under the Apache License, Version 2.0 (the "License")\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http:#www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an "AS IS" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n\nThis version is designed for ShedSkin\nD code translated to Python by leonardo maffi, v.1.0, Jun 14 2011\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import sys' statement (line 20)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'sys', sys, module_type_store)


# Call to setrecursionlimit(...): (line 22)
# Processing the call arguments (line 22)
int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'int')
# Processing the call keyword arguments (line 22)
kwargs_5 = {}
# Getting the type of 'sys' (line 22)
sys_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'sys', False)
# Obtaining the member 'setrecursionlimit' of a type (line 22)
setrecursionlimit_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 0), sys_2, 'setrecursionlimit')
# Calling setrecursionlimit(args, kwargs) (line 22)
setrecursionlimit_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 22, 0), setrecursionlimit_3, *[int_4], **kwargs_5)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from sys import stdout' statement (line 23)
try:
    from sys import stdout

except:
    stdout = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'sys', None, module_type_store, ['stdout'], [stdout])

# Declaration of the 'Simple_loop' class

class Simple_loop(object, ):
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n    Basic representation of loops, a loop has an entry point,\n    one or more exit edges, a set of basic blocks, and potentially\n    an outer loop - a "parent" loop.\n\n    Furthermore, it can have any set of properties, e.g.,\n    it can be an irreducible loop, have control flow, be\n    a candidate for transformations, and what not.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple_loop.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 38):
        
        # Call to set(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_9 = {}
        # Getting the type of 'set' (line 38)
        set_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'set', False)
        # Calling set(args, kwargs) (line 38)
        set_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), set_8, *[], **kwargs_9)
        
        # Getting the type of 'self' (line 38)
        self_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'basic_blocks_' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_11, 'basic_blocks_', set_call_result_10)
        
        # Assigning a Call to a Attribute (line 39):
        
        # Call to set(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_13 = {}
        # Getting the type of 'set' (line 39)
        set_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'set', False)
        # Calling set(args, kwargs) (line 39)
        set_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 39, 25), set_12, *[], **kwargs_13)
        
        # Getting the type of 'self' (line 39)
        self_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'children_' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_15, 'children_', set_call_result_14)
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'None' (line 40)
        None_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'None')
        # Getting the type of 'self' (line 40)
        self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'parent_' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_17, 'parent_', None_16)
        
        # Assigning a Name to a Attribute (line 41):
        # Getting the type of 'False' (line 41)
        False_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'False')
        # Getting the type of 'self' (line 41)
        self_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'is_root_' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_19, 'is_root_', False_18)
        
        # Assigning a Num to a Attribute (line 42):
        int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'int')
        # Getting the type of 'self' (line 42)
        self_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'counter_' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_21, 'counter_', int_20)
        
        # Assigning a Num to a Attribute (line 43):
        int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'int')
        # Getting the type of 'self' (line 43)
        self_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'nesting_level_' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_23, 'nesting_level_', int_22)
        
        # Assigning a Num to a Attribute (line 44):
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'int')
        # Getting the type of 'self' (line 44)
        self_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self')
        # Setting the type of the member 'depth_level_' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_25, 'depth_level_', int_24)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_node(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_node'
        module_type_store = module_type_store.open_function_context('add_node', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Simple_loop.add_node.__dict__.__setitem__('stypy_localization', localization)
        Simple_loop.add_node.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Simple_loop.add_node.__dict__.__setitem__('stypy_type_store', module_type_store)
        Simple_loop.add_node.__dict__.__setitem__('stypy_function_name', 'Simple_loop.add_node')
        Simple_loop.add_node.__dict__.__setitem__('stypy_param_names_list', ['basic_block'])
        Simple_loop.add_node.__dict__.__setitem__('stypy_varargs_param_name', None)
        Simple_loop.add_node.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Simple_loop.add_node.__dict__.__setitem__('stypy_call_defaults', defaults)
        Simple_loop.add_node.__dict__.__setitem__('stypy_call_varargs', varargs)
        Simple_loop.add_node.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Simple_loop.add_node.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple_loop.add_node', ['basic_block'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_node', localization, ['basic_block'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_node(...)' code ##################

        
        # Call to add(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'basic_block' (line 47)
        basic_block_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'basic_block', False)
        # Processing the call keyword arguments (line 47)
        kwargs_30 = {}
        # Getting the type of 'self' (line 47)
        self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'basic_blocks_' of a type (line 47)
        basic_blocks__27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_26, 'basic_blocks_')
        # Obtaining the member 'add' of a type (line 47)
        add_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), basic_blocks__27, 'add')
        # Calling add(args, kwargs) (line 47)
        add_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), add_28, *[basic_block_29], **kwargs_30)
        
        
        # ################# End of 'add_node(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_node' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_node'
        return stypy_return_type_32


    @norecursion
    def add_child_loop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_child_loop'
        module_type_store = module_type_store.open_function_context('add_child_loop', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_localization', localization)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_type_store', module_type_store)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_function_name', 'Simple_loop.add_child_loop')
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_param_names_list', ['loop'])
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_varargs_param_name', None)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_call_defaults', defaults)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_call_varargs', varargs)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Simple_loop.add_child_loop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple_loop.add_child_loop', ['loop'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_child_loop', localization, ['loop'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_child_loop(...)' code ##################

        
        # Call to add(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'loop' (line 50)
        loop_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'loop', False)
        # Processing the call keyword arguments (line 50)
        kwargs_37 = {}
        # Getting the type of 'self' (line 50)
        self_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member 'children_' of a type (line 50)
        children__34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_33, 'children_')
        # Obtaining the member 'add' of a type (line 50)
        add_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), children__34, 'add')
        # Calling add(args, kwargs) (line 50)
        add_call_result_38 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), add_35, *[loop_36], **kwargs_37)
        
        
        # ################# End of 'add_child_loop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_child_loop' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_child_loop'
        return stypy_return_type_39


    @norecursion
    def dump(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump'
        module_type_store = module_type_store.open_function_context('dump', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Simple_loop.dump.__dict__.__setitem__('stypy_localization', localization)
        Simple_loop.dump.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Simple_loop.dump.__dict__.__setitem__('stypy_type_store', module_type_store)
        Simple_loop.dump.__dict__.__setitem__('stypy_function_name', 'Simple_loop.dump')
        Simple_loop.dump.__dict__.__setitem__('stypy_param_names_list', [])
        Simple_loop.dump.__dict__.__setitem__('stypy_varargs_param_name', None)
        Simple_loop.dump.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Simple_loop.dump.__dict__.__setitem__('stypy_call_defaults', defaults)
        Simple_loop.dump.__dict__.__setitem__('stypy_call_varargs', varargs)
        Simple_loop.dump.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Simple_loop.dump.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple_loop.dump', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump(...)' code ##################

        pass
        
        # ################# End of 'dump(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump'
        return stypy_return_type_40


    @norecursion
    def set_parent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_parent'
        module_type_store = module_type_store.open_function_context('set_parent', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Simple_loop.set_parent.__dict__.__setitem__('stypy_localization', localization)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_type_store', module_type_store)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_function_name', 'Simple_loop.set_parent')
        Simple_loop.set_parent.__dict__.__setitem__('stypy_param_names_list', ['parent'])
        Simple_loop.set_parent.__dict__.__setitem__('stypy_varargs_param_name', None)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_call_defaults', defaults)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_call_varargs', varargs)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Simple_loop.set_parent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple_loop.set_parent', ['parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_parent', localization, ['parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_parent(...)' code ##################

        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'parent' (line 59)
        parent_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'parent')
        # Getting the type of 'self' (line 59)
        self_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'parent_' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_42, 'parent_', parent_41)
        
        # Call to add_child_loop(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 30), 'self', False)
        # Processing the call keyword arguments (line 60)
        kwargs_46 = {}
        # Getting the type of 'parent' (line 60)
        parent_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'parent', False)
        # Obtaining the member 'add_child_loop' of a type (line 60)
        add_child_loop_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), parent_43, 'add_child_loop')
        # Calling add_child_loop(args, kwargs) (line 60)
        add_child_loop_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), add_child_loop_44, *[self_45], **kwargs_46)
        
        
        # ################# End of 'set_parent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_parent' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_parent'
        return stypy_return_type_48


    @norecursion
    def set_nesting_level(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_nesting_level'
        module_type_store = module_type_store.open_function_context('set_nesting_level', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_localization', localization)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_type_store', module_type_store)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_function_name', 'Simple_loop.set_nesting_level')
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_param_names_list', ['level'])
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_varargs_param_name', None)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_call_defaults', defaults)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_call_varargs', varargs)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Simple_loop.set_nesting_level.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple_loop.set_nesting_level', ['level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_nesting_level', localization, ['level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_nesting_level(...)' code ##################

        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'level' (line 63)
        level_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'level')
        # Getting the type of 'self' (line 63)
        self_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'nesting_level_' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_50, 'nesting_level_', level_49)
        
        # Getting the type of 'level' (line 64)
        level_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'level')
        int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'int')
        # Applying the binary operator '==' (line 64)
        result_eq_53 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '==', level_51, int_52)
        
        # Testing if the type of an if condition is none (line 64)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_53):
            pass
        else:
            
            # Testing the type of an if condition (line 64)
            if_condition_54 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_53)
            # Assigning a type to the variable 'if_condition_54' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_54', if_condition_54)
            # SSA begins for if statement (line 64)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 65):
            # Getting the type of 'True' (line 65)
            True_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'True')
            # Getting the type of 'self' (line 65)
            self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'self')
            # Setting the type of the member 'is_root_' of a type (line 65)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), self_56, 'is_root_', True_55)
            # SSA join for if statement (line 64)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'set_nesting_level(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_nesting_level' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_57)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_nesting_level'
        return stypy_return_type_57


# Assigning a type to the variable 'Simple_loop' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'Simple_loop', Simple_loop)
# Declaration of the 'Loop_structure_graph' class

class Loop_structure_graph(object, ):
    str_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', '\n    Maintain loop structure for a given cfg.\n\n    Two values are maintained for this loop graph, depth, and nesting level.\n    For example:\n\n    loop        nesting level    depth\n    ---------------------------------------\n    loop-0      2                0\n      loop-1    1                1\n      loop-3    1                1\n        loop-2  0                2\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loop_structure_graph.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 84):
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        
        # Getting the type of 'self' (line 84)
        self_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'loops_' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_60, 'loops_', list_59)
        
        # Assigning a Num to a Attribute (line 85):
        int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 29), 'int')
        # Getting the type of 'self' (line 85)
        self_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'loop_counter_' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_62, 'loop_counter_', int_61)
        
        # Assigning a Call to a Attribute (line 86):
        
        # Call to Simple_loop(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_64 = {}
        # Getting the type of 'Simple_loop' (line 86)
        Simple_loop_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'Simple_loop', False)
        # Calling Simple_loop(args, kwargs) (line 86)
        Simple_loop_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 86, 21), Simple_loop_63, *[], **kwargs_64)
        
        # Getting the type of 'self' (line 86)
        self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member 'root_' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_66, 'root_', Simple_loop_call_result_65)
        
        # Call to set_nesting_level(...): (line 87)
        # Processing the call arguments (line 87)
        int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 37), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_71 = {}
        # Getting the type of 'self' (line 87)
        self_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member 'root_' of a type (line 87)
        root__68 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_67, 'root_')
        # Obtaining the member 'set_nesting_level' of a type (line 87)
        set_nesting_level_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), root__68, 'set_nesting_level')
        # Calling set_nesting_level(args, kwargs) (line 87)
        set_nesting_level_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), set_nesting_level_69, *[int_70], **kwargs_71)
        
        
        # Assigning a Attribute to a Attribute (line 88):
        # Getting the type of 'self' (line 88)
        self_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'self')
        # Obtaining the member 'loop_counter_' of a type (line 88)
        loop_counter__74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 30), self_73, 'loop_counter_')
        # Getting the type of 'self' (line 88)
        self_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Obtaining the member 'root_' of a type (line 88)
        root__76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_75, 'root_')
        # Setting the type of the member 'counter_' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), root__76, 'counter_', loop_counter__74)
        
        # Getting the type of 'self' (line 89)
        self_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Obtaining the member 'loop_counter_' of a type (line 89)
        loop_counter__78 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_77, 'loop_counter_')
        int_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'int')
        # Applying the binary operator '+=' (line 89)
        result_iadd_80 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 8), '+=', loop_counter__78, int_79)
        # Getting the type of 'self' (line 89)
        self_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'loop_counter_' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_81, 'loop_counter_', result_iadd_80)
        
        
        # Call to append(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'self' (line 90)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'self', False)
        # Obtaining the member 'root_' of a type (line 90)
        root__86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 27), self_85, 'root_')
        # Processing the call keyword arguments (line 90)
        kwargs_87 = {}
        # Getting the type of 'self' (line 90)
        self_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'loops_' of a type (line 90)
        loops__83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_82, 'loops_')
        # Obtaining the member 'append' of a type (line 90)
        append_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), loops__83, 'append')
        # Calling append(args, kwargs) (line 90)
        append_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), append_84, *[root__86], **kwargs_87)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def create_new_loop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_new_loop'
        module_type_store = module_type_store.open_function_context('create_new_loop', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_localization', localization)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_type_store', module_type_store)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_function_name', 'Loop_structure_graph.create_new_loop')
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_param_names_list', [])
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_varargs_param_name', None)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_call_defaults', defaults)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_call_varargs', varargs)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Loop_structure_graph.create_new_loop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loop_structure_graph.create_new_loop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_new_loop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_new_loop(...)' code ##################

        
        # Assigning a Call to a Name (line 93):
        
        # Call to Simple_loop(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_90 = {}
        # Getting the type of 'Simple_loop' (line 93)
        Simple_loop_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'Simple_loop', False)
        # Calling Simple_loop(args, kwargs) (line 93)
        Simple_loop_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), Simple_loop_89, *[], **kwargs_90)
        
        # Assigning a type to the variable 'loop' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'loop', Simple_loop_call_result_91)
        
        # Assigning a Attribute to a Attribute (line 94):
        # Getting the type of 'self' (line 94)
        self_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'self')
        # Obtaining the member 'loop_counter_' of a type (line 94)
        loop_counter__93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 24), self_92, 'loop_counter_')
        # Getting the type of 'loop' (line 94)
        loop_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'loop')
        # Setting the type of the member 'counter_' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), loop_94, 'counter_', loop_counter__93)
        
        # Getting the type of 'self' (line 95)
        self_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Obtaining the member 'loop_counter_' of a type (line 95)
        loop_counter__96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_95, 'loop_counter_')
        int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'int')
        # Applying the binary operator '+=' (line 95)
        result_iadd_98 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 8), '+=', loop_counter__96, int_97)
        # Getting the type of 'self' (line 95)
        self_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'loop_counter_' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_99, 'loop_counter_', result_iadd_98)
        
        # Getting the type of 'loop' (line 96)
        loop_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'loop')
        # Assigning a type to the variable 'stypy_return_type' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', loop_100)
        
        # ################# End of 'create_new_loop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_new_loop' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_101)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_new_loop'
        return stypy_return_type_101


    @norecursion
    def dump(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump'
        module_type_store = module_type_store.open_function_context('dump', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_localization', localization)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_type_store', module_type_store)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_function_name', 'Loop_structure_graph.dump')
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_param_names_list', [])
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_varargs_param_name', None)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_call_defaults', defaults)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_call_varargs', varargs)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Loop_structure_graph.dump.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loop_structure_graph.dump', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump(...)' code ##################

        
        # Call to dump_rec(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'self' (line 99)
        self_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'self', False)
        # Obtaining the member 'root_' of a type (line 99)
        root__105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), self_104, 'root_')
        int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 34), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_107 = {}
        # Getting the type of 'self' (line 99)
        self_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self', False)
        # Obtaining the member 'dump_rec' of a type (line 99)
        dump_rec_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_102, 'dump_rec')
        # Calling dump_rec(args, kwargs) (line 99)
        dump_rec_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), dump_rec_103, *[root__105, int_106], **kwargs_107)
        
        
        # ################# End of 'dump(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_109)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump'
        return stypy_return_type_109


    @norecursion
    def dump_rec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump_rec'
        module_type_store = module_type_store.open_function_context('dump_rec', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_localization', localization)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_type_store', module_type_store)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_function_name', 'Loop_structure_graph.dump_rec')
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_param_names_list', ['loop', 'indent'])
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_varargs_param_name', None)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_call_defaults', defaults)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_call_varargs', varargs)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Loop_structure_graph.dump_rec.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loop_structure_graph.dump_rec', ['loop', 'indent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump_rec', localization, ['loop', 'indent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump_rec(...)' code ##################

        
        # Call to dump(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_112 = {}
        # Getting the type of 'loop' (line 103)
        loop_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'loop', False)
        # Obtaining the member 'dump' of a type (line 103)
        dump_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), loop_110, 'dump')
        # Calling dump(args, kwargs) (line 103)
        dump_call_result_113 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), dump_111, *[], **kwargs_112)
        
        
        # Getting the type of 'loop' (line 105)
        loop_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'loop')
        # Obtaining the member 'children_' of a type (line 105)
        children__115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 21), loop_114, 'children_')
        # Testing if the for loop is going to be iterated (line 105)
        # Testing the type of a for loop iterable (line 105)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 8), children__115)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 8), children__115):
            # Getting the type of the for loop variable (line 105)
            for_loop_var_116 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 8), children__115)
            # Assigning a type to the variable 'liter' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'liter', for_loop_var_116)
            # SSA begins for a for statement (line 105)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            pass
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'dump_rec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump_rec' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump_rec'
        return stypy_return_type_117


    @norecursion
    def calculate_nesting_level(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'calculate_nesting_level'
        module_type_store = module_type_store.open_function_context('calculate_nesting_level', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_localization', localization)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_type_store', module_type_store)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_function_name', 'Loop_structure_graph.calculate_nesting_level')
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_param_names_list', [])
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_varargs_param_name', None)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_call_defaults', defaults)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_call_varargs', varargs)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Loop_structure_graph.calculate_nesting_level.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loop_structure_graph.calculate_nesting_level', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'calculate_nesting_level', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'calculate_nesting_level(...)' code ##################

        
        # Getting the type of 'self' (line 110)
        self_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'self')
        # Obtaining the member 'loops_' of a type (line 110)
        loops__119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), self_118, 'loops_')
        # Testing if the for loop is going to be iterated (line 110)
        # Testing the type of a for loop iterable (line 110)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 8), loops__119)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 8), loops__119):
            # Getting the type of the for loop variable (line 110)
            for_loop_var_120 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 8), loops__119)
            # Assigning a type to the variable 'loop' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'loop', for_loop_var_120)
            # SSA begins for a for statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'loop' (line 111)
            loop_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'loop')
            # Obtaining the member 'is_root_' of a type (line 111)
            is_root__122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), loop_121, 'is_root_')
            # Testing if the type of an if condition is none (line 111)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 111, 12), is_root__122):
                pass
            else:
                
                # Testing the type of an if condition (line 111)
                if_condition_123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), is_root__122)
                # Assigning a type to the variable 'if_condition_123' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_123', if_condition_123)
                # SSA begins for if statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 111)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'loop' (line 113)
            loop_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'loop')
            # Obtaining the member 'parent_' of a type (line 113)
            parent__125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), loop_124, 'parent_')
            # Applying the 'not' unary operator (line 113)
            result_not__126 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'not', parent__125)
            
            # Testing if the type of an if condition is none (line 113)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 113, 12), result_not__126):
                pass
            else:
                
                # Testing the type of an if condition (line 113)
                if_condition_127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_not__126)
                # Assigning a type to the variable 'if_condition_127' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_127', if_condition_127)
                # SSA begins for if statement (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_parent(...): (line 114)
                # Processing the call arguments (line 114)
                # Getting the type of 'self' (line 114)
                self_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 32), 'self', False)
                # Obtaining the member 'root_' of a type (line 114)
                root__131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 32), self_130, 'root_')
                # Processing the call keyword arguments (line 114)
                kwargs_132 = {}
                # Getting the type of 'loop' (line 114)
                loop_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'loop', False)
                # Obtaining the member 'set_parent' of a type (line 114)
                set_parent_129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), loop_128, 'set_parent')
                # Calling set_parent(args, kwargs) (line 114)
                set_parent_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), set_parent_129, *[root__131], **kwargs_132)
                
                # SSA join for if statement (line 113)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to calculate_nesting_level_rec(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 41), 'self', False)
        # Obtaining the member 'root_' of a type (line 117)
        root__137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 41), self_136, 'root_')
        int_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 53), 'int')
        # Processing the call keyword arguments (line 117)
        kwargs_139 = {}
        # Getting the type of 'self' (line 117)
        self_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', False)
        # Obtaining the member 'calculate_nesting_level_rec' of a type (line 117)
        calculate_nesting_level_rec_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_134, 'calculate_nesting_level_rec')
        # Calling calculate_nesting_level_rec(args, kwargs) (line 117)
        calculate_nesting_level_rec_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), calculate_nesting_level_rec_135, *[root__137, int_138], **kwargs_139)
        
        
        # ################# End of 'calculate_nesting_level(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'calculate_nesting_level' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_141)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'calculate_nesting_level'
        return stypy_return_type_141


    @norecursion
    def calculate_nesting_level_rec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'calculate_nesting_level_rec'
        module_type_store = module_type_store.open_function_context('calculate_nesting_level_rec', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_localization', localization)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_type_store', module_type_store)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_function_name', 'Loop_structure_graph.calculate_nesting_level_rec')
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_param_names_list', ['loop', 'depth'])
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_varargs_param_name', None)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_call_defaults', defaults)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_call_varargs', varargs)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Loop_structure_graph.calculate_nesting_level_rec.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loop_structure_graph.calculate_nesting_level_rec', ['loop', 'depth'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'calculate_nesting_level_rec', localization, ['loop', 'depth'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'calculate_nesting_level_rec(...)' code ##################

        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'depth' (line 120)
        depth_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'depth')
        # Getting the type of 'loop' (line 120)
        loop_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'loop')
        # Setting the type of the member 'depth_level_' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), loop_143, 'depth_level_', depth_142)
        
        # Getting the type of 'loop' (line 121)
        loop_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'loop')
        # Obtaining the member 'children_' of a type (line 121)
        children__145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 18), loop_144, 'children_')
        # Testing if the for loop is going to be iterated (line 121)
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 8), children__145)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 121, 8), children__145):
            # Getting the type of the for loop variable (line 121)
            for_loop_var_146 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 8), children__145)
            # Assigning a type to the variable 'ch' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'ch', for_loop_var_146)
            # SSA begins for a for statement (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to calculate_nesting_level_rec(...): (line 122)
            # Processing the call arguments (line 122)
            # Getting the type of 'ch' (line 122)
            ch_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'ch', False)
            # Getting the type of 'depth' (line 122)
            depth_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 44), 'depth', False)
            int_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'int')
            # Applying the binary operator '+' (line 122)
            result_add_151 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 44), '+', depth_149, int_150)
            
            # Processing the call keyword arguments (line 122)
            kwargs_152 = {}
            # Getting the type of 'calculate_nesting_level_rec' (line 122)
            calculate_nesting_level_rec_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'calculate_nesting_level_rec', False)
            # Calling calculate_nesting_level_rec(args, kwargs) (line 122)
            calculate_nesting_level_rec_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), calculate_nesting_level_rec_147, *[ch_148, result_add_151], **kwargs_152)
            
            
            # Assigning a Call to a Attribute (line 123):
            
            # Call to max(...): (line 123)
            # Processing the call arguments (line 123)
            # Getting the type of 'loop' (line 123)
            loop_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'loop', False)
            # Obtaining the member 'nesting_level_' of a type (line 123)
            nesting_level__156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 38), loop_155, 'nesting_level_')
            int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 59), 'int')
            # Getting the type of 'ch' (line 123)
            ch_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 63), 'ch', False)
            # Obtaining the member 'nesting_level_' of a type (line 123)
            nesting_level__159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 63), ch_158, 'nesting_level_')
            # Applying the binary operator '+' (line 123)
            result_add_160 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 59), '+', int_157, nesting_level__159)
            
            # Processing the call keyword arguments (line 123)
            kwargs_161 = {}
            # Getting the type of 'max' (line 123)
            max_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'max', False)
            # Calling max(args, kwargs) (line 123)
            max_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 123, 34), max_154, *[nesting_level__156, result_add_160], **kwargs_161)
            
            # Getting the type of 'loop' (line 123)
            loop_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'loop')
            # Setting the type of the member 'nesting_level_' of a type (line 123)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), loop_163, 'nesting_level_', max_call_result_162)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'calculate_nesting_level_rec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'calculate_nesting_level_rec' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'calculate_nesting_level_rec'
        return stypy_return_type_164


# Assigning a type to the variable 'Loop_structure_graph' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'Loop_structure_graph', Loop_structure_graph)
# Declaration of the 'Union_find_node' class

class Union_find_node(object, ):
    str_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', '\n    Union/Find algorithm after Tarjan, R.E., 1983, Data Structures\n    and Network Algorithms.\n    ')

    @norecursion
    def init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'init'
        module_type_store = module_type_store.open_function_context('init', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Union_find_node.init.__dict__.__setitem__('stypy_localization', localization)
        Union_find_node.init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Union_find_node.init.__dict__.__setitem__('stypy_type_store', module_type_store)
        Union_find_node.init.__dict__.__setitem__('stypy_function_name', 'Union_find_node.init')
        Union_find_node.init.__dict__.__setitem__('stypy_param_names_list', ['bb', 'dfs_number'])
        Union_find_node.init.__dict__.__setitem__('stypy_varargs_param_name', None)
        Union_find_node.init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Union_find_node.init.__dict__.__setitem__('stypy_call_defaults', defaults)
        Union_find_node.init.__dict__.__setitem__('stypy_call_varargs', varargs)
        Union_find_node.init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Union_find_node.init.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Union_find_node.init', ['bb', 'dfs_number'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'init', localization, ['bb', 'dfs_number'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'init(...)' code ##################

        
        # Assigning a Name to a Attribute (line 138):
        # Getting the type of 'self' (line 138)
        self_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'self')
        # Getting the type of 'self' (line 138)
        self_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
        # Setting the type of the member 'parent_' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_167, 'parent_', self_166)
        
        # Assigning a Name to a Attribute (line 139):
        # Getting the type of 'bb' (line 139)
        bb_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'bb')
        # Getting the type of 'self' (line 139)
        self_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member 'bb_' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_169, 'bb_', bb_168)
        
        # Assigning a Name to a Attribute (line 140):
        # Getting the type of 'dfs_number' (line 140)
        dfs_number_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'dfs_number')
        # Getting the type of 'self' (line 140)
        self_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self')
        # Setting the type of the member 'dfs_number_' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_171, 'dfs_number_', dfs_number_170)
        
        # Assigning a Name to a Attribute (line 141):
        # Getting the type of 'None' (line 141)
        None_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'None')
        # Getting the type of 'self' (line 141)
        self_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self')
        # Setting the type of the member 'loop_' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_173, 'loop_', None_172)
        
        # ################# End of 'init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'init' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_174)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'init'
        return stypy_return_type_174


    @norecursion
    def find_set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_set'
        module_type_store = module_type_store.open_function_context('find_set', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Union_find_node.find_set.__dict__.__setitem__('stypy_localization', localization)
        Union_find_node.find_set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Union_find_node.find_set.__dict__.__setitem__('stypy_type_store', module_type_store)
        Union_find_node.find_set.__dict__.__setitem__('stypy_function_name', 'Union_find_node.find_set')
        Union_find_node.find_set.__dict__.__setitem__('stypy_param_names_list', [])
        Union_find_node.find_set.__dict__.__setitem__('stypy_varargs_param_name', None)
        Union_find_node.find_set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Union_find_node.find_set.__dict__.__setitem__('stypy_call_defaults', defaults)
        Union_find_node.find_set.__dict__.__setitem__('stypy_call_varargs', varargs)
        Union_find_node.find_set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Union_find_node.find_set.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Union_find_node.find_set', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_set', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_set(...)' code ##################

        str_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, (-1)), 'str', '\n        Union/Find Algorithm - The find routine.\n\n        Implemented with Path Compression (inner loops are only\n        visited and collapsed once, however, deep nests would still\n        result in significant traversals).\n        ')
        
        # Assigning a List to a Name (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        
        # Assigning a type to the variable 'nodeList' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'nodeList', list_176)
        
        # Assigning a Name to a Name (line 153):
        # Getting the type of 'self' (line 153)
        self_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'self')
        # Assigning a type to the variable 'node' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'node', self_177)
        
        
        # Getting the type of 'node' (line 154)
        node_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 14), 'node')
        # Getting the type of 'node' (line 154)
        node_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'node')
        # Obtaining the member 'parent_' of a type (line 154)
        parent__180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), node_179, 'parent_')
        # Applying the binary operator '!=' (line 154)
        result_ne_181 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 14), '!=', node_178, parent__180)
        
        # Testing if the while is going to be iterated (line 154)
        # Testing the type of an if condition (line 154)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), result_ne_181)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 154, 8), result_ne_181):
            # SSA begins for while statement (line 154)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 'node' (line 155)
            node_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'node')
            # Obtaining the member 'parent_' of a type (line 155)
            parent__183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 15), node_182, 'parent_')
            # Getting the type of 'node' (line 155)
            node_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 31), 'node')
            # Obtaining the member 'parent_' of a type (line 155)
            parent__185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 31), node_184, 'parent_')
            # Obtaining the member 'parent_' of a type (line 155)
            parent__186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 31), parent__185, 'parent_')
            # Applying the binary operator '!=' (line 155)
            result_ne_187 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '!=', parent__183, parent__186)
            
            # Testing if the type of an if condition is none (line 155)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 155, 12), result_ne_187):
                pass
            else:
                
                # Testing the type of an if condition (line 155)
                if_condition_188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 12), result_ne_187)
                # Assigning a type to the variable 'if_condition_188' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'if_condition_188', if_condition_188)
                # SSA begins for if statement (line 155)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 156)
                # Processing the call arguments (line 156)
                # Getting the type of 'node' (line 156)
                node_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'node', False)
                # Processing the call keyword arguments (line 156)
                kwargs_192 = {}
                # Getting the type of 'nodeList' (line 156)
                nodeList_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'nodeList', False)
                # Obtaining the member 'append' of a type (line 156)
                append_190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), nodeList_189, 'append')
                # Calling append(args, kwargs) (line 156)
                append_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), append_190, *[node_191], **kwargs_192)
                
                # SSA join for if statement (line 155)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Attribute to a Name (line 157):
            # Getting the type of 'node' (line 157)
            node_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'node')
            # Obtaining the member 'parent_' of a type (line 157)
            parent__195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 19), node_194, 'parent_')
            # Assigning a type to the variable 'node' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'node', parent__195)
            # SSA join for while statement (line 154)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'nodeList' (line 160)
        nodeList_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'nodeList')
        # Testing if the for loop is going to be iterated (line 160)
        # Testing the type of a for loop iterable (line 160)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 8), nodeList_196)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 160, 8), nodeList_196):
            # Getting the type of the for loop variable (line 160)
            for_loop_var_197 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 8), nodeList_196)
            # Assigning a type to the variable 'n' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'n', for_loop_var_197)
            # SSA begins for a for statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Attribute to a Attribute (line 161):
            # Getting the type of 'node' (line 161)
            node_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'node')
            # Obtaining the member 'parent_' of a type (line 161)
            parent__199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 24), node_198, 'parent_')
            # Getting the type of 'n' (line 161)
            n_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'n')
            # Setting the type of the member 'parent_' of a type (line 161)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), n_200, 'parent_', parent__199)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'node' (line 163)
        node_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', node_201)
        
        # ################# End of 'find_set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_set' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_set'
        return stypy_return_type_202


    @norecursion
    def do_union(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'do_union'
        module_type_store = module_type_store.open_function_context('do_union', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Union_find_node.do_union.__dict__.__setitem__('stypy_localization', localization)
        Union_find_node.do_union.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Union_find_node.do_union.__dict__.__setitem__('stypy_type_store', module_type_store)
        Union_find_node.do_union.__dict__.__setitem__('stypy_function_name', 'Union_find_node.do_union')
        Union_find_node.do_union.__dict__.__setitem__('stypy_param_names_list', ['B'])
        Union_find_node.do_union.__dict__.__setitem__('stypy_varargs_param_name', None)
        Union_find_node.do_union.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Union_find_node.do_union.__dict__.__setitem__('stypy_call_defaults', defaults)
        Union_find_node.do_union.__dict__.__setitem__('stypy_call_varargs', varargs)
        Union_find_node.do_union.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Union_find_node.do_union.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Union_find_node.do_union', ['B'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'do_union', localization, ['B'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'do_union(...)' code ##################

        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'B' (line 167)
        B_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'B')
        # Getting the type of 'self' (line 167)
        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'parent_' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_204, 'parent_', B_203)
        
        # ################# End of 'do_union(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'do_union' in the type store
        # Getting the type of 'stypy_return_type' (line 166)
        stypy_return_type_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'do_union'
        return stypy_return_type_205


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 131, 0, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Union_find_node.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Union_find_node' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'Union_find_node', Union_find_node)
# Declaration of the 'Basic_block_class' class

class Basic_block_class(object, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 170, 0, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Basic_block_class.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Basic_block_class' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'Basic_block_class', Basic_block_class)

# Assigning a Num to a Name (line 171):
int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 10), 'int')
# Getting the type of 'Basic_block_class'
Basic_block_class_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Basic_block_class')
# Setting the type of the member 'TOP' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Basic_block_class_207, 'TOP', int_206)

# Assigning a Num to a Name (line 172):
int_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 16), 'int')
# Getting the type of 'Basic_block_class'
Basic_block_class_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Basic_block_class')
# Setting the type of the member 'NONHEADER' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Basic_block_class_209, 'NONHEADER', int_208)

# Assigning a Num to a Name (line 173):
int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 16), 'int')
# Getting the type of 'Basic_block_class'
Basic_block_class_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Basic_block_class')
# Setting the type of the member 'REDUCIBLE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Basic_block_class_211, 'REDUCIBLE', int_210)

# Assigning a Num to a Name (line 174):
int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 11), 'int')
# Getting the type of 'Basic_block_class'
Basic_block_class_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Basic_block_class')
# Setting the type of the member 'SELF' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Basic_block_class_213, 'SELF', int_212)

# Assigning a Num to a Name (line 175):
int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 18), 'int')
# Getting the type of 'Basic_block_class'
Basic_block_class_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Basic_block_class')
# Setting the type of the member 'IRREDUCIBLE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Basic_block_class_215, 'IRREDUCIBLE', int_214)

# Assigning a Num to a Name (line 176):
int_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 11), 'int')
# Getting the type of 'Basic_block_class'
Basic_block_class_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Basic_block_class')
# Setting the type of the member 'DEAD' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Basic_block_class_217, 'DEAD', int_216)

# Assigning a Num to a Name (line 177):
int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 11), 'int')
# Getting the type of 'Basic_block_class'
Basic_block_class_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Basic_block_class')
# Setting the type of the member 'LAST' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Basic_block_class_219, 'LAST', int_218)
# Declaration of the 'Havlak_loop_finder' class

class Havlak_loop_finder(object, ):
    str_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', '\n    Loop Recognition\n\n    based on:\n      Paul Havlak, Nesting of Reducible and Irreducible Loops,\n         Rice University.\n\n      We adef doing tree balancing and instead use path compression\n      to adef traversing parent pointers over and over.\n\n      Most of the variable names and identifiers are taken literally\n      from_n this paper (and the original Tarjan paper mentioned above).\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Havlak_loop_finder.__init__', ['cfg', 'lsg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['cfg', 'lsg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 196):
        # Getting the type of 'cfg' (line 196)
        cfg_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'cfg')
        # Getting the type of 'self' (line 196)
        self_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member 'cfg_' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_222, 'cfg_', cfg_221)
        
        # Assigning a Name to a Attribute (line 197):
        # Getting the type of 'lsg' (line 197)
        lsg_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'lsg')
        # Getting the type of 'self' (line 197)
        self_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self')
        # Setting the type of the member 'lsg_' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_224, 'lsg_', lsg_223)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()

    str_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, (-1)), 'str', '\n    As described in the paper, determine whether a node \'w\' is a\n    "True" ancestor for node \'v\'.\n\n    Dominance can be tested quickly using a pre-order trick\n    for depth-first spanning trees. This is why dfs is the first\n    thing we run below.\n    ')

    @staticmethod
    @norecursion
    def is_ancestor(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_ancestor'
        module_type_store = module_type_store.open_function_context('is_ancestor', 215, 4, False)
        
        # Passed parameters checking function
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_localization', localization)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_type_of_self', None)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_type_store', module_type_store)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_function_name', 'is_ancestor')
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_param_names_list', ['w', 'v', 'last'])
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_varargs_param_name', None)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_call_defaults', defaults)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_call_varargs', varargs)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Havlak_loop_finder.is_ancestor.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, 'is_ancestor', ['w', 'v', 'last'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_ancestor', localization, ['v', 'last'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_ancestor(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Getting the type of 'w' (line 217)
        w_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'w')
        # Getting the type of 'v' (line 217)
        v_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'v')
        # Applying the binary operator '<=' (line 217)
        result_le_228 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 15), '<=', w_226, v_227)
        
        
        # Getting the type of 'v' (line 217)
        v_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'v')
        
        # Obtaining the type of the subscript
        # Getting the type of 'w' (line 217)
        w_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 36), 'w')
        # Getting the type of 'last' (line 217)
        last_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'last')
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 31), last_231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 217, 31), getitem___232, w_230)
        
        # Applying the binary operator '<=' (line 217)
        result_le_234 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 26), '<=', v_229, subscript_call_result_233)
        
        # Applying the binary operator 'and' (line 217)
        result_and_keyword_235 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 15), 'and', result_le_228, result_le_234)
        
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', result_and_keyword_235)
        
        # ################# End of 'is_ancestor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_ancestor' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_ancestor'
        return stypy_return_type_236


    @staticmethod
    @norecursion
    def dfs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dfs'
        module_type_store = module_type_store.open_function_context('dfs', 219, 4, False)
        
        # Passed parameters checking function
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_localization', localization)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_type_of_self', None)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_type_store', module_type_store)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_function_name', 'dfs')
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_param_names_list', ['current_node', 'nodes', 'number', 'last', 'current'])
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_varargs_param_name', None)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_call_defaults', defaults)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_call_varargs', varargs)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Havlak_loop_finder.dfs.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, None, module_type_store, 'dfs', ['current_node', 'nodes', 'number', 'last', 'current'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dfs', localization, ['nodes', 'number', 'last', 'current'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dfs(...)' code ##################

        
        # Call to init(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'current_node' (line 222)
        current_node_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'current_node', False)
        # Getting the type of 'current' (line 222)
        current_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 42), 'current', False)
        # Processing the call keyword arguments (line 222)
        kwargs_244 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'current' (line 222)
        current_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 14), 'current', False)
        # Getting the type of 'nodes' (line 222)
        nodes_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'nodes', False)
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), nodes_238, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), getitem___239, current_237)
        
        # Obtaining the member 'init' of a type (line 222)
        init_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), subscript_call_result_240, 'init')
        # Calling init(args, kwargs) (line 222)
        init_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), init_241, *[current_node_242, current_243], **kwargs_244)
        
        
        # Assigning a Name to a Subscript (line 223):
        # Getting the type of 'current' (line 223)
        current_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 31), 'current')
        # Getting the type of 'number' (line 223)
        number_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'number')
        # Getting the type of 'current_node' (line 223)
        current_node_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'current_node')
        # Storing an element on a container (line 223)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 8), number_247, (current_node_248, current_246))
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'current' (line 225)
        current_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'current')
        # Assigning a type to the variable 'lastid' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'lastid', current_249)
        
        # Getting the type of 'current_node' (line 226)
        current_node_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 22), 'current_node')
        # Obtaining the member 'out_edges_' of a type (line 226)
        out_edges__251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 22), current_node_250, 'out_edges_')
        # Testing if the for loop is going to be iterated (line 226)
        # Testing the type of a for loop iterable (line 226)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), out_edges__251)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 226, 8), out_edges__251):
            # Getting the type of the for loop variable (line 226)
            for_loop_var_252 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), out_edges__251)
            # Assigning a type to the variable 'target' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'target', for_loop_var_252)
            # SSA begins for a for statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'target' (line 227)
            target_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'target')
            # Getting the type of 'number' (line 227)
            number_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'number')
            # Obtaining the member '__getitem__' of a type (line 227)
            getitem___255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 15), number_254, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 227)
            subscript_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), getitem___255, target_253)
            
            # Getting the type of 'Havlak_loop_finder' (line 227)
            Havlak_loop_finder_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'Havlak_loop_finder')
            # Obtaining the member 'K_UNVISITED' of a type (line 227)
            K_UNVISITED_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 33), Havlak_loop_finder_257, 'K_UNVISITED')
            # Applying the binary operator '==' (line 227)
            result_eq_259 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 15), '==', subscript_call_result_256, K_UNVISITED_258)
            
            # Testing if the type of an if condition is none (line 227)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 227, 12), result_eq_259):
                pass
            else:
                
                # Testing the type of an if condition (line 227)
                if_condition_260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 12), result_eq_259)
                # Assigning a type to the variable 'if_condition_260' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'if_condition_260', if_condition_260)
                # SSA begins for if statement (line 227)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 228):
                
                # Call to dfs(...): (line 228)
                # Processing the call arguments (line 228)
                # Getting the type of 'target' (line 228)
                target_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 48), 'target', False)
                # Getting the type of 'nodes' (line 228)
                nodes_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 56), 'nodes', False)
                # Getting the type of 'number' (line 228)
                number_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 63), 'number', False)
                # Getting the type of 'last' (line 228)
                last_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 71), 'last', False)
                # Getting the type of 'lastid' (line 228)
                lastid_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 77), 'lastid', False)
                int_268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 86), 'int')
                # Applying the binary operator '+' (line 228)
                result_add_269 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 77), '+', lastid_267, int_268)
                
                # Processing the call keyword arguments (line 228)
                kwargs_270 = {}
                # Getting the type of 'Havlak_loop_finder' (line 228)
                Havlak_loop_finder_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), 'Havlak_loop_finder', False)
                # Obtaining the member 'dfs' of a type (line 228)
                dfs_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 25), Havlak_loop_finder_261, 'dfs')
                # Calling dfs(args, kwargs) (line 228)
                dfs_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 228, 25), dfs_262, *[target_263, nodes_264, number_265, last_266, result_add_269], **kwargs_270)
                
                # Assigning a type to the variable 'lastid' (line 228)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'lastid', dfs_call_result_271)
                # SSA join for if statement (line 227)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Subscript (line 230):
        # Getting the type of 'lastid' (line 230)
        lastid_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 37), 'lastid')
        # Getting the type of 'last' (line 230)
        last_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'last')
        
        # Obtaining the type of the subscript
        # Getting the type of 'current_node' (line 230)
        current_node_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'current_node')
        # Getting the type of 'number' (line 230)
        number_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 13), 'number')
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 13), number_275, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 230, 13), getitem___276, current_node_274)
        
        # Storing an element on a container (line 230)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 8), last_273, (subscript_call_result_277, lastid_272))
        # Getting the type of 'lastid' (line 231)
        lastid_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'lastid')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', lastid_278)
        
        # ################# End of 'dfs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dfs' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dfs'
        return stypy_return_type_279

    str_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'str', "\n    Find loops and build loop forest using Havlak's algorithm, which\n    is derived from_n Tarjan. Variable names and step numbering has\n    been chosen to be identical to the nomenclature in Havlak's\n    paper (which is similar to the one used by Tarjan).\n    ")

    @norecursion
    def find_loops(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_loops'
        module_type_store = module_type_store.open_function_context('find_loops', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_localization', localization)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_type_store', module_type_store)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_function_name', 'Havlak_loop_finder.find_loops')
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_param_names_list', [])
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_varargs_param_name', None)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_call_defaults', defaults)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_call_varargs', varargs)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Havlak_loop_finder.find_loops.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Havlak_loop_finder.find_loops', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_loops', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_loops(...)' code ##################

        
        # Getting the type of 'self' (line 241)
        self_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'self')
        # Obtaining the member 'cfg_' of a type (line 241)
        cfg__282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), self_281, 'cfg_')
        # Obtaining the member 'start_node_' of a type (line 241)
        start_node__283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), cfg__282, 'start_node_')
        # Applying the 'not' unary operator (line 241)
        result_not__284 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 11), 'not', start_node__283)
        
        # Testing if the type of an if condition is none (line 241)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 241, 8), result_not__284):
            pass
        else:
            
            # Testing the type of an if condition (line 241)
            if_condition_285 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), result_not__284)
            # Assigning a type to the variable 'if_condition_285' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_285', if_condition_285)
            # SSA begins for if statement (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 241)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 244):
        
        # Call to len(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'self' (line 244)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'self', False)
        # Obtaining the member 'cfg_' of a type (line 244)
        cfg__288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), self_287, 'cfg_')
        # Obtaining the member 'basic_block_map_' of a type (line 244)
        basic_block_map__289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), cfg__288, 'basic_block_map_')
        # Processing the call keyword arguments (line 244)
        kwargs_290 = {}
        # Getting the type of 'len' (line 244)
        len_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'len', False)
        # Calling len(args, kwargs) (line 244)
        len_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 244, 15), len_286, *[basic_block_map__289], **kwargs_290)
        
        # Assigning a type to the variable 'size' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'size', len_call_result_291)
        
        # Assigning a ListComp to a Name (line 245):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'size' (line 245)
        size_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 48), 'size', False)
        # Processing the call keyword arguments (line 245)
        kwargs_297 = {}
        # Getting the type of 'xrange' (line 245)
        xrange_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'xrange', False)
        # Calling xrange(args, kwargs) (line 245)
        xrange_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 245, 41), xrange_295, *[size_296], **kwargs_297)
        
        comprehension_299 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 26), xrange_call_result_298)
        # Assigning a type to the variable '_' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), '_', comprehension_299)
        
        # Call to set(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_293 = {}
        # Getting the type of 'set' (line 245)
        set_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'set', False)
        # Calling set(args, kwargs) (line 245)
        set_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 245, 26), set_292, *[], **kwargs_293)
        
        list_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 26), list_300, set_call_result_294)
        # Assigning a type to the variable 'non_back_preds' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'non_back_preds', list_300)
        
        # Assigning a ListComp to a Name (line 246):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'size' (line 246)
        size_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 41), 'size', False)
        # Processing the call keyword arguments (line 246)
        kwargs_304 = {}
        # Getting the type of 'xrange' (line 246)
        xrange_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 34), 'xrange', False)
        # Calling xrange(args, kwargs) (line 246)
        xrange_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 246, 34), xrange_302, *[size_303], **kwargs_304)
        
        comprehension_306 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 22), xrange_call_result_305)
        # Assigning a type to the variable '_' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), '_', comprehension_306)
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        
        list_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 22), list_307, list_301)
        # Assigning a type to the variable 'back_preds' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'back_preds', list_307)
        
        # Assigning a BinOp to a Name (line 247):
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 17), list_308, int_309)
        
        # Getting the type of 'size' (line 247)
        size_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'size')
        # Applying the binary operator '*' (line 247)
        result_mul_311 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 17), '*', list_308, size_310)
        
        # Assigning a type to the variable 'header' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'header', result_mul_311)
        
        # Assigning a BinOp to a Name (line 248):
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        int_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 15), list_312, int_313)
        
        # Getting the type of 'size' (line 248)
        size_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'size')
        # Applying the binary operator '*' (line 248)
        result_mul_315 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 15), '*', list_312, size_314)
        
        # Assigning a type to the variable 'type' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'type', result_mul_315)
        
        # Assigning a BinOp to a Name (line 249):
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        int_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 15), list_316, int_317)
        
        # Getting the type of 'size' (line 249)
        size_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'size')
        # Applying the binary operator '*' (line 249)
        result_mul_319 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 15), '*', list_316, size_318)
        
        # Assigning a type to the variable 'last' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'last', result_mul_319)
        
        # Assigning a ListComp to a Name (line 250):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'size' (line 250)
        size_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 51), 'size', False)
        # Processing the call keyword arguments (line 250)
        kwargs_325 = {}
        # Getting the type of 'xrange' (line 250)
        xrange_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 44), 'xrange', False)
        # Calling xrange(args, kwargs) (line 250)
        xrange_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 250, 44), xrange_323, *[size_324], **kwargs_325)
        
        comprehension_327 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 17), xrange_call_result_326)
        # Assigning a type to the variable '_' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 17), '_', comprehension_327)
        
        # Call to Union_find_node(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_321 = {}
        # Getting the type of 'Union_find_node' (line 250)
        Union_find_node_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 17), 'Union_find_node', False)
        # Calling Union_find_node(args, kwargs) (line 250)
        Union_find_node_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 250, 17), Union_find_node_320, *[], **kwargs_321)
        
        list_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 17), list_328, Union_find_node_call_result_322)
        # Assigning a type to the variable 'nodes' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'nodes', list_328)
        
        # Assigning a Dict to a Name (line 252):
        
        # Obtaining an instance of the builtin type 'dict' (line 252)
        dict_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 252)
        
        # Assigning a type to the variable 'number' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'number', dict_329)
        
        
        # Call to itervalues(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_334 = {}
        # Getting the type of 'self' (line 259)
        self_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 22), 'self', False)
        # Obtaining the member 'cfg_' of a type (line 259)
        cfg__331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 22), self_330, 'cfg_')
        # Obtaining the member 'basic_block_map_' of a type (line 259)
        basic_block_map__332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 22), cfg__331, 'basic_block_map_')
        # Obtaining the member 'itervalues' of a type (line 259)
        itervalues_333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 22), basic_block_map__332, 'itervalues')
        # Calling itervalues(args, kwargs) (line 259)
        itervalues_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 259, 22), itervalues_333, *[], **kwargs_334)
        
        # Testing if the for loop is going to be iterated (line 259)
        # Testing the type of a for loop iterable (line 259)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 259, 8), itervalues_call_result_335)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 259, 8), itervalues_call_result_335):
            # Getting the type of the for loop variable (line 259)
            for_loop_var_336 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 259, 8), itervalues_call_result_335)
            # Assigning a type to the variable 'bblock' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'bblock', for_loop_var_336)
            # SSA begins for a for statement (line 259)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Attribute to a Subscript (line 260):
            # Getting the type of 'Havlak_loop_finder' (line 260)
            Havlak_loop_finder_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'Havlak_loop_finder')
            # Obtaining the member 'K_UNVISITED' of a type (line 260)
            K_UNVISITED_338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 29), Havlak_loop_finder_337, 'K_UNVISITED')
            # Getting the type of 'number' (line 260)
            number_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'number')
            # Getting the type of 'bblock' (line 260)
            bblock_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 'bblock')
            # Storing an element on a container (line 260)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 12), number_339, (bblock_340, K_UNVISITED_338))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to dfs(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'self' (line 262)
        self_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 31), 'self', False)
        # Obtaining the member 'cfg_' of a type (line 262)
        cfg__344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 31), self_343, 'cfg_')
        # Obtaining the member 'start_node_' of a type (line 262)
        start_node__345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 31), cfg__344, 'start_node_')
        # Getting the type of 'nodes' (line 262)
        nodes_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 54), 'nodes', False)
        # Getting the type of 'number' (line 262)
        number_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 61), 'number', False)
        # Getting the type of 'last' (line 262)
        last_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 69), 'last', False)
        int_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 75), 'int')
        # Processing the call keyword arguments (line 262)
        kwargs_350 = {}
        # Getting the type of 'Havlak_loop_finder' (line 262)
        Havlak_loop_finder_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'Havlak_loop_finder', False)
        # Obtaining the member 'dfs' of a type (line 262)
        dfs_342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), Havlak_loop_finder_341, 'dfs')
        # Calling dfs(args, kwargs) (line 262)
        dfs_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), dfs_342, *[start_node__345, nodes_346, number_347, last_348, int_349], **kwargs_350)
        
        
        
        # Call to xrange(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'size' (line 273)
        size_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'size', False)
        # Processing the call keyword arguments (line 273)
        kwargs_354 = {}
        # Getting the type of 'xrange' (line 273)
        xrange_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 273)
        xrange_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 273, 17), xrange_352, *[size_353], **kwargs_354)
        
        # Testing if the for loop is going to be iterated (line 273)
        # Testing the type of a for loop iterable (line 273)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 273, 8), xrange_call_result_355)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 273, 8), xrange_call_result_355):
            # Getting the type of the for loop variable (line 273)
            for_loop_var_356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 273, 8), xrange_call_result_355)
            # Assigning a type to the variable 'w' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'w', for_loop_var_356)
            # SSA begins for a for statement (line 273)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Subscript (line 274):
            int_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 24), 'int')
            # Getting the type of 'header' (line 274)
            header_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'header')
            # Getting the type of 'w' (line 274)
            w_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'w')
            # Storing an element on a container (line 274)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), header_358, (w_359, int_357))
            
            # Assigning a Attribute to a Subscript (line 275):
            # Getting the type of 'Basic_block_class' (line 275)
            Basic_block_class_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'Basic_block_class')
            # Obtaining the member 'NONHEADER' of a type (line 275)
            NONHEADER_361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 22), Basic_block_class_360, 'NONHEADER')
            # Getting the type of 'type' (line 275)
            type_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'type')
            # Getting the type of 'w' (line 275)
            w_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 17), 'w')
            # Storing an element on a container (line 275)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), type_362, (w_363, NONHEADER_361))
            
            # Assigning a Attribute to a Name (line 277):
            
            # Obtaining the type of the subscript
            # Getting the type of 'w' (line 277)
            w_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'w')
            # Getting the type of 'nodes' (line 277)
            nodes_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'nodes')
            # Obtaining the member '__getitem__' of a type (line 277)
            getitem___366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 21), nodes_365, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 277)
            subscript_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 277, 21), getitem___366, w_364)
            
            # Obtaining the member 'bb_' of a type (line 277)
            bb__368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 21), subscript_call_result_367, 'bb_')
            # Assigning a type to the variable 'node_w' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'node_w', bb__368)
            
            # Getting the type of 'node_w' (line 278)
            node_w_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'node_w')
            # Applying the 'not' unary operator (line 278)
            result_not__370 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 15), 'not', node_w_369)
            
            # Testing if the type of an if condition is none (line 278)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 278, 12), result_not__370):
                pass
            else:
                
                # Testing the type of an if condition (line 278)
                if_condition_371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 12), result_not__370)
                # Assigning a type to the variable 'if_condition_371' (line 278)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'if_condition_371', if_condition_371)
                # SSA begins for if statement (line 278)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Attribute to a Subscript (line 279):
                # Getting the type of 'Basic_block_class' (line 279)
                Basic_block_class_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 26), 'Basic_block_class')
                # Obtaining the member 'DEAD' of a type (line 279)
                DEAD_373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 26), Basic_block_class_372, 'DEAD')
                # Getting the type of 'type' (line 279)
                type_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'type')
                # Getting the type of 'w' (line 279)
                w_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 21), 'w')
                # Storing an element on a container (line 279)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 16), type_374, (w_375, DEAD_373))
                # SSA join for if statement (line 278)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to len(...): (line 282)
            # Processing the call arguments (line 282)
            # Getting the type of 'node_w' (line 282)
            node_w_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'node_w', False)
            # Obtaining the member 'in_edges_' of a type (line 282)
            in_edges__378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), node_w_377, 'in_edges_')
            # Processing the call keyword arguments (line 282)
            kwargs_379 = {}
            # Getting the type of 'len' (line 282)
            len_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'len', False)
            # Calling len(args, kwargs) (line 282)
            len_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), len_376, *[in_edges__378], **kwargs_379)
            
            # Testing if the type of an if condition is none (line 282)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 282, 12), len_call_result_380):
                pass
            else:
                
                # Testing the type of an if condition (line 282)
                if_condition_381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 12), len_call_result_380)
                # Assigning a type to the variable 'if_condition_381' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'if_condition_381', if_condition_381)
                # SSA begins for if statement (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'node_w' (line 283)
                node_w_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 30), 'node_w')
                # Obtaining the member 'in_edges_' of a type (line 283)
                in_edges__383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 30), node_w_382, 'in_edges_')
                # Testing if the for loop is going to be iterated (line 283)
                # Testing the type of a for loop iterable (line 283)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 16), in_edges__383)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 283, 16), in_edges__383):
                    # Getting the type of the for loop variable (line 283)
                    for_loop_var_384 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 16), in_edges__383)
                    # Assigning a type to the variable 'node_v' (line 283)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'node_v', for_loop_var_384)
                    # SSA begins for a for statement (line 283)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Subscript to a Name (line 284):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'node_v' (line 284)
                    node_v_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 31), 'node_v')
                    # Getting the type of 'number' (line 284)
                    number_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'number')
                    # Obtaining the member '__getitem__' of a type (line 284)
                    getitem___387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 24), number_386, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 284)
                    subscript_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 284, 24), getitem___387, node_v_385)
                    
                    # Assigning a type to the variable 'v' (line 284)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'v', subscript_call_result_388)
                    
                    # Getting the type of 'v' (line 285)
                    v_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 23), 'v')
                    # Getting the type of 'Havlak_loop_finder' (line 285)
                    Havlak_loop_finder_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 28), 'Havlak_loop_finder')
                    # Obtaining the member 'K_UNVISITED' of a type (line 285)
                    K_UNVISITED_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 28), Havlak_loop_finder_390, 'K_UNVISITED')
                    # Applying the binary operator '==' (line 285)
                    result_eq_392 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 23), '==', v_389, K_UNVISITED_391)
                    
                    # Testing if the type of an if condition is none (line 285)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 285, 20), result_eq_392):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 285)
                        if_condition_393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 20), result_eq_392)
                        # Assigning a type to the variable 'if_condition_393' (line 285)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'if_condition_393', if_condition_393)
                        # SSA begins for if statement (line 285)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # SSA join for if statement (line 285)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Call to is_ancestor(...): (line 288)
                    # Processing the call arguments (line 288)
                    # Getting the type of 'w' (line 288)
                    w_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 54), 'w', False)
                    # Getting the type of 'v' (line 288)
                    v_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 57), 'v', False)
                    # Getting the type of 'last' (line 288)
                    last_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 60), 'last', False)
                    # Processing the call keyword arguments (line 288)
                    kwargs_399 = {}
                    # Getting the type of 'Havlak_loop_finder' (line 288)
                    Havlak_loop_finder_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'Havlak_loop_finder', False)
                    # Obtaining the member 'is_ancestor' of a type (line 288)
                    is_ancestor_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 23), Havlak_loop_finder_394, 'is_ancestor')
                    # Calling is_ancestor(args, kwargs) (line 288)
                    is_ancestor_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 288, 23), is_ancestor_395, *[w_396, v_397, last_398], **kwargs_399)
                    
                    # Testing if the type of an if condition is none (line 288)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 288, 20), is_ancestor_call_result_400):
                        
                        # Call to add(...): (line 291)
                        # Processing the call arguments (line 291)
                        # Getting the type of 'v' (line 291)
                        v_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 46), 'v', False)
                        # Processing the call keyword arguments (line 291)
                        kwargs_416 = {}
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 291)
                        w_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 39), 'w', False)
                        # Getting the type of 'non_back_preds' (line 291)
                        non_back_preds_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'non_back_preds', False)
                        # Obtaining the member '__getitem__' of a type (line 291)
                        getitem___412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 24), non_back_preds_411, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
                        subscript_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 291, 24), getitem___412, w_410)
                        
                        # Obtaining the member 'add' of a type (line 291)
                        add_414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 24), subscript_call_result_413, 'add')
                        # Calling add(args, kwargs) (line 291)
                        add_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 291, 24), add_414, *[v_415], **kwargs_416)
                        
                    else:
                        
                        # Testing the type of an if condition (line 288)
                        if_condition_401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 20), is_ancestor_call_result_400)
                        # Assigning a type to the variable 'if_condition_401' (line 288)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 20), 'if_condition_401', if_condition_401)
                        # SSA begins for if statement (line 288)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 289)
                        # Processing the call arguments (line 289)
                        # Getting the type of 'v' (line 289)
                        v_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 45), 'v', False)
                        # Processing the call keyword arguments (line 289)
                        kwargs_408 = {}
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 289)
                        w_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'w', False)
                        # Getting the type of 'back_preds' (line 289)
                        back_preds_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 24), 'back_preds', False)
                        # Obtaining the member '__getitem__' of a type (line 289)
                        getitem___404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 24), back_preds_403, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                        subscript_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 289, 24), getitem___404, w_402)
                        
                        # Obtaining the member 'append' of a type (line 289)
                        append_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 24), subscript_call_result_405, 'append')
                        # Calling append(args, kwargs) (line 289)
                        append_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 289, 24), append_406, *[v_407], **kwargs_408)
                        
                        # SSA branch for the else part of an if statement (line 288)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to add(...): (line 291)
                        # Processing the call arguments (line 291)
                        # Getting the type of 'v' (line 291)
                        v_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 46), 'v', False)
                        # Processing the call keyword arguments (line 291)
                        kwargs_416 = {}
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 291)
                        w_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 39), 'w', False)
                        # Getting the type of 'non_back_preds' (line 291)
                        non_back_preds_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'non_back_preds', False)
                        # Obtaining the member '__getitem__' of a type (line 291)
                        getitem___412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 24), non_back_preds_411, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
                        subscript_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 291, 24), getitem___412, w_410)
                        
                        # Obtaining the member 'add' of a type (line 291)
                        add_414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 24), subscript_call_result_413, 'add')
                        # Calling add(args, kwargs) (line 291)
                        add_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 291, 24), add_414, *[v_415], **kwargs_416)
                        
                        # SSA join for if statement (line 288)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Num to a Subscript (line 294):
        int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'int')
        # Getting the type of 'header' (line 294)
        header_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'header')
        int_420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 15), 'int')
        # Storing an element on a container (line 294)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 8), header_419, (int_420, int_418))
        
        
        # Call to xrange(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'size' (line 307)
        size_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 24), 'size', False)
        int_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 31), 'int')
        # Applying the binary operator '-' (line 307)
        result_sub_424 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 24), '-', size_422, int_423)
        
        int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 34), 'int')
        int_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 38), 'int')
        # Processing the call keyword arguments (line 307)
        kwargs_427 = {}
        # Getting the type of 'xrange' (line 307)
        xrange_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 307)
        xrange_call_result_428 = invoke(stypy.reporting.localization.Localization(__file__, 307, 17), xrange_421, *[result_sub_424, int_425, int_426], **kwargs_427)
        
        # Testing if the for loop is going to be iterated (line 307)
        # Testing the type of a for loop iterable (line 307)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 307, 8), xrange_call_result_428)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 307, 8), xrange_call_result_428):
            # Getting the type of the for loop variable (line 307)
            for_loop_var_429 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 307, 8), xrange_call_result_428)
            # Assigning a type to the variable 'w' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'w', for_loop_var_429)
            # SSA begins for a for statement (line 307)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Name (line 308):
            
            # Obtaining an instance of the builtin type 'list' (line 308)
            list_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 308)
            
            # Assigning a type to the variable 'node_pool' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'node_pool', list_430)
            
            # Assigning a Attribute to a Name (line 309):
            
            # Obtaining the type of the subscript
            # Getting the type of 'w' (line 309)
            w_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 27), 'w')
            # Getting the type of 'nodes' (line 309)
            nodes_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 21), 'nodes')
            # Obtaining the member '__getitem__' of a type (line 309)
            getitem___433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 21), nodes_432, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 309)
            subscript_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 309, 21), getitem___433, w_431)
            
            # Obtaining the member 'bb_' of a type (line 309)
            bb__435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 21), subscript_call_result_434, 'bb_')
            # Assigning a type to the variable 'node_w' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'node_w', bb__435)
            
            # Getting the type of 'node_w' (line 310)
            node_w_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'node_w')
            # Applying the 'not' unary operator (line 310)
            result_not__437 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 15), 'not', node_w_436)
            
            # Testing if the type of an if condition is none (line 310)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 310, 12), result_not__437):
                pass
            else:
                
                # Testing the type of an if condition (line 310)
                if_condition_438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 12), result_not__437)
                # Assigning a type to the variable 'if_condition_438' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'if_condition_438', if_condition_438)
                # SSA begins for if statement (line 310)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 310)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Obtaining the type of the subscript
            # Getting the type of 'w' (line 314)
            w_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 40), 'w')
            # Getting the type of 'back_preds' (line 314)
            back_preds_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 29), 'back_preds')
            # Obtaining the member '__getitem__' of a type (line 314)
            getitem___441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 29), back_preds_440, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 314)
            subscript_call_result_442 = invoke(stypy.reporting.localization.Localization(__file__, 314, 29), getitem___441, w_439)
            
            # Testing if the for loop is going to be iterated (line 314)
            # Testing the type of a for loop iterable (line 314)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 314, 12), subscript_call_result_442)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 314, 12), subscript_call_result_442):
                # Getting the type of the for loop variable (line 314)
                for_loop_var_443 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 314, 12), subscript_call_result_442)
                # Assigning a type to the variable 'back_pred' (line 314)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'back_pred', for_loop_var_443)
                # SSA begins for a for statement (line 314)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'back_pred' (line 315)
                back_pred_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'back_pred')
                # Getting the type of 'w' (line 315)
                w_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 32), 'w')
                # Applying the binary operator '!=' (line 315)
                result_ne_446 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 19), '!=', back_pred_444, w_445)
                
                # Testing if the type of an if condition is none (line 315)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 315, 16), result_ne_446):
                    
                    # Assigning a Attribute to a Subscript (line 318):
                    # Getting the type of 'Basic_block_class' (line 318)
                    Basic_block_class_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'Basic_block_class')
                    # Obtaining the member 'SELF' of a type (line 318)
                    SELF_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 30), Basic_block_class_459, 'SELF')
                    # Getting the type of 'type' (line 318)
                    type_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'type')
                    # Getting the type of 'w' (line 318)
                    w_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'w')
                    # Storing an element on a container (line 318)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 20), type_461, (w_462, SELF_460))
                else:
                    
                    # Testing the type of an if condition (line 315)
                    if_condition_447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 16), result_ne_446)
                    # Assigning a type to the variable 'if_condition_447' (line 315)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'if_condition_447', if_condition_447)
                    # SSA begins for if statement (line 315)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 316)
                    # Processing the call arguments (line 316)
                    
                    # Call to find_set(...): (line 316)
                    # Processing the call keyword arguments (line 316)
                    kwargs_455 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'back_pred' (line 316)
                    back_pred_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 43), 'back_pred', False)
                    # Getting the type of 'nodes' (line 316)
                    nodes_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 37), 'nodes', False)
                    # Obtaining the member '__getitem__' of a type (line 316)
                    getitem___452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 37), nodes_451, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
                    subscript_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 316, 37), getitem___452, back_pred_450)
                    
                    # Obtaining the member 'find_set' of a type (line 316)
                    find_set_454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 37), subscript_call_result_453, 'find_set')
                    # Calling find_set(args, kwargs) (line 316)
                    find_set_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 316, 37), find_set_454, *[], **kwargs_455)
                    
                    # Processing the call keyword arguments (line 316)
                    kwargs_457 = {}
                    # Getting the type of 'node_pool' (line 316)
                    node_pool_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'node_pool', False)
                    # Obtaining the member 'append' of a type (line 316)
                    append_449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 20), node_pool_448, 'append')
                    # Calling append(args, kwargs) (line 316)
                    append_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 316, 20), append_449, *[find_set_call_result_456], **kwargs_457)
                    
                    # SSA branch for the else part of an if statement (line 315)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Attribute to a Subscript (line 318):
                    # Getting the type of 'Basic_block_class' (line 318)
                    Basic_block_class_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'Basic_block_class')
                    # Obtaining the member 'SELF' of a type (line 318)
                    SELF_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 30), Basic_block_class_459, 'SELF')
                    # Getting the type of 'type' (line 318)
                    type_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'type')
                    # Getting the type of 'w' (line 318)
                    w_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'w')
                    # Storing an element on a container (line 318)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 20), type_461, (w_462, SELF_460))
                    # SSA join for if statement (line 315)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a List to a Name (line 321):
            
            # Obtaining an instance of the builtin type 'list' (line 321)
            list_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 321)
            
            # Assigning a type to the variable 'worklist' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'worklist', list_463)
            
            # Getting the type of 'node_pool' (line 322)
            node_pool_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 22), 'node_pool')
            # Testing if the for loop is going to be iterated (line 322)
            # Testing the type of a for loop iterable (line 322)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 322, 12), node_pool_464)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 322, 12), node_pool_464):
                # Getting the type of the for loop variable (line 322)
                for_loop_var_465 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 322, 12), node_pool_464)
                # Assigning a type to the variable 'np' (line 322)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'np', for_loop_var_465)
                # SSA begins for a for statement (line 322)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 323)
                # Processing the call arguments (line 323)
                # Getting the type of 'np' (line 323)
                np_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 32), 'np', False)
                # Processing the call keyword arguments (line 323)
                kwargs_469 = {}
                # Getting the type of 'worklist' (line 323)
                worklist_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'worklist', False)
                # Obtaining the member 'append' of a type (line 323)
                append_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 16), worklist_466, 'append')
                # Calling append(args, kwargs) (line 323)
                append_call_result_470 = invoke(stypy.reporting.localization.Localization(__file__, 323, 16), append_467, *[np_468], **kwargs_469)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to len(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'node_pool' (line 325)
            node_pool_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'node_pool', False)
            # Processing the call keyword arguments (line 325)
            kwargs_473 = {}
            # Getting the type of 'len' (line 325)
            len_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'len', False)
            # Calling len(args, kwargs) (line 325)
            len_call_result_474 = invoke(stypy.reporting.localization.Localization(__file__, 325, 15), len_471, *[node_pool_472], **kwargs_473)
            
            # Testing if the type of an if condition is none (line 325)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 325, 12), len_call_result_474):
                pass
            else:
                
                # Testing the type of an if condition (line 325)
                if_condition_475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 12), len_call_result_474)
                # Assigning a type to the variable 'if_condition_475' (line 325)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'if_condition_475', if_condition_475)
                # SSA begins for if statement (line 325)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Attribute to a Subscript (line 326):
                # Getting the type of 'Basic_block_class' (line 326)
                Basic_block_class_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), 'Basic_block_class')
                # Obtaining the member 'REDUCIBLE' of a type (line 326)
                REDUCIBLE_477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 26), Basic_block_class_476, 'REDUCIBLE')
                # Getting the type of 'type' (line 326)
                type_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'type')
                # Getting the type of 'w' (line 326)
                w_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), 'w')
                # Storing an element on a container (line 326)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 16), type_478, (w_479, REDUCIBLE_477))
                # SSA join for if statement (line 325)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to len(...): (line 330)
            # Processing the call arguments (line 330)
            # Getting the type of 'worklist' (line 330)
            worklist_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'worklist', False)
            # Processing the call keyword arguments (line 330)
            kwargs_482 = {}
            # Getting the type of 'len' (line 330)
            len_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 18), 'len', False)
            # Calling len(args, kwargs) (line 330)
            len_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 330, 18), len_480, *[worklist_481], **kwargs_482)
            
            # Testing if the while is going to be iterated (line 330)
            # Testing the type of an if condition (line 330)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 12), len_call_result_483)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 330, 12), len_call_result_483):
                # SSA begins for while statement (line 330)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a Subscript to a Name (line 331):
                
                # Obtaining the type of the subscript
                int_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 29), 'int')
                # Getting the type of 'worklist' (line 331)
                worklist_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'worklist')
                # Obtaining the member '__getitem__' of a type (line 331)
                getitem___486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 20), worklist_485, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 331)
                subscript_call_result_487 = invoke(stypy.reporting.localization.Localization(__file__, 331, 20), getitem___486, int_484)
                
                # Assigning a type to the variable 'x' (line 331)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'x', subscript_call_result_487)
                
                # Assigning a Subscript to a Name (line 332):
                
                # Obtaining the type of the subscript
                int_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 36), 'int')
                slice_489 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 332, 27), int_488, None, None)
                # Getting the type of 'worklist' (line 332)
                worklist_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 27), 'worklist')
                # Obtaining the member '__getitem__' of a type (line 332)
                getitem___491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 27), worklist_490, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 332)
                subscript_call_result_492 = invoke(stypy.reporting.localization.Localization(__file__, 332, 27), getitem___491, slice_489)
                
                # Assigning a type to the variable 'worklist' (line 332)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'worklist', subscript_call_result_492)
                
                # Assigning a Call to a Name (line 344):
                
                # Call to len(...): (line 344)
                # Processing the call arguments (line 344)
                
                # Obtaining the type of the subscript
                # Getting the type of 'x' (line 344)
                x_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 51), 'x', False)
                # Obtaining the member 'dfs_number_' of a type (line 344)
                dfs_number__495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 51), x_494, 'dfs_number_')
                # Getting the type of 'non_back_preds' (line 344)
                non_back_preds_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 36), 'non_back_preds', False)
                # Obtaining the member '__getitem__' of a type (line 344)
                getitem___497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 36), non_back_preds_496, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 344)
                subscript_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 344, 36), getitem___497, dfs_number__495)
                
                # Processing the call keyword arguments (line 344)
                kwargs_499 = {}
                # Getting the type of 'len' (line 344)
                len_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'len', False)
                # Calling len(args, kwargs) (line 344)
                len_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 344, 32), len_493, *[subscript_call_result_498], **kwargs_499)
                
                # Assigning a type to the variable 'non_back_size' (line 344)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'non_back_size', len_call_result_500)
                
                # Getting the type of 'non_back_size' (line 345)
                non_back_size_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'non_back_size')
                # Getting the type of 'Havlak_loop_finder' (line 345)
                Havlak_loop_finder_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 35), 'Havlak_loop_finder')
                # Obtaining the member 'K_MAX_NON_BACK_PREDS' of a type (line 345)
                K_MAX_NON_BACK_PREDS_503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 35), Havlak_loop_finder_502, 'K_MAX_NON_BACK_PREDS')
                # Applying the binary operator '>' (line 345)
                result_gt_504 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 19), '>', non_back_size_501, K_MAX_NON_BACK_PREDS_503)
                
                # Testing if the type of an if condition is none (line 345)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 345, 16), result_gt_504):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 345)
                    if_condition_505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 345, 16), result_gt_504)
                    # Assigning a type to the variable 'if_condition_505' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'if_condition_505', if_condition_505)
                    # SSA begins for if statement (line 345)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Assigning a type to the variable 'stypy_return_type' (line 346)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'stypy_return_type', types.NoneType)
                    # SSA join for if statement (line 345)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                
                # Obtaining the type of the subscript
                # Getting the type of 'x' (line 348)
                x_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 57), 'x')
                # Obtaining the member 'dfs_number_' of a type (line 348)
                dfs_number__507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 57), x_506, 'dfs_number_')
                # Getting the type of 'non_back_preds' (line 348)
                non_back_preds_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 42), 'non_back_preds')
                # Obtaining the member '__getitem__' of a type (line 348)
                getitem___509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 42), non_back_preds_508, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 348)
                subscript_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 348, 42), getitem___509, dfs_number__507)
                
                # Testing if the for loop is going to be iterated (line 348)
                # Testing the type of a for loop iterable (line 348)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 348, 16), subscript_call_result_510)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 348, 16), subscript_call_result_510):
                    # Getting the type of the for loop variable (line 348)
                    for_loop_var_511 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 348, 16), subscript_call_result_510)
                    # Assigning a type to the variable 'non_back_pred_iter' (line 348)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'non_back_pred_iter', for_loop_var_511)
                    # SSA begins for a for statement (line 348)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Subscript to a Name (line 349):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'non_back_pred_iter' (line 349)
                    non_back_pred_iter_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 30), 'non_back_pred_iter')
                    # Getting the type of 'nodes' (line 349)
                    nodes_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'nodes')
                    # Obtaining the member '__getitem__' of a type (line 349)
                    getitem___514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 24), nodes_513, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
                    subscript_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 349, 24), getitem___514, non_back_pred_iter_512)
                    
                    # Assigning a type to the variable 'y' (line 349)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 20), 'y', subscript_call_result_515)
                    
                    # Assigning a Call to a Name (line 350):
                    
                    # Call to find_set(...): (line 350)
                    # Processing the call keyword arguments (line 350)
                    kwargs_518 = {}
                    # Getting the type of 'y' (line 350)
                    y_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'y', False)
                    # Obtaining the member 'find_set' of a type (line 350)
                    find_set_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 28), y_516, 'find_set')
                    # Calling find_set(args, kwargs) (line 350)
                    find_set_call_result_519 = invoke(stypy.reporting.localization.Localization(__file__, 350, 28), find_set_517, *[], **kwargs_518)
                    
                    # Assigning a type to the variable 'ydash' (line 350)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 20), 'ydash', find_set_call_result_519)
                    
                    
                    # Call to is_ancestor(...): (line 352)
                    # Processing the call arguments (line 352)
                    # Getting the type of 'w' (line 352)
                    w_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 58), 'w', False)
                    # Getting the type of 'ydash' (line 352)
                    ydash_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 61), 'ydash', False)
                    # Obtaining the member 'dfs_number_' of a type (line 352)
                    dfs_number__524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 61), ydash_523, 'dfs_number_')
                    # Getting the type of 'last' (line 352)
                    last_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 80), 'last', False)
                    # Processing the call keyword arguments (line 352)
                    kwargs_526 = {}
                    # Getting the type of 'Havlak_loop_finder' (line 352)
                    Havlak_loop_finder_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'Havlak_loop_finder', False)
                    # Obtaining the member 'is_ancestor' of a type (line 352)
                    is_ancestor_521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 27), Havlak_loop_finder_520, 'is_ancestor')
                    # Calling is_ancestor(args, kwargs) (line 352)
                    is_ancestor_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 352, 27), is_ancestor_521, *[w_522, dfs_number__524, last_525], **kwargs_526)
                    
                    # Applying the 'not' unary operator (line 352)
                    result_not__528 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 23), 'not', is_ancestor_call_result_527)
                    
                    # Testing if the type of an if condition is none (line 352)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 352, 20), result_not__528):
                        
                        # Getting the type of 'ydash' (line 356)
                        ydash_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'ydash')
                        # Obtaining the member 'dfs_number_' of a type (line 356)
                        dfs_number__544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 27), ydash_543, 'dfs_number_')
                        # Getting the type of 'w' (line 356)
                        w_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 48), 'w')
                        # Applying the binary operator '!=' (line 356)
                        result_ne_546 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 27), '!=', dfs_number__544, w_545)
                        
                        # Testing if the type of an if condition is none (line 356)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 356, 24), result_ne_546):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 356)
                            if_condition_547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 24), result_ne_546)
                            # Assigning a type to the variable 'if_condition_547' (line 356)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'if_condition_547', if_condition_547)
                            # SSA begins for if statement (line 356)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'ydash' (line 357)
                            ydash_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'ydash')
                            # Getting the type of 'node_pool' (line 357)
                            node_pool_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 44), 'node_pool')
                            # Applying the binary operator 'notin' (line 357)
                            result_contains_550 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 31), 'notin', ydash_548, node_pool_549)
                            
                            # Testing if the type of an if condition is none (line 357)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 357, 28), result_contains_550):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 357)
                                if_condition_551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 28), result_contains_550)
                                # Assigning a type to the variable 'if_condition_551' (line 357)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'if_condition_551', if_condition_551)
                                # SSA begins for if statement (line 357)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 358)
                                # Processing the call arguments (line 358)
                                # Getting the type of 'ydash' (line 358)
                                ydash_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 48), 'ydash', False)
                                # Processing the call keyword arguments (line 358)
                                kwargs_555 = {}
                                # Getting the type of 'worklist' (line 358)
                                worklist_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 32), 'worklist', False)
                                # Obtaining the member 'append' of a type (line 358)
                                append_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 32), worklist_552, 'append')
                                # Calling append(args, kwargs) (line 358)
                                append_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 358, 32), append_553, *[ydash_554], **kwargs_555)
                                
                                
                                # Call to append(...): (line 359)
                                # Processing the call arguments (line 359)
                                # Getting the type of 'ydash' (line 359)
                                ydash_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 49), 'ydash', False)
                                # Processing the call keyword arguments (line 359)
                                kwargs_560 = {}
                                # Getting the type of 'node_pool' (line 359)
                                node_pool_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'node_pool', False)
                                # Obtaining the member 'append' of a type (line 359)
                                append_558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 32), node_pool_557, 'append')
                                # Calling append(args, kwargs) (line 359)
                                append_call_result_561 = invoke(stypy.reporting.localization.Localization(__file__, 359, 32), append_558, *[ydash_559], **kwargs_560)
                                
                                # SSA join for if statement (line 357)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 356)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 352)
                        if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 20), result_not__528)
                        # Assigning a type to the variable 'if_condition_529' (line 352)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 20), 'if_condition_529', if_condition_529)
                        # SSA begins for if statement (line 352)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Attribute to a Subscript (line 353):
                        # Getting the type of 'Basic_block_class' (line 353)
                        Basic_block_class_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 'Basic_block_class')
                        # Obtaining the member 'IRREDUCIBLE' of a type (line 353)
                        IRREDUCIBLE_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 34), Basic_block_class_530, 'IRREDUCIBLE')
                        # Getting the type of 'type' (line 353)
                        type_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 'type')
                        # Getting the type of 'w' (line 353)
                        w_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 29), 'w')
                        # Storing an element on a container (line 353)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 24), type_532, (w_533, IRREDUCIBLE_531))
                        
                        # Call to add(...): (line 354)
                        # Processing the call arguments (line 354)
                        # Getting the type of 'ydash' (line 354)
                        ydash_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 46), 'ydash', False)
                        # Obtaining the member 'dfs_number_' of a type (line 354)
                        dfs_number__540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 46), ydash_539, 'dfs_number_')
                        # Processing the call keyword arguments (line 354)
                        kwargs_541 = {}
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 354)
                        w_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 39), 'w', False)
                        # Getting the type of 'non_back_preds' (line 354)
                        non_back_preds_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'non_back_preds', False)
                        # Obtaining the member '__getitem__' of a type (line 354)
                        getitem___536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 24), non_back_preds_535, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
                        subscript_call_result_537 = invoke(stypy.reporting.localization.Localization(__file__, 354, 24), getitem___536, w_534)
                        
                        # Obtaining the member 'add' of a type (line 354)
                        add_538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 24), subscript_call_result_537, 'add')
                        # Calling add(args, kwargs) (line 354)
                        add_call_result_542 = invoke(stypy.reporting.localization.Localization(__file__, 354, 24), add_538, *[dfs_number__540], **kwargs_541)
                        
                        # SSA branch for the else part of an if statement (line 352)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'ydash' (line 356)
                        ydash_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'ydash')
                        # Obtaining the member 'dfs_number_' of a type (line 356)
                        dfs_number__544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 27), ydash_543, 'dfs_number_')
                        # Getting the type of 'w' (line 356)
                        w_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 48), 'w')
                        # Applying the binary operator '!=' (line 356)
                        result_ne_546 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 27), '!=', dfs_number__544, w_545)
                        
                        # Testing if the type of an if condition is none (line 356)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 356, 24), result_ne_546):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 356)
                            if_condition_547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 24), result_ne_546)
                            # Assigning a type to the variable 'if_condition_547' (line 356)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'if_condition_547', if_condition_547)
                            # SSA begins for if statement (line 356)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'ydash' (line 357)
                            ydash_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'ydash')
                            # Getting the type of 'node_pool' (line 357)
                            node_pool_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 44), 'node_pool')
                            # Applying the binary operator 'notin' (line 357)
                            result_contains_550 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 31), 'notin', ydash_548, node_pool_549)
                            
                            # Testing if the type of an if condition is none (line 357)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 357, 28), result_contains_550):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 357)
                                if_condition_551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 28), result_contains_550)
                                # Assigning a type to the variable 'if_condition_551' (line 357)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'if_condition_551', if_condition_551)
                                # SSA begins for if statement (line 357)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 358)
                                # Processing the call arguments (line 358)
                                # Getting the type of 'ydash' (line 358)
                                ydash_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 48), 'ydash', False)
                                # Processing the call keyword arguments (line 358)
                                kwargs_555 = {}
                                # Getting the type of 'worklist' (line 358)
                                worklist_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 32), 'worklist', False)
                                # Obtaining the member 'append' of a type (line 358)
                                append_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 32), worklist_552, 'append')
                                # Calling append(args, kwargs) (line 358)
                                append_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 358, 32), append_553, *[ydash_554], **kwargs_555)
                                
                                
                                # Call to append(...): (line 359)
                                # Processing the call arguments (line 359)
                                # Getting the type of 'ydash' (line 359)
                                ydash_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 49), 'ydash', False)
                                # Processing the call keyword arguments (line 359)
                                kwargs_560 = {}
                                # Getting the type of 'node_pool' (line 359)
                                node_pool_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'node_pool', False)
                                # Obtaining the member 'append' of a type (line 359)
                                append_558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 32), node_pool_557, 'append')
                                # Calling append(args, kwargs) (line 359)
                                append_call_result_561 = invoke(stypy.reporting.localization.Localization(__file__, 359, 32), append_558, *[ydash_559], **kwargs_560)
                                
                                # SSA join for if statement (line 357)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 356)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 352)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for while statement (line 330)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Evaluating a boolean operation
            
            # Call to len(...): (line 364)
            # Processing the call arguments (line 364)
            # Getting the type of 'node_pool' (line 364)
            node_pool_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 19), 'node_pool', False)
            # Processing the call keyword arguments (line 364)
            kwargs_564 = {}
            # Getting the type of 'len' (line 364)
            len_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'len', False)
            # Calling len(args, kwargs) (line 364)
            len_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), len_562, *[node_pool_563], **kwargs_564)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'w' (line 364)
            w_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 38), 'w')
            # Getting the type of 'type' (line 364)
            type_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 33), 'type')
            # Obtaining the member '__getitem__' of a type (line 364)
            getitem___568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 33), type_567, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 364)
            subscript_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 364, 33), getitem___568, w_566)
            
            # Getting the type of 'Basic_block_class' (line 364)
            Basic_block_class_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 44), 'Basic_block_class')
            # Obtaining the member 'SELF' of a type (line 364)
            SELF_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 44), Basic_block_class_570, 'SELF')
            # Applying the binary operator '==' (line 364)
            result_eq_572 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 33), '==', subscript_call_result_569, SELF_571)
            
            # Applying the binary operator 'or' (line 364)
            result_or_keyword_573 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 15), 'or', len_call_result_565, result_eq_572)
            
            # Testing if the type of an if condition is none (line 364)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 364, 12), result_or_keyword_573):
                pass
            else:
                
                # Testing the type of an if condition (line 364)
                if_condition_574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 12), result_or_keyword_573)
                # Assigning a type to the variable 'if_condition_574' (line 364)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'if_condition_574', if_condition_574)
                # SSA begins for if statement (line 364)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 365):
                
                # Call to create_new_loop(...): (line 365)
                # Processing the call keyword arguments (line 365)
                kwargs_578 = {}
                # Getting the type of 'self' (line 365)
                self_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 23), 'self', False)
                # Obtaining the member 'lsg_' of a type (line 365)
                lsg__576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 23), self_575, 'lsg_')
                # Obtaining the member 'create_new_loop' of a type (line 365)
                create_new_loop_577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 23), lsg__576, 'create_new_loop')
                # Calling create_new_loop(args, kwargs) (line 365)
                create_new_loop_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 365, 23), create_new_loop_577, *[], **kwargs_578)
                
                # Assigning a type to the variable 'loop' (line 365)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'loop', create_new_loop_call_result_579)
                
                # Assigning a Name to a Attribute (line 381):
                # Getting the type of 'loop' (line 381)
                loop_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 33), 'loop')
                
                # Obtaining the type of the subscript
                # Getting the type of 'w' (line 381)
                w_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 22), 'w')
                # Getting the type of 'nodes' (line 381)
                nodes_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'nodes')
                # Obtaining the member '__getitem__' of a type (line 381)
                getitem___583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 16), nodes_582, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 381)
                subscript_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 381, 16), getitem___583, w_581)
                
                # Setting the type of the member 'loop_' of a type (line 381)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 16), subscript_call_result_584, 'loop_', loop_580)
                
                # Getting the type of 'node_pool' (line 383)
                node_pool_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 28), 'node_pool')
                # Testing if the for loop is going to be iterated (line 383)
                # Testing the type of a for loop iterable (line 383)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 16), node_pool_585)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 383, 16), node_pool_585):
                    # Getting the type of the for loop variable (line 383)
                    for_loop_var_586 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 16), node_pool_585)
                    # Assigning a type to the variable 'node' (line 383)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'node', for_loop_var_586)
                    # SSA begins for a for statement (line 383)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Name to a Subscript (line 385):
                    # Getting the type of 'w' (line 385)
                    w_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 47), 'w')
                    # Getting the type of 'header' (line 385)
                    header_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'header')
                    # Getting the type of 'node' (line 385)
                    node_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 27), 'node')
                    # Obtaining the member 'dfs_number_' of a type (line 385)
                    dfs_number__590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 27), node_589, 'dfs_number_')
                    # Storing an element on a container (line 385)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 20), header_588, (dfs_number__590, w_587))
                    
                    # Call to do_union(...): (line 386)
                    # Processing the call arguments (line 386)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'w' (line 386)
                    w_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 40), 'w', False)
                    # Getting the type of 'nodes' (line 386)
                    nodes_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 34), 'nodes', False)
                    # Obtaining the member '__getitem__' of a type (line 386)
                    getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 34), nodes_594, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
                    subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 386, 34), getitem___595, w_593)
                    
                    # Processing the call keyword arguments (line 386)
                    kwargs_597 = {}
                    # Getting the type of 'node' (line 386)
                    node_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 20), 'node', False)
                    # Obtaining the member 'do_union' of a type (line 386)
                    do_union_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 20), node_591, 'do_union')
                    # Calling do_union(args, kwargs) (line 386)
                    do_union_call_result_598 = invoke(stypy.reporting.localization.Localization(__file__, 386, 20), do_union_592, *[subscript_call_result_596], **kwargs_597)
                    
                    # Getting the type of 'node' (line 389)
                    node_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 23), 'node')
                    # Obtaining the member 'loop_' of a type (line 389)
                    loop__600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 23), node_599, 'loop_')
                    # Testing if the type of an if condition is none (line 389)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 389, 20), loop__600):
                        
                        # Call to add_node(...): (line 392)
                        # Processing the call arguments (line 392)
                        # Getting the type of 'node' (line 392)
                        node_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 38), 'node', False)
                        # Obtaining the member 'bb_' of a type (line 392)
                        bb__608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 38), node_607, 'bb_')
                        # Processing the call keyword arguments (line 392)
                        kwargs_609 = {}
                        # Getting the type of 'loop' (line 392)
                        loop_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 24), 'loop', False)
                        # Obtaining the member 'add_node' of a type (line 392)
                        add_node_606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 24), loop_605, 'add_node')
                        # Calling add_node(args, kwargs) (line 392)
                        add_node_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 392, 24), add_node_606, *[bb__608], **kwargs_609)
                        
                    else:
                        
                        # Testing the type of an if condition (line 389)
                        if_condition_601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 20), loop__600)
                        # Assigning a type to the variable 'if_condition_601' (line 389)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'if_condition_601', if_condition_601)
                        # SSA begins for if statement (line 389)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 390):
                        # Getting the type of 'loop' (line 390)
                        loop_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 45), 'loop')
                        # Getting the type of 'node' (line 390)
                        node_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'node')
                        # Obtaining the member 'loop_' of a type (line 390)
                        loop__604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 24), node_603, 'loop_')
                        # Setting the type of the member 'parent_' of a type (line 390)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 24), loop__604, 'parent_', loop_602)
                        # SSA branch for the else part of an if statement (line 389)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to add_node(...): (line 392)
                        # Processing the call arguments (line 392)
                        # Getting the type of 'node' (line 392)
                        node_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 38), 'node', False)
                        # Obtaining the member 'bb_' of a type (line 392)
                        bb__608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 38), node_607, 'bb_')
                        # Processing the call keyword arguments (line 392)
                        kwargs_609 = {}
                        # Getting the type of 'loop' (line 392)
                        loop_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 24), 'loop', False)
                        # Obtaining the member 'add_node' of a type (line 392)
                        add_node_606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 24), loop_605, 'add_node')
                        # Calling add_node(args, kwargs) (line 392)
                        add_node_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 392, 24), add_node_606, *[bb__608], **kwargs_609)
                        
                        # SSA join for if statement (line 389)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Call to append(...): (line 394)
                # Processing the call arguments (line 394)
                # Getting the type of 'loop' (line 394)
                loop_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 40), 'loop', False)
                # Processing the call keyword arguments (line 394)
                kwargs_616 = {}
                # Getting the type of 'self' (line 394)
                self_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'self', False)
                # Obtaining the member 'lsg_' of a type (line 394)
                lsg__612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), self_611, 'lsg_')
                # Obtaining the member 'loops_' of a type (line 394)
                loops__613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), lsg__612, 'loops_')
                # Obtaining the member 'append' of a type (line 394)
                append_614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), loops__613, 'append')
                # Calling append(args, kwargs) (line 394)
                append_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 394, 16), append_614, *[loop_615], **kwargs_616)
                
                # SSA join for if statement (line 364)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'find_loops(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_loops' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_loops'
        return stypy_return_type_618


# Assigning a type to the variable 'Havlak_loop_finder' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'Havlak_loop_finder', Havlak_loop_finder)

# Assigning a Num to a Name (line 201):
int_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 18), 'int')
# Getting the type of 'Havlak_loop_finder'
Havlak_loop_finder_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Havlak_loop_finder')
# Setting the type of the member 'K_UNVISITED' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Havlak_loop_finder_620, 'K_UNVISITED', int_619)

# Assigning a BinOp to a Name (line 204):
int_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 27), 'int')
int_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 32), 'int')
# Applying the binary operator '*' (line 204)
result_mul_623 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 27), '*', int_621, int_622)

# Getting the type of 'Havlak_loop_finder'
Havlak_loop_finder_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Havlak_loop_finder')
# Setting the type of the member 'K_MAX_NON_BACK_PREDS' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Havlak_loop_finder_624, 'K_MAX_NON_BACK_PREDS', result_mul_623)

@norecursion
def find_havlak_loops(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_havlak_loops'
    module_type_store = module_type_store.open_function_context('find_havlak_loops', 397, 0, False)
    
    # Passed parameters checking function
    find_havlak_loops.stypy_localization = localization
    find_havlak_loops.stypy_type_of_self = None
    find_havlak_loops.stypy_type_store = module_type_store
    find_havlak_loops.stypy_function_name = 'find_havlak_loops'
    find_havlak_loops.stypy_param_names_list = ['cfg', 'lsg']
    find_havlak_loops.stypy_varargs_param_name = None
    find_havlak_loops.stypy_kwargs_param_name = None
    find_havlak_loops.stypy_call_defaults = defaults
    find_havlak_loops.stypy_call_varargs = varargs
    find_havlak_loops.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_havlak_loops', ['cfg', 'lsg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_havlak_loops', localization, ['cfg', 'lsg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_havlak_loops(...)' code ##################

    str_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 4), 'str', 'External entry point.')
    
    # Assigning a Call to a Name (line 399):
    
    # Call to Havlak_loop_finder(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'cfg' (line 399)
    cfg_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 32), 'cfg', False)
    # Getting the type of 'lsg' (line 399)
    lsg_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 37), 'lsg', False)
    # Processing the call keyword arguments (line 399)
    kwargs_629 = {}
    # Getting the type of 'Havlak_loop_finder' (line 399)
    Havlak_loop_finder_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 13), 'Havlak_loop_finder', False)
    # Calling Havlak_loop_finder(args, kwargs) (line 399)
    Havlak_loop_finder_call_result_630 = invoke(stypy.reporting.localization.Localization(__file__, 399, 13), Havlak_loop_finder_626, *[cfg_627, lsg_628], **kwargs_629)
    
    # Assigning a type to the variable 'finder' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'finder', Havlak_loop_finder_call_result_630)
    
    # Call to find_loops(...): (line 400)
    # Processing the call keyword arguments (line 400)
    kwargs_633 = {}
    # Getting the type of 'finder' (line 400)
    finder_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'finder', False)
    # Obtaining the member 'find_loops' of a type (line 400)
    find_loops_632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 4), finder_631, 'find_loops')
    # Calling find_loops(args, kwargs) (line 400)
    find_loops_call_result_634 = invoke(stypy.reporting.localization.Localization(__file__, 400, 4), find_loops_632, *[], **kwargs_633)
    
    
    # Call to len(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'lsg' (line 401)
    lsg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'lsg', False)
    # Obtaining the member 'loops_' of a type (line 401)
    loops__637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 15), lsg_636, 'loops_')
    # Processing the call keyword arguments (line 401)
    kwargs_638 = {}
    # Getting the type of 'len' (line 401)
    len_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 11), 'len', False)
    # Calling len(args, kwargs) (line 401)
    len_call_result_639 = invoke(stypy.reporting.localization.Localization(__file__, 401, 11), len_635, *[loops__637], **kwargs_638)
    
    # Assigning a type to the variable 'stypy_return_type' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type', len_call_result_639)
    
    # ################# End of 'find_havlak_loops(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_havlak_loops' in the type store
    # Getting the type of 'stypy_return_type' (line 397)
    stypy_return_type_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_havlak_loops'
    return stypy_return_type_640

# Assigning a type to the variable 'find_havlak_loops' (line 397)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), 'find_havlak_loops', find_havlak_loops)

@norecursion
def build_diamond(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'build_diamond'
    module_type_store = module_type_store.open_function_context('build_diamond', 404, 0, False)
    
    # Passed parameters checking function
    build_diamond.stypy_localization = localization
    build_diamond.stypy_type_of_self = None
    build_diamond.stypy_type_store = module_type_store
    build_diamond.stypy_function_name = 'build_diamond'
    build_diamond.stypy_param_names_list = ['cfg', 'start']
    build_diamond.stypy_varargs_param_name = None
    build_diamond.stypy_kwargs_param_name = None
    build_diamond.stypy_call_defaults = defaults
    build_diamond.stypy_call_varargs = varargs
    build_diamond.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'build_diamond', ['cfg', 'start'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'build_diamond', localization, ['cfg', 'start'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'build_diamond(...)' code ##################

    
    # Assigning a Name to a Name (line 405):
    # Getting the type of 'start' (line 405)
    start_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 10), 'start')
    # Assigning a type to the variable 'bb0' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'bb0', start_641)
    
    # Call to Basic_block_edge(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'cfg' (line 406)
    cfg_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'cfg', False)
    # Getting the type of 'bb0' (line 406)
    bb0_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'bb0', False)
    # Getting the type of 'bb0' (line 406)
    bb0_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 31), 'bb0', False)
    int_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 37), 'int')
    # Applying the binary operator '+' (line 406)
    result_add_647 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 31), '+', bb0_645, int_646)
    
    # Processing the call keyword arguments (line 406)
    kwargs_648 = {}
    # Getting the type of 'Basic_block_edge' (line 406)
    Basic_block_edge_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'Basic_block_edge', False)
    # Calling Basic_block_edge(args, kwargs) (line 406)
    Basic_block_edge_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), Basic_block_edge_642, *[cfg_643, bb0_644, result_add_647], **kwargs_648)
    
    
    # Call to Basic_block_edge(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'cfg' (line 407)
    cfg_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'cfg', False)
    # Getting the type of 'bb0' (line 407)
    bb0_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 26), 'bb0', False)
    # Getting the type of 'bb0' (line 407)
    bb0_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 31), 'bb0', False)
    int_654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 37), 'int')
    # Applying the binary operator '+' (line 407)
    result_add_655 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 31), '+', bb0_653, int_654)
    
    # Processing the call keyword arguments (line 407)
    kwargs_656 = {}
    # Getting the type of 'Basic_block_edge' (line 407)
    Basic_block_edge_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'Basic_block_edge', False)
    # Calling Basic_block_edge(args, kwargs) (line 407)
    Basic_block_edge_call_result_657 = invoke(stypy.reporting.localization.Localization(__file__, 407, 4), Basic_block_edge_650, *[cfg_651, bb0_652, result_add_655], **kwargs_656)
    
    
    # Call to Basic_block_edge(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'cfg' (line 408)
    cfg_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 21), 'cfg', False)
    # Getting the type of 'bb0' (line 408)
    bb0_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 26), 'bb0', False)
    int_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 32), 'int')
    # Applying the binary operator '+' (line 408)
    result_add_662 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 26), '+', bb0_660, int_661)
    
    # Getting the type of 'bb0' (line 408)
    bb0_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 35), 'bb0', False)
    int_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 41), 'int')
    # Applying the binary operator '+' (line 408)
    result_add_665 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 35), '+', bb0_663, int_664)
    
    # Processing the call keyword arguments (line 408)
    kwargs_666 = {}
    # Getting the type of 'Basic_block_edge' (line 408)
    Basic_block_edge_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'Basic_block_edge', False)
    # Calling Basic_block_edge(args, kwargs) (line 408)
    Basic_block_edge_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 408, 4), Basic_block_edge_658, *[cfg_659, result_add_662, result_add_665], **kwargs_666)
    
    
    # Call to Basic_block_edge(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'cfg' (line 409)
    cfg_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'cfg', False)
    # Getting the type of 'bb0' (line 409)
    bb0_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 26), 'bb0', False)
    int_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 32), 'int')
    # Applying the binary operator '+' (line 409)
    result_add_672 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 26), '+', bb0_670, int_671)
    
    # Getting the type of 'bb0' (line 409)
    bb0_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 35), 'bb0', False)
    int_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 41), 'int')
    # Applying the binary operator '+' (line 409)
    result_add_675 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 35), '+', bb0_673, int_674)
    
    # Processing the call keyword arguments (line 409)
    kwargs_676 = {}
    # Getting the type of 'Basic_block_edge' (line 409)
    Basic_block_edge_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'Basic_block_edge', False)
    # Calling Basic_block_edge(args, kwargs) (line 409)
    Basic_block_edge_call_result_677 = invoke(stypy.reporting.localization.Localization(__file__, 409, 4), Basic_block_edge_668, *[cfg_669, result_add_672, result_add_675], **kwargs_676)
    
    # Getting the type of 'bb0' (line 410)
    bb0_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 11), 'bb0')
    int_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 17), 'int')
    # Applying the binary operator '+' (line 410)
    result_add_680 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 11), '+', bb0_678, int_679)
    
    # Assigning a type to the variable 'stypy_return_type' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type', result_add_680)
    
    # ################# End of 'build_diamond(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'build_diamond' in the type store
    # Getting the type of 'stypy_return_type' (line 404)
    stypy_return_type_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_681)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'build_diamond'
    return stypy_return_type_681

# Assigning a type to the variable 'build_diamond' (line 404)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'build_diamond', build_diamond)

@norecursion
def build_connect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'build_connect'
    module_type_store = module_type_store.open_function_context('build_connect', 413, 0, False)
    
    # Passed parameters checking function
    build_connect.stypy_localization = localization
    build_connect.stypy_type_of_self = None
    build_connect.stypy_type_store = module_type_store
    build_connect.stypy_function_name = 'build_connect'
    build_connect.stypy_param_names_list = ['cfg', 'start', 'end']
    build_connect.stypy_varargs_param_name = None
    build_connect.stypy_kwargs_param_name = None
    build_connect.stypy_call_defaults = defaults
    build_connect.stypy_call_varargs = varargs
    build_connect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'build_connect', ['cfg', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'build_connect', localization, ['cfg', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'build_connect(...)' code ##################

    
    # Call to Basic_block_edge(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'cfg' (line 414)
    cfg_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 21), 'cfg', False)
    # Getting the type of 'start' (line 414)
    start_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'start', False)
    # Getting the type of 'end' (line 414)
    end_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 33), 'end', False)
    # Processing the call keyword arguments (line 414)
    kwargs_686 = {}
    # Getting the type of 'Basic_block_edge' (line 414)
    Basic_block_edge_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'Basic_block_edge', False)
    # Calling Basic_block_edge(args, kwargs) (line 414)
    Basic_block_edge_call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 414, 4), Basic_block_edge_682, *[cfg_683, start_684, end_685], **kwargs_686)
    
    
    # ################# End of 'build_connect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'build_connect' in the type store
    # Getting the type of 'stypy_return_type' (line 413)
    stypy_return_type_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_688)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'build_connect'
    return stypy_return_type_688

# Assigning a type to the variable 'build_connect' (line 413)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 0), 'build_connect', build_connect)

@norecursion
def build_straight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'build_straight'
    module_type_store = module_type_store.open_function_context('build_straight', 417, 0, False)
    
    # Passed parameters checking function
    build_straight.stypy_localization = localization
    build_straight.stypy_type_of_self = None
    build_straight.stypy_type_store = module_type_store
    build_straight.stypy_function_name = 'build_straight'
    build_straight.stypy_param_names_list = ['cfg', 'start', 'n']
    build_straight.stypy_varargs_param_name = None
    build_straight.stypy_kwargs_param_name = None
    build_straight.stypy_call_defaults = defaults
    build_straight.stypy_call_varargs = varargs
    build_straight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'build_straight', ['cfg', 'start', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'build_straight', localization, ['cfg', 'start', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'build_straight(...)' code ##################

    
    
    # Call to xrange(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'n' (line 418)
    n_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 20), 'n', False)
    # Processing the call keyword arguments (line 418)
    kwargs_691 = {}
    # Getting the type of 'xrange' (line 418)
    xrange_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 418)
    xrange_call_result_692 = invoke(stypy.reporting.localization.Localization(__file__, 418, 13), xrange_689, *[n_690], **kwargs_691)
    
    # Testing if the for loop is going to be iterated (line 418)
    # Testing the type of a for loop iterable (line 418)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 418, 4), xrange_call_result_692)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 418, 4), xrange_call_result_692):
        # Getting the type of the for loop variable (line 418)
        for_loop_var_693 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 418, 4), xrange_call_result_692)
        # Assigning a type to the variable 'i' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'i', for_loop_var_693)
        # SSA begins for a for statement (line 418)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to build_connect(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'cfg' (line 419)
        cfg_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 'cfg', False)
        # Getting the type of 'start' (line 419)
        start_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), 'start', False)
        # Getting the type of 'i' (line 419)
        i_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 35), 'i', False)
        # Applying the binary operator '+' (line 419)
        result_add_698 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 27), '+', start_696, i_697)
        
        # Getting the type of 'start' (line 419)
        start_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 38), 'start', False)
        # Getting the type of 'i' (line 419)
        i_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 46), 'i', False)
        # Applying the binary operator '+' (line 419)
        result_add_701 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 38), '+', start_699, i_700)
        
        int_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 50), 'int')
        # Applying the binary operator '+' (line 419)
        result_add_703 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 48), '+', result_add_701, int_702)
        
        # Processing the call keyword arguments (line 419)
        kwargs_704 = {}
        # Getting the type of 'build_connect' (line 419)
        build_connect_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'build_connect', False)
        # Calling build_connect(args, kwargs) (line 419)
        build_connect_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), build_connect_694, *[cfg_695, result_add_698, result_add_703], **kwargs_704)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'start' (line 420)
    start_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 11), 'start')
    # Getting the type of 'n' (line 420)
    n_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'n')
    # Applying the binary operator '+' (line 420)
    result_add_708 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 11), '+', start_706, n_707)
    
    # Assigning a type to the variable 'stypy_return_type' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type', result_add_708)
    
    # ################# End of 'build_straight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'build_straight' in the type store
    # Getting the type of 'stypy_return_type' (line 417)
    stypy_return_type_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_709)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'build_straight'
    return stypy_return_type_709

# Assigning a type to the variable 'build_straight' (line 417)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'build_straight', build_straight)

@norecursion
def build_base_loop(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'build_base_loop'
    module_type_store = module_type_store.open_function_context('build_base_loop', 423, 0, False)
    
    # Passed parameters checking function
    build_base_loop.stypy_localization = localization
    build_base_loop.stypy_type_of_self = None
    build_base_loop.stypy_type_store = module_type_store
    build_base_loop.stypy_function_name = 'build_base_loop'
    build_base_loop.stypy_param_names_list = ['cfg', 'from_n']
    build_base_loop.stypy_varargs_param_name = None
    build_base_loop.stypy_kwargs_param_name = None
    build_base_loop.stypy_call_defaults = defaults
    build_base_loop.stypy_call_varargs = varargs
    build_base_loop.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'build_base_loop', ['cfg', 'from_n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'build_base_loop', localization, ['cfg', 'from_n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'build_base_loop(...)' code ##################

    
    # Assigning a Call to a Name (line 424):
    
    # Call to build_straight(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'cfg' (line 424)
    cfg_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 28), 'cfg', False)
    # Getting the type of 'from_n' (line 424)
    from_n_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 33), 'from_n', False)
    int_713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 41), 'int')
    # Processing the call keyword arguments (line 424)
    kwargs_714 = {}
    # Getting the type of 'build_straight' (line 424)
    build_straight_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 13), 'build_straight', False)
    # Calling build_straight(args, kwargs) (line 424)
    build_straight_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 424, 13), build_straight_710, *[cfg_711, from_n_712, int_713], **kwargs_714)
    
    # Assigning a type to the variable 'header' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'header', build_straight_call_result_715)
    
    # Assigning a Call to a Name (line 425):
    
    # Call to build_diamond(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'cfg' (line 425)
    cfg_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 29), 'cfg', False)
    # Getting the type of 'header' (line 425)
    header_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 34), 'header', False)
    # Processing the call keyword arguments (line 425)
    kwargs_719 = {}
    # Getting the type of 'build_diamond' (line 425)
    build_diamond_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'build_diamond', False)
    # Calling build_diamond(args, kwargs) (line 425)
    build_diamond_call_result_720 = invoke(stypy.reporting.localization.Localization(__file__, 425, 15), build_diamond_716, *[cfg_717, header_718], **kwargs_719)
    
    # Assigning a type to the variable 'diamond1' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'diamond1', build_diamond_call_result_720)
    
    # Assigning a Call to a Name (line 426):
    
    # Call to build_straight(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'cfg' (line 426)
    cfg_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 25), 'cfg', False)
    # Getting the type of 'diamond1' (line 426)
    diamond1_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 30), 'diamond1', False)
    int_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 40), 'int')
    # Processing the call keyword arguments (line 426)
    kwargs_725 = {}
    # Getting the type of 'build_straight' (line 426)
    build_straight_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 10), 'build_straight', False)
    # Calling build_straight(args, kwargs) (line 426)
    build_straight_call_result_726 = invoke(stypy.reporting.localization.Localization(__file__, 426, 10), build_straight_721, *[cfg_722, diamond1_723, int_724], **kwargs_725)
    
    # Assigning a type to the variable 'd11' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'd11', build_straight_call_result_726)
    
    # Assigning a Call to a Name (line 427):
    
    # Call to build_diamond(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'cfg' (line 427)
    cfg_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 29), 'cfg', False)
    # Getting the type of 'd11' (line 427)
    d11_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 34), 'd11', False)
    # Processing the call keyword arguments (line 427)
    kwargs_730 = {}
    # Getting the type of 'build_diamond' (line 427)
    build_diamond_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'build_diamond', False)
    # Calling build_diamond(args, kwargs) (line 427)
    build_diamond_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), build_diamond_727, *[cfg_728, d11_729], **kwargs_730)
    
    # Assigning a type to the variable 'diamond2' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'diamond2', build_diamond_call_result_731)
    
    # Assigning a Call to a Name (line 428):
    
    # Call to build_straight(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'cfg' (line 428)
    cfg_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 28), 'cfg', False)
    # Getting the type of 'diamond2' (line 428)
    diamond2_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 33), 'diamond2', False)
    int_735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 43), 'int')
    # Processing the call keyword arguments (line 428)
    kwargs_736 = {}
    # Getting the type of 'build_straight' (line 428)
    build_straight_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'build_straight', False)
    # Calling build_straight(args, kwargs) (line 428)
    build_straight_call_result_737 = invoke(stypy.reporting.localization.Localization(__file__, 428, 13), build_straight_732, *[cfg_733, diamond2_734, int_735], **kwargs_736)
    
    # Assigning a type to the variable 'footer' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'footer', build_straight_call_result_737)
    
    # Call to build_connect(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'cfg' (line 429)
    cfg_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 18), 'cfg', False)
    # Getting the type of 'diamond2' (line 429)
    diamond2_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'diamond2', False)
    # Getting the type of 'd11' (line 429)
    d11_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 33), 'd11', False)
    # Processing the call keyword arguments (line 429)
    kwargs_742 = {}
    # Getting the type of 'build_connect' (line 429)
    build_connect_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'build_connect', False)
    # Calling build_connect(args, kwargs) (line 429)
    build_connect_call_result_743 = invoke(stypy.reporting.localization.Localization(__file__, 429, 4), build_connect_738, *[cfg_739, diamond2_740, d11_741], **kwargs_742)
    
    
    # Call to build_connect(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'cfg' (line 430)
    cfg_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 18), 'cfg', False)
    # Getting the type of 'diamond1' (line 430)
    diamond1_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 23), 'diamond1', False)
    # Getting the type of 'header' (line 430)
    header_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 33), 'header', False)
    # Processing the call keyword arguments (line 430)
    kwargs_748 = {}
    # Getting the type of 'build_connect' (line 430)
    build_connect_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'build_connect', False)
    # Calling build_connect(args, kwargs) (line 430)
    build_connect_call_result_749 = invoke(stypy.reporting.localization.Localization(__file__, 430, 4), build_connect_744, *[cfg_745, diamond1_746, header_747], **kwargs_748)
    
    
    # Call to build_connect(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'cfg' (line 431)
    cfg_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'cfg', False)
    # Getting the type of 'footer' (line 431)
    footer_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'footer', False)
    # Getting the type of 'from_n' (line 431)
    from_n_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'from_n', False)
    # Processing the call keyword arguments (line 431)
    kwargs_754 = {}
    # Getting the type of 'build_connect' (line 431)
    build_connect_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'build_connect', False)
    # Calling build_connect(args, kwargs) (line 431)
    build_connect_call_result_755 = invoke(stypy.reporting.localization.Localization(__file__, 431, 4), build_connect_750, *[cfg_751, footer_752, from_n_753], **kwargs_754)
    
    
    # Assigning a Call to a Name (line 432):
    
    # Call to build_straight(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'cfg' (line 432)
    cfg_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 28), 'cfg', False)
    # Getting the type of 'footer' (line 432)
    footer_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 33), 'footer', False)
    int_759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 41), 'int')
    # Processing the call keyword arguments (line 432)
    kwargs_760 = {}
    # Getting the type of 'build_straight' (line 432)
    build_straight_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 13), 'build_straight', False)
    # Calling build_straight(args, kwargs) (line 432)
    build_straight_call_result_761 = invoke(stypy.reporting.localization.Localization(__file__, 432, 13), build_straight_756, *[cfg_757, footer_758, int_759], **kwargs_760)
    
    # Assigning a type to the variable 'footer' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'footer', build_straight_call_result_761)
    # Getting the type of 'footer' (line 433)
    footer_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'footer')
    # Assigning a type to the variable 'stypy_return_type' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type', footer_762)
    
    # ################# End of 'build_base_loop(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'build_base_loop' in the type store
    # Getting the type of 'stypy_return_type' (line 423)
    stypy_return_type_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_763)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'build_base_loop'
    return stypy_return_type_763

# Assigning a type to the variable 'build_base_loop' (line 423)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'build_base_loop', build_base_loop)
# Declaration of the 'Basic_block_edge' class

class Basic_block_edge(object, ):
    str_764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 4), 'str', 'Basic_block_edge only maintains two pointers to BasicBlocks.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Basic_block_edge.__init__', ['cfg', 'from_name', 'to_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['cfg', 'from_name', 'to_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 443):
        
        # Call to create_node(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'from_name' (line 443)
        from_name_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 37), 'from_name', False)
        # Processing the call keyword arguments (line 443)
        kwargs_768 = {}
        # Getting the type of 'cfg' (line 443)
        cfg_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 21), 'cfg', False)
        # Obtaining the member 'create_node' of a type (line 443)
        create_node_766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 21), cfg_765, 'create_node')
        # Calling create_node(args, kwargs) (line 443)
        create_node_call_result_769 = invoke(stypy.reporting.localization.Localization(__file__, 443, 21), create_node_766, *[from_name_767], **kwargs_768)
        
        # Getting the type of 'self' (line 443)
        self_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self')
        # Setting the type of the member 'from_' of a type (line 443)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_770, 'from_', create_node_call_result_769)
        
        # Assigning a Call to a Attribute (line 444):
        
        # Call to create_node(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'to_name' (line 444)
        to_name_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 35), 'to_name', False)
        # Processing the call keyword arguments (line 444)
        kwargs_774 = {}
        # Getting the type of 'cfg' (line 444)
        cfg_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'cfg', False)
        # Obtaining the member 'create_node' of a type (line 444)
        create_node_772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 19), cfg_771, 'create_node')
        # Calling create_node(args, kwargs) (line 444)
        create_node_call_result_775 = invoke(stypy.reporting.localization.Localization(__file__, 444, 19), create_node_772, *[to_name_773], **kwargs_774)
        
        # Getting the type of 'self' (line 444)
        self_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'self')
        # Setting the type of the member 'to_' of a type (line 444)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), self_776, 'to_', create_node_call_result_775)
        
        # Call to append(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'self' (line 445)
        self_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 37), 'self', False)
        # Obtaining the member 'to_' of a type (line 445)
        to__782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 37), self_781, 'to_')
        # Processing the call keyword arguments (line 445)
        kwargs_783 = {}
        # Getting the type of 'self' (line 445)
        self_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'self', False)
        # Obtaining the member 'from_' of a type (line 445)
        from__778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), self_777, 'from_')
        # Obtaining the member 'out_edges_' of a type (line 445)
        out_edges__779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), from__778, 'out_edges_')
        # Obtaining the member 'append' of a type (line 445)
        append_780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), out_edges__779, 'append')
        # Calling append(args, kwargs) (line 445)
        append_call_result_784 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), append_780, *[to__782], **kwargs_783)
        
        
        # Call to append(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'self' (line 446)
        self_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 34), 'self', False)
        # Obtaining the member 'from_' of a type (line 446)
        from__790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 34), self_789, 'from_')
        # Processing the call keyword arguments (line 446)
        kwargs_791 = {}
        # Getting the type of 'self' (line 446)
        self_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'self', False)
        # Obtaining the member 'to_' of a type (line 446)
        to__786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), self_785, 'to_')
        # Obtaining the member 'in_edges_' of a type (line 446)
        in_edges__787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), to__786, 'in_edges_')
        # Obtaining the member 'append' of a type (line 446)
        append_788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), in_edges__787, 'append')
        # Calling append(args, kwargs) (line 446)
        append_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), append_788, *[from__790], **kwargs_791)
        
        
        # Call to append(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'self' (line 447)
        self_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 30), 'self', False)
        # Processing the call keyword arguments (line 447)
        kwargs_797 = {}
        # Getting the type of 'cfg' (line 447)
        cfg_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'cfg', False)
        # Obtaining the member 'edge_list_' of a type (line 447)
        edge_list__794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), cfg_793, 'edge_list_')
        # Obtaining the member 'append' of a type (line 447)
        append_795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), edge_list__794, 'append')
        # Calling append(args, kwargs) (line 447)
        append_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), append_795, *[self_796], **kwargs_797)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Basic_block_edge' (line 439)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'Basic_block_edge', Basic_block_edge)
# Declaration of the 'Basic_block' class

class Basic_block(object, ):
    str_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 4), 'str', 'Basic_block only maintains a vector of in-edges and a vector of out-edges.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 453, 4, False)
        # Assigning a type to the variable 'self' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Basic_block.__init__', ['name'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 454):
        
        # Obtaining an instance of the builtin type 'list' (line 454)
        list_800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 454)
        
        # Getting the type of 'self' (line 454)
        self_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self')
        # Setting the type of the member 'in_edges_' of a type (line 454)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_801, 'in_edges_', list_800)
        
        # Assigning a List to a Attribute (line 455):
        
        # Obtaining an instance of the builtin type 'list' (line 455)
        list_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 455)
        
        # Getting the type of 'self' (line 455)
        self_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self')
        # Setting the type of the member 'out_edges_' of a type (line 455)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_803, 'out_edges_', list_802)
        
        # Assigning a Name to a Attribute (line 456):
        # Getting the type of 'name' (line 456)
        name_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 21), 'name')
        # Getting the type of 'self' (line 456)
        self_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'self')
        # Setting the type of the member 'name_' of a type (line 456)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), self_805, 'name_', name_804)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Basic_block' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'Basic_block', Basic_block)
# Declaration of the 'MaoCFG' class

class MaoCFG(object, ):
    str_806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 4), 'str', 'MaoCFG maintains a list of nodes.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 462, 4, False)
        # Assigning a type to the variable 'self' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaoCFG.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Dict to a Attribute (line 463):
        
        # Obtaining an instance of the builtin type 'dict' (line 463)
        dict_807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 32), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 463)
        
        # Getting the type of 'self' (line 463)
        self_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'self')
        # Setting the type of the member 'basic_block_map_' of a type (line 463)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), self_808, 'basic_block_map_', dict_807)
        
        # Assigning a Name to a Attribute (line 464):
        # Getting the type of 'None' (line 464)
        None_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 27), 'None')
        # Getting the type of 'self' (line 464)
        self_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'self')
        # Setting the type of the member 'start_node_' of a type (line 464)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), self_810, 'start_node_', None_809)
        
        # Assigning a List to a Attribute (line 465):
        
        # Obtaining an instance of the builtin type 'list' (line 465)
        list_811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 465)
        
        # Getting the type of 'self' (line 465)
        self_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'self')
        # Setting the type of the member 'edge_list_' of a type (line 465)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 8), self_812, 'edge_list_', list_811)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def create_node(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_node'
        module_type_store = module_type_store.open_function_context('create_node', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaoCFG.create_node.__dict__.__setitem__('stypy_localization', localization)
        MaoCFG.create_node.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaoCFG.create_node.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaoCFG.create_node.__dict__.__setitem__('stypy_function_name', 'MaoCFG.create_node')
        MaoCFG.create_node.__dict__.__setitem__('stypy_param_names_list', ['name'])
        MaoCFG.create_node.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaoCFG.create_node.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaoCFG.create_node.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaoCFG.create_node.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaoCFG.create_node.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaoCFG.create_node.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaoCFG.create_node', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_node', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_node(...)' code ##################

        
        # Getting the type of 'name' (line 468)
        name_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'name')
        # Getting the type of 'self' (line 468)
        self_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 19), 'self')
        # Obtaining the member 'basic_block_map_' of a type (line 468)
        basic_block_map__815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 19), self_814, 'basic_block_map_')
        # Applying the binary operator 'in' (line 468)
        result_contains_816 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 11), 'in', name_813, basic_block_map__815)
        
        # Testing if the type of an if condition is none (line 468)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 468, 8), result_contains_816):
            
            # Assigning a Call to a Name (line 471):
            
            # Call to Basic_block(...): (line 471)
            # Processing the call arguments (line 471)
            # Getting the type of 'name' (line 471)
            name_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 31), 'name', False)
            # Processing the call keyword arguments (line 471)
            kwargs_825 = {}
            # Getting the type of 'Basic_block' (line 471)
            Basic_block_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'Basic_block', False)
            # Calling Basic_block(args, kwargs) (line 471)
            Basic_block_call_result_826 = invoke(stypy.reporting.localization.Localization(__file__, 471, 19), Basic_block_823, *[name_824], **kwargs_825)
            
            # Assigning a type to the variable 'node' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'node', Basic_block_call_result_826)
            
            # Assigning a Name to a Subscript (line 472):
            # Getting the type of 'node' (line 472)
            node_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 42), 'node')
            # Getting the type of 'self' (line 472)
            self_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'self')
            # Obtaining the member 'basic_block_map_' of a type (line 472)
            basic_block_map__829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), self_828, 'basic_block_map_')
            # Getting the type of 'name' (line 472)
            name_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 34), 'name')
            # Storing an element on a container (line 472)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 12), basic_block_map__829, (name_830, node_827))
        else:
            
            # Testing the type of an if condition (line 468)
            if_condition_817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 8), result_contains_816)
            # Assigning a type to the variable 'if_condition_817' (line 468)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'if_condition_817', if_condition_817)
            # SSA begins for if statement (line 468)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 469):
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 469)
            name_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 41), 'name')
            # Getting the type of 'self' (line 469)
            self_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'self')
            # Obtaining the member 'basic_block_map_' of a type (line 469)
            basic_block_map__820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 19), self_819, 'basic_block_map_')
            # Obtaining the member '__getitem__' of a type (line 469)
            getitem___821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 19), basic_block_map__820, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 469)
            subscript_call_result_822 = invoke(stypy.reporting.localization.Localization(__file__, 469, 19), getitem___821, name_818)
            
            # Assigning a type to the variable 'node' (line 469)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'node', subscript_call_result_822)
            # SSA branch for the else part of an if statement (line 468)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 471):
            
            # Call to Basic_block(...): (line 471)
            # Processing the call arguments (line 471)
            # Getting the type of 'name' (line 471)
            name_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 31), 'name', False)
            # Processing the call keyword arguments (line 471)
            kwargs_825 = {}
            # Getting the type of 'Basic_block' (line 471)
            Basic_block_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'Basic_block', False)
            # Calling Basic_block(args, kwargs) (line 471)
            Basic_block_call_result_826 = invoke(stypy.reporting.localization.Localization(__file__, 471, 19), Basic_block_823, *[name_824], **kwargs_825)
            
            # Assigning a type to the variable 'node' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'node', Basic_block_call_result_826)
            
            # Assigning a Name to a Subscript (line 472):
            # Getting the type of 'node' (line 472)
            node_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 42), 'node')
            # Getting the type of 'self' (line 472)
            self_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'self')
            # Obtaining the member 'basic_block_map_' of a type (line 472)
            basic_block_map__829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), self_828, 'basic_block_map_')
            # Getting the type of 'name' (line 472)
            name_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 34), 'name')
            # Storing an element on a container (line 472)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 12), basic_block_map__829, (name_830, node_827))
            # SSA join for if statement (line 468)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'self' (line 474)
        self_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'self', False)
        # Obtaining the member 'basic_block_map_' of a type (line 474)
        basic_block_map__833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 15), self_832, 'basic_block_map_')
        # Processing the call keyword arguments (line 474)
        kwargs_834 = {}
        # Getting the type of 'len' (line 474)
        len_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'len', False)
        # Calling len(args, kwargs) (line 474)
        len_call_result_835 = invoke(stypy.reporting.localization.Localization(__file__, 474, 11), len_831, *[basic_block_map__833], **kwargs_834)
        
        int_836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 41), 'int')
        # Applying the binary operator '==' (line 474)
        result_eq_837 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 11), '==', len_call_result_835, int_836)
        
        # Testing if the type of an if condition is none (line 474)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 474, 8), result_eq_837):
            pass
        else:
            
            # Testing the type of an if condition (line 474)
            if_condition_838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 8), result_eq_837)
            # Assigning a type to the variable 'if_condition_838' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'if_condition_838', if_condition_838)
            # SSA begins for if statement (line 474)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 475):
            # Getting the type of 'node' (line 475)
            node_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'node')
            # Getting the type of 'self' (line 475)
            self_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'self')
            # Setting the type of the member 'start_node_' of a type (line 475)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), self_840, 'start_node_', node_839)
            # SSA join for if statement (line 474)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'node' (line 477)
        node_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'stypy_return_type', node_841)
        
        # ################# End of 'create_node(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_node' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_node'
        return stypy_return_type_842


# Assigning a type to the variable 'MaoCFG' (line 459)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 0), 'MaoCFG', MaoCFG)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 483, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a Call to a Name (line 486):
    
    # Call to MaoCFG(...): (line 486)
    # Processing the call keyword arguments (line 486)
    kwargs_844 = {}
    # Getting the type of 'MaoCFG' (line 486)
    MaoCFG_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 10), 'MaoCFG', False)
    # Calling MaoCFG(args, kwargs) (line 486)
    MaoCFG_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 486, 10), MaoCFG_843, *[], **kwargs_844)
    
    # Assigning a type to the variable 'cfg' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'cfg', MaoCFG_call_result_845)
    
    # Assigning a Call to a Name (line 487):
    
    # Call to Loop_structure_graph(...): (line 487)
    # Processing the call keyword arguments (line 487)
    kwargs_847 = {}
    # Getting the type of 'Loop_structure_graph' (line 487)
    Loop_structure_graph_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 10), 'Loop_structure_graph', False)
    # Calling Loop_structure_graph(args, kwargs) (line 487)
    Loop_structure_graph_call_result_848 = invoke(stypy.reporting.localization.Localization(__file__, 487, 10), Loop_structure_graph_846, *[], **kwargs_847)
    
    # Assigning a type to the variable 'lsg' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'lsg', Loop_structure_graph_call_result_848)
    
    # Call to create_node(...): (line 490)
    # Processing the call arguments (line 490)
    int_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 20), 'int')
    # Processing the call keyword arguments (line 490)
    kwargs_852 = {}
    # Getting the type of 'cfg' (line 490)
    cfg_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'cfg', False)
    # Obtaining the member 'create_node' of a type (line 490)
    create_node_850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), cfg_849, 'create_node')
    # Calling create_node(args, kwargs) (line 490)
    create_node_call_result_853 = invoke(stypy.reporting.localization.Localization(__file__, 490, 4), create_node_850, *[int_851], **kwargs_852)
    
    
    # Call to build_base_loop(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'cfg' (line 491)
    cfg_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'cfg', False)
    int_856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 25), 'int')
    # Processing the call keyword arguments (line 491)
    kwargs_857 = {}
    # Getting the type of 'build_base_loop' (line 491)
    build_base_loop_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'build_base_loop', False)
    # Calling build_base_loop(args, kwargs) (line 491)
    build_base_loop_call_result_858 = invoke(stypy.reporting.localization.Localization(__file__, 491, 4), build_base_loop_854, *[cfg_855, int_856], **kwargs_857)
    
    
    # Call to create_node(...): (line 492)
    # Processing the call arguments (line 492)
    int_861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 20), 'int')
    # Processing the call keyword arguments (line 492)
    kwargs_862 = {}
    # Getting the type of 'cfg' (line 492)
    cfg_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'cfg', False)
    # Obtaining the member 'create_node' of a type (line 492)
    create_node_860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 4), cfg_859, 'create_node')
    # Calling create_node(args, kwargs) (line 492)
    create_node_call_result_863 = invoke(stypy.reporting.localization.Localization(__file__, 492, 4), create_node_860, *[int_861], **kwargs_862)
    
    
    # Call to Basic_block_edge(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'cfg' (line 493)
    cfg_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 21), 'cfg', False)
    int_866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 26), 'int')
    int_867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 29), 'int')
    # Processing the call keyword arguments (line 493)
    kwargs_868 = {}
    # Getting the type of 'Basic_block_edge' (line 493)
    Basic_block_edge_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'Basic_block_edge', False)
    # Calling Basic_block_edge(args, kwargs) (line 493)
    Basic_block_edge_call_result_869 = invoke(stypy.reporting.localization.Localization(__file__, 493, 4), Basic_block_edge_864, *[cfg_865, int_866, int_867], **kwargs_868)
    
    
    
    # Call to xrange(...): (line 496)
    # Processing the call arguments (line 496)
    int_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 29), 'int')
    # Processing the call keyword arguments (line 496)
    kwargs_872 = {}
    # Getting the type of 'xrange' (line 496)
    xrange_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 22), 'xrange', False)
    # Calling xrange(args, kwargs) (line 496)
    xrange_call_result_873 = invoke(stypy.reporting.localization.Localization(__file__, 496, 22), xrange_870, *[int_871], **kwargs_872)
    
    # Testing if the for loop is going to be iterated (line 496)
    # Testing the type of a for loop iterable (line 496)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 496, 4), xrange_call_result_873)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 496, 4), xrange_call_result_873):
        # Getting the type of the for loop variable (line 496)
        for_loop_var_874 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 496, 4), xrange_call_result_873)
        # Assigning a type to the variable 'dummyLoops' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'dummyLoops', for_loop_var_874)
        # SSA begins for a for statement (line 496)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 497):
        
        # Call to Loop_structure_graph(...): (line 497)
        # Processing the call keyword arguments (line 497)
        kwargs_876 = {}
        # Getting the type of 'Loop_structure_graph' (line 497)
        Loop_structure_graph_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'Loop_structure_graph', False)
        # Calling Loop_structure_graph(args, kwargs) (line 497)
        Loop_structure_graph_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 497, 19), Loop_structure_graph_875, *[], **kwargs_876)
        
        # Assigning a type to the variable 'lsglocal' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'lsglocal', Loop_structure_graph_call_result_877)
        
        # Call to find_havlak_loops(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'cfg' (line 498)
        cfg_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 26), 'cfg', False)
        # Getting the type of 'lsglocal' (line 498)
        lsglocal_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'lsglocal', False)
        # Processing the call keyword arguments (line 498)
        kwargs_881 = {}
        # Getting the type of 'find_havlak_loops' (line 498)
        find_havlak_loops_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'find_havlak_loops', False)
        # Calling find_havlak_loops(args, kwargs) (line 498)
        find_havlak_loops_call_result_882 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), find_havlak_loops_878, *[cfg_879, lsglocal_880], **kwargs_881)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Num to a Name (line 501):
    int_883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 8), 'int')
    # Assigning a type to the variable 'n' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'n', int_883)
    
    
    # Call to xrange(...): (line 503)
    # Processing the call arguments (line 503)
    int_885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 31), 'int')
    # Processing the call keyword arguments (line 503)
    kwargs_886 = {}
    # Getting the type of 'xrange' (line 503)
    xrange_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'xrange', False)
    # Calling xrange(args, kwargs) (line 503)
    xrange_call_result_887 = invoke(stypy.reporting.localization.Localization(__file__, 503, 24), xrange_884, *[int_885], **kwargs_886)
    
    # Testing if the for loop is going to be iterated (line 503)
    # Testing the type of a for loop iterable (line 503)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 503, 4), xrange_call_result_887)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 503, 4), xrange_call_result_887):
        # Getting the type of the for loop variable (line 503)
        for_loop_var_888 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 503, 4), xrange_call_result_887)
        # Assigning a type to the variable 'parlooptrees' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'parlooptrees', for_loop_var_888)
        # SSA begins for a for statement (line 503)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to create_node(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'n' (line 504)
        n_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 24), 'n', False)
        int_892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 28), 'int')
        # Applying the binary operator '+' (line 504)
        result_add_893 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 24), '+', n_891, int_892)
        
        # Processing the call keyword arguments (line 504)
        kwargs_894 = {}
        # Getting the type of 'cfg' (line 504)
        cfg_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'cfg', False)
        # Obtaining the member 'create_node' of a type (line 504)
        create_node_890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 8), cfg_889, 'create_node')
        # Calling create_node(args, kwargs) (line 504)
        create_node_call_result_895 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), create_node_890, *[result_add_893], **kwargs_894)
        
        
        # Call to build_connect(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'cfg' (line 505)
        cfg_897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 22), 'cfg', False)
        int_898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 27), 'int')
        # Getting the type of 'n' (line 505)
        n_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 30), 'n', False)
        int_900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 34), 'int')
        # Applying the binary operator '+' (line 505)
        result_add_901 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 30), '+', n_899, int_900)
        
        # Processing the call keyword arguments (line 505)
        kwargs_902 = {}
        # Getting the type of 'build_connect' (line 505)
        build_connect_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'build_connect', False)
        # Calling build_connect(args, kwargs) (line 505)
        build_connect_call_result_903 = invoke(stypy.reporting.localization.Localization(__file__, 505, 8), build_connect_896, *[cfg_897, int_898, result_add_901], **kwargs_902)
        
        
        # Getting the type of 'n' (line 506)
        n_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'n')
        int_905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 13), 'int')
        # Applying the binary operator '+=' (line 506)
        result_iadd_906 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 8), '+=', n_904, int_905)
        # Assigning a type to the variable 'n' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'n', result_iadd_906)
        
        
        
        # Call to xrange(...): (line 508)
        # Processing the call arguments (line 508)
        int_908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 24), 'int')
        # Processing the call keyword arguments (line 508)
        kwargs_909 = {}
        # Getting the type of 'xrange' (line 508)
        xrange_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 508)
        xrange_call_result_910 = invoke(stypy.reporting.localization.Localization(__file__, 508, 17), xrange_907, *[int_908], **kwargs_909)
        
        # Testing if the for loop is going to be iterated (line 508)
        # Testing the type of a for loop iterable (line 508)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 508, 8), xrange_call_result_910)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 508, 8), xrange_call_result_910):
            # Getting the type of the for loop variable (line 508)
            for_loop_var_911 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 508, 8), xrange_call_result_910)
            # Assigning a type to the variable 'i' (line 508)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'i', for_loop_var_911)
            # SSA begins for a for statement (line 508)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Name (line 509):
            # Getting the type of 'n' (line 509)
            n_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 18), 'n')
            # Assigning a type to the variable 'top' (line 509)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'top', n_912)
            
            # Assigning a Call to a Name (line 510):
            
            # Call to build_straight(...): (line 510)
            # Processing the call arguments (line 510)
            # Getting the type of 'cfg' (line 510)
            cfg_914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 31), 'cfg', False)
            # Getting the type of 'n' (line 510)
            n_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 36), 'n', False)
            int_916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 39), 'int')
            # Processing the call keyword arguments (line 510)
            kwargs_917 = {}
            # Getting the type of 'build_straight' (line 510)
            build_straight_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'build_straight', False)
            # Calling build_straight(args, kwargs) (line 510)
            build_straight_call_result_918 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), build_straight_913, *[cfg_914, n_915, int_916], **kwargs_917)
            
            # Assigning a type to the variable 'n' (line 510)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'n', build_straight_call_result_918)
            
            
            # Call to xrange(...): (line 511)
            # Processing the call arguments (line 511)
            int_920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 28), 'int')
            # Processing the call keyword arguments (line 511)
            kwargs_921 = {}
            # Getting the type of 'xrange' (line 511)
            xrange_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 511)
            xrange_call_result_922 = invoke(stypy.reporting.localization.Localization(__file__, 511, 21), xrange_919, *[int_920], **kwargs_921)
            
            # Testing if the for loop is going to be iterated (line 511)
            # Testing the type of a for loop iterable (line 511)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 511, 12), xrange_call_result_922)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 511, 12), xrange_call_result_922):
                # Getting the type of the for loop variable (line 511)
                for_loop_var_923 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 511, 12), xrange_call_result_922)
                # Assigning a type to the variable 'j' (line 511)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'j', for_loop_var_923)
                # SSA begins for a for statement (line 511)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 512):
                
                # Call to build_base_loop(...): (line 512)
                # Processing the call arguments (line 512)
                # Getting the type of 'cfg' (line 512)
                cfg_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 36), 'cfg', False)
                # Getting the type of 'n' (line 512)
                n_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 41), 'n', False)
                # Processing the call keyword arguments (line 512)
                kwargs_927 = {}
                # Getting the type of 'build_base_loop' (line 512)
                build_base_loop_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 20), 'build_base_loop', False)
                # Calling build_base_loop(args, kwargs) (line 512)
                build_base_loop_call_result_928 = invoke(stypy.reporting.localization.Localization(__file__, 512, 20), build_base_loop_924, *[cfg_925, n_926], **kwargs_927)
                
                # Assigning a type to the variable 'n' (line 512)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'n', build_base_loop_call_result_928)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 513):
            
            # Call to build_straight(...): (line 513)
            # Processing the call arguments (line 513)
            # Getting the type of 'cfg' (line 513)
            cfg_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 36), 'cfg', False)
            # Getting the type of 'n' (line 513)
            n_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 41), 'n', False)
            int_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 44), 'int')
            # Processing the call keyword arguments (line 513)
            kwargs_933 = {}
            # Getting the type of 'build_straight' (line 513)
            build_straight_929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'build_straight', False)
            # Calling build_straight(args, kwargs) (line 513)
            build_straight_call_result_934 = invoke(stypy.reporting.localization.Localization(__file__, 513, 21), build_straight_929, *[cfg_930, n_931, int_932], **kwargs_933)
            
            # Assigning a type to the variable 'bottom' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'bottom', build_straight_call_result_934)
            
            # Call to build_connect(...): (line 514)
            # Processing the call arguments (line 514)
            # Getting the type of 'cfg' (line 514)
            cfg_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 26), 'cfg', False)
            # Getting the type of 'n' (line 514)
            n_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 31), 'n', False)
            # Getting the type of 'top' (line 514)
            top_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 34), 'top', False)
            # Processing the call keyword arguments (line 514)
            kwargs_939 = {}
            # Getting the type of 'build_connect' (line 514)
            build_connect_935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'build_connect', False)
            # Calling build_connect(args, kwargs) (line 514)
            build_connect_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 514, 12), build_connect_935, *[cfg_936, n_937, top_938], **kwargs_939)
            
            
            # Assigning a Name to a Name (line 515):
            # Getting the type of 'bottom' (line 515)
            bottom_941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 16), 'bottom')
            # Assigning a type to the variable 'n' (line 515)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'n', bottom_941)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to build_connect(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'cfg' (line 516)
        cfg_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 22), 'cfg', False)
        # Getting the type of 'n' (line 516)
        n_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 27), 'n', False)
        int_945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 30), 'int')
        # Processing the call keyword arguments (line 516)
        kwargs_946 = {}
        # Getting the type of 'build_connect' (line 516)
        build_connect_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'build_connect', False)
        # Calling build_connect(args, kwargs) (line 516)
        build_connect_call_result_947 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), build_connect_942, *[cfg_943, n_944, int_945], **kwargs_946)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 483)
    stypy_return_type_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_948)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_948

# Assigning a type to the variable 'main' (line 483)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 536, 0, False)
    
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

    
    # Call to main(...): (line 537)
    # Processing the call keyword arguments (line 537)
    kwargs_950 = {}
    # Getting the type of 'main' (line 537)
    main_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'main', False)
    # Calling main(args, kwargs) (line 537)
    main_call_result_951 = invoke(stypy.reporting.localization.Localization(__file__, 537, 4), main_949, *[], **kwargs_950)
    
    # Getting the type of 'True' (line 538)
    True_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'stypy_return_type', True_952)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 536)
    stypy_return_type_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_953)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_953

# Assigning a type to the variable 'run' (line 536)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 0), 'run', run)

# Call to run(...): (line 541)
# Processing the call keyword arguments (line 541)
kwargs_955 = {}
# Getting the type of 'run' (line 541)
run_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 0), 'run', False)
# Calling run(args, kwargs) (line 541)
run_call_result_956 = invoke(stypy.reporting.localization.Localization(__file__, 541, 0), run_954, *[], **kwargs_955)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

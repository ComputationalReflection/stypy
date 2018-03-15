
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: ##                   (c) David MacKay - Free software. License: GPL
3: ## For license statement see  http://www.gnu.org/copyleft/gpl.html
4: '''
5: This is a BLOCK compression algorithm
6: that uses the Huffman algorithm.
7: 
8: This simple block compressor assumes that the source file
9: is an exact multiple of the block length.
10: The encoding does not itself delimit the size of the file, so
11: the decoder needs to knows where the end of the compressed
12: file is. Thus we must either ensure the decoder knows
13: the original file's length, or we have to use a special
14: end-of-file character. A superior compressor would first
15: encode the source file's length at the head of the compressed
16: file; then the decoder would be able to stop at the right
17: point and correct any truncation errors. I'll fix this
18: in block2.py.
19: 
20: The package contains the following functions:
21: 
22:  findprobs(f=0.01,N=6):
23:     Find probabilities of all the events
24:     000000
25:     000001
26:      ...
27:     111111
28:     <-N ->
29: 
30:  Bencode(string,symbols,N):
31:     Reads in a string of 0s and 1s, forms blocks, and encodes with Huffman.
32: 
33:  Bdecode(string,root,N):
34:     Decode a binary string into blocks, then return appropriate 0s and 1s
35: 
36:  compress_it( inputfile, outputfile ):
37:     Make Huffman code, and compress
38: 
39:  uncompress_it( inputfile, outputfile ):
40:     Make Huffman code, and uncompress
41: 
42:  There are also three test functions.
43:  If block.py is run from a terminal, it invokes compress_it (using stdin)
44:  or uncompress_it (using stdin), respectively if there are zero arguments
45:  or one argument.
46: 
47: '''
48: ## /home/mackay/python/compression/huffman/Huffman3.py
49: ## This supplies the huffman algorithm, complete with encoders and decoders:
50: 
51: import sys, os
52: 
53: 
54: class node:
55:     def __init__(self, count, index, name=""):
56:         self.count = float(count)
57:         self.index = index
58:         self.name = name  ## optional argument
59:         if self.name == "": self.name = '_' + str(index)
60:         self.word = ""  ## codeword will go here
61:         self.isinternal = 0
62: 
63:     def __cmp__(self, other):
64:         return cmp(self.count, other.count)
65: 
66:     def report(self):
67:         if (self.index == 1):
68:             pass  # print '#Symbol\tCount\tCodeword'
69:         # print '%s\t(%2.2g)\t%s' % (self.name,self.count,self.word)
70:         pass
71: 
72:     def associate(self, internalnode):
73:         self.internalnode = internalnode
74:         internalnode.leaf = 1
75:         internalnode.name = self.name
76:         pass
77: 
78: 
79: class internalnode:
80:     def __init__(self):
81:         self.leaf = 0
82:         self.child = []
83:         pass
84: 
85:     def children(self, child0, child1):
86:         self.leaf = 0
87:         self.child.append(child0)
88:         self.child.append(child1)
89:         pass
90: 
91: 
92: def find_idx(seq, index):
93:     for item in seq:
94:         if item.index == index:
95:             return item
96: 
97: 
98: def find_name(seq, name):
99:     for item in seq:
100:         if item.name == name:
101:             return item
102: 
103: 
104: def iterate(c):
105:     '''
106:     Run the Huffman algorithm on the list of "nodes" c.
107:     The list of nodes c is destroyed as we go, then recreated.
108:     Codewords 'co.word' are assigned to each node during the recreation of the list.
109:     The order of the recreated list may well be different.
110:     Use the list c for encoding.
111: 
112:     The root of a new tree of "internalnodes" is returned.
113:     This root should be used when decoding.
114: 
115:     >>> c = [ node(0.5,1,'a'),  \
116:               node(0.25,2,'b'), \
117:               node(0.125,3,'c'),\
118:               node(0.125,4,'d') ]   # my doctest query has been resolved
119:     >>> root = iterate(c)           # "iterate(c)" returns a node, not nothing, and doctest cares!
120:     >>> reportcode(c)               # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
121:     #Symbol   Count     Codeword
122:     a         (0.5)     1
123:     b         (0.25)    01
124:     c         (0.12)    000
125:     d         (0.12)    001
126:     '''
127:     if (len(c) > 1):
128:         c.sort()  ## sort the nodes by count, using the __cmp__ function defined in the node class
129:         deletednode = c[0]  ## keep copy of smallest node so that we can put it back in later
130:         second = c[1].index  ## index of second smallest node
131:         # MERGE THE BOTTOM TWO
132:         c[1].count += c[0].count  ##  this merged node retains the name of the bigger leaf.
133:         del c[0]
134: 
135:         root = iterate(c)
136: 
137:         ## Fill in the new information in the ENCODING list, c
138:         ## find the codeword that has been split/joined at this step
139:         co = find_idx(c, second)
140:         #        co = find( lambda p: p.index == second , c )
141:         deletednode.word = co.word + '0'
142:         c.append(deletednode)  ## smaller one gets 0
143:         co.word += '1'
144:         co.count -= deletednode.count  ## restore correct count
145: 
146:         ## make the new branches in the DECODING tree
147:         newnode0 = internalnode()
148:         newnode1 = internalnode()
149:         treenode = co.internalnode  # find the node that got two children
150:         treenode.children(newnode0, newnode1)
151:         deletednode.associate(newnode0)
152:         co.associate(newnode1)
153:         pass
154:     else:
155:         c[0].word = ""
156:         root = internalnode()
157:         c[0].associate(root)
158:         pass
159:     return root
160: 
161: 
162: def encode(sourcelist, code):
163:     '''
164:     Takes a list of source symbols. Returns a binary string.
165:     '''
166:     answer = ""
167:     for s in sourcelist:
168:         co = find_name(code, s)
169:         #        co = find(lambda p: p.name == s, code)
170:         if (not co):
171:             print >> sys.stderr, "Warning: symbol", `s`, "has no encoding!"
172:             pass
173:         else:
174:             answer = answer + co.word
175:             pass
176:     return answer
177: 
178: 
179: def decode(string, root):
180:     '''
181:     Decodes a binary string using the Huffman tree accessed via root
182:     '''
183:     ## split the string into a list
184:     ## then copy the elements of the list one by one.
185:     answer = []
186:     clist = list(string)
187:     ## start from root
188:     currentnode = root
189:     for c in clist:
190:         if (c == '\n'):  continue  ## special case for newline characters
191:         assert (c == '0') or (c == '1')
192:         currentnode = currentnode.child[int(c)]
193:         if currentnode.leaf != 0:
194:             answer.append(str(currentnode.name))
195:             currentnode = root
196:         pass
197:     assert (
198:                 currentnode == root)  ## if this is not true then we have run out of characters and are half-way through a codeword
199:     return answer
200: 
201: 
202: ## alternate way of calling huffman with a list of counts ## for use as package by other programs ######
203: ## two example ways of using it:
204: # >>> from Huffman3 import *
205: # >>> huffman([1, 2, 3, 4],1)
206: # >>> (c,root) = huffman([1, 2, 3, 4])
207: 
208: ## end ##########################################################################
209: 
210: def makenodes(probs):
211:     '''
212:     Creates a list of nodes ready for the Huffman algorithm.
213:     Each node will receive a codeword when Huffman algorithm "iterate" runs.
214: 
215:     probs should be a list of pairs('<symbol>', <value>).
216: 
217:     >>> probs=[('a',0.5), ('b',0.25), ('c',0.125), ('d',0.125)]
218:     >>> symbols = makenodes(probs)
219:     >>> root = iterate(symbols)
220:     >>> zipped = encode(['a','a','b','a','c','b','c','d'], symbols)
221:     >>> print zipped
222:     1101100001000001
223:     >>> print decode( zipped, root )
224:     ['a', 'a', 'b', 'a', 'c', 'b', 'c', 'd']
225: 
226:     See also the file Example.py for a python program that uses this package.
227:     '''
228:     m = 0
229:     c = []
230:     for p in probs:
231:         m += 1;
232:         c.append(node(p[1], m, p[0]))
233:         pass
234:     return c
235: 
236: 
237: def dec_to_bin(n, digits):
238:     ''' n is the number to convert to binary;  digits is the number of bits you want
239:     Always prints full number of digits
240:     >>> print dec_to_bin( 17 , 9)
241:     000010001
242:     >>> print dec_to_bin( 17 , 5)
243:     10001
244: 
245:     Will behead the standard binary number if requested
246:     >>> print dec_to_bin( 17 , 4)
247:     0001
248:     '''
249:     if (n < 0):
250:         sys.stderr.write("warning, negative n not expected\n")
251:     i = digits - 1
252:     ans = ""
253:     while i >= 0:
254:         b = (((1 << i) & n) > 0)
255:         i -= 1
256:         ans = ans + str(int(b))
257:     return ans
258: 
259: 
260: verbose = 0
261: 
262: 
263: def weight(string):
264:     '''
265:     ## returns number of 0s and number of 1s in the string
266:     >>> print weight("00011")
267:     (3, 2)
268:     '''
269:     w0 = 0;
270:     w1 = 0
271:     for c in list(string):
272:         if (c == '0'):
273:             w0 += 1
274:             pass
275:         elif (c == '1'):
276:             w1 += 1
277:             pass
278:         pass
279:     return (w0, w1)
280: 
281: 
282: def findprobs(f=0.01, N=6):
283:     ''' Find probabilities of all the events
284:     000000
285:     000001
286:      ...
287:     111111
288:     <-N ->
289:     >>> print findprobs(0.1,3)              # doctest:+ELLIPSIS
290:     [('000', 0.7...),..., ('111', 0.001...)]
291:     '''
292:     answer = []
293:     for n in range(2 ** N):
294:         s = dec_to_bin(n, N)
295:         (w0, w1) = weight(s)
296:         if verbose and 0:
297:             pass  # print s,w0,w1
298:         answer.append((s, f ** w1 * (1 - f) ** w0))
299:         pass
300:     assert (len(answer) == 2 ** N)
301:     return answer
302: 
303: 
304: def Bencode(string, symbols, N):
305:     '''
306:     Reads in a string of 0s and 1s.
307:     Creates a list of blocks of size N.
308:     Sends this list to the general-purpose Huffman encoder
309:     defined by the nodes in the list "symbols".
310:     '''
311:     blocks = []
312:     chars = list(string)
313: 
314:     s = ""
315:     for c in chars:
316:         s = s + c
317:         if (len(s) >= N):
318:             blocks.append(s)
319:             s = ""
320:             pass
321:         pass
322:     if (len(s) > 0):
323:         print >> sys.stderr, "warning, padding last block with 0s"
324:         while (len(s) < N):
325:             s = s + '0'
326:             pass
327:         blocks.append(s)
328:         pass
329: 
330:     if verbose:
331:         # print "blocks to be encoded:"
332:         # print blocks
333:         pass
334:     zipped = encode(blocks, symbols)
335:     return zipped
336: 
337: 
338: def Bdecode(string, root, N):
339:     '''
340:     Decode a binary string into blocks.
341:     '''
342:     answer = decode(string, root)
343:     if verbose:
344:         # print "blocks from decoder:"
345:         # print answer
346:         pass
347:     output = "".join(answer)
348:     ## this assumes that the source file was an exact multiple of the blocklength
349:     return output
350: 
351: 
352: def easytest():
353:     '''
354:     Tests block code with N=3, f=0.01 on a tiny example.
355:     >>> easytest()                 # doctest:+NORMALIZE_WHITESPACE
356:     #Symbol     Count           Codeword
357:     000         (0.97)          1
358:     001         (0.0098)        001
359:     010         (0.0098)        010
360:     011         (9.9e-05)       00001
361:     100         (0.0098)        011
362:     101         (9.9e-05)       00010
363:     110         (9.9e-05)       00011
364:     111         (1e-06)         00000
365:     zipped  = 1001010000010110111
366:     decoded = ['000', '001', '010', '011', '100', '100', '000']
367:     OK!
368:     '''
369:     N = 3
370:     f = 0.01
371:     probs = findprobs(f, N)
372:     #    if len(probs) > 999 :
373:     #        sys.setrecursionlimit( len(probs)+100 )
374:     symbols = makenodes(probs)  # makenodes is defined at the bottom of Huffman3 package
375:     root = iterate(
376:         symbols)  # make huffman code and put it into the symbols' nodes, and return the root of the decoding tree
377: 
378:     symbols.sort(lambda x, y: cmp(x.index, y.index))  # sort by index
379:     for co in symbols:  # and write the answer
380:         co.report()
381: 
382:     source = ['000', '001', '010', '011', '100', '100', '000']
383:     zipped = encode(source, symbols)
384:     print "zipped  =",zipped
385:     answer = decode(zipped, root)
386:     print "decoded =",answer
387:     if (source != answer):
388:         print "ERROR"
389:     else:
390:         print "OK!"
391:     pass
392: 
393: 
394: def test():
395:     easytest()
396:     hardertest()
397: 
398: 
399: def Relative(path):
400:     return os.path.join(os.path.dirname(__file__), path)
401: 
402: 
403: def hardertest():
404:     # print "Reading the BentCoinFile"
405:     inputfile = open(Relative("testdata/BentCoinFile"), "r")
406:     outputfile = open(Relative("tmp.zip"), "w")
407:     # print  "Compressing to tmp.zip"
408:     compress_it(inputfile, outputfile)
409:     outputfile.close();
410:     inputfile.close()
411:     #    print "DONE compressing"
412: 
413:     inputfile = open(Relative("tmp.zip"), "r")
414:     outputfile = open(Relative("tmp2"), "w")
415:     # print  "Uncompressing to tmp2"
416:     uncompress_it(inputfile, outputfile)
417:     outputfile.close();
418:     inputfile.close()
419:     #    print "DONE uncompressing"
420: 
421:     ##    print "Checking for differences..."
422:     ##    os.system( "diff testdata/BentCoinFile tmp2" )
423:     ##    os.system( "wc tmp.zip testdata/BentCoinFile tmp2" )
424:     pass
425: 
426: 
427: f = 0.01;
428: N = 12  # 1244 bits if N==12
429: f = 0.01;
430: N = 5  # 2266  bits if N==5
431: f = 0.01;
432: N = 10  # 1379 bits if N==10
433: 
434: 
435: def compress_it(inputfile, outputfile):
436:     '''
437:     Make Huffman code for blocks, and
438:     Compress from file (possibly stdin).
439:     '''
440:     probs = findprobs(f, N)
441:     symbols = makenodes(probs)
442:     #    if len(probs) > 999 :
443:     #        sys.setrecursionlimit( len(probs)+100 )
444:     root = iterate(
445:         symbols)  # make huffman code and put it into the symbols' nodes, and return the root of the decoding tree
446: 
447:     string = inputfile.read()
448:     outputfile.write(Bencode(string, symbols, N))
449:     pass
450: 
451: 
452: def uncompress_it(inputfile, outputfile):
453:     '''
454:     Make Huffman code for blocks, and
455:     UNCompress from file (possibly stdin).
456:     '''
457:     probs = findprobs(f, N)
458:     #    if len(probs) > 999 :
459:     #        sys.setrecursionlimit( len(probs)+100 )
460:     symbols = makenodes(probs)
461:     root = iterate(
462:         symbols)  # make huffman code and put it into the symbols' nodes, and return the root of the decoding tree
463: 
464:     string = inputfile.read()
465:     outputfile.write(Bdecode(string, root, N))
466:     pass
467: 
468: 
469: def run():
470:     sys.setrecursionlimit(10000)
471:     test()
472:     return True
473: 
474: 
475: run()
476: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', "\nThis is a BLOCK compression algorithm\nthat uses the Huffman algorithm.\n\nThis simple block compressor assumes that the source file\nis an exact multiple of the block length.\nThe encoding does not itself delimit the size of the file, so\nthe decoder needs to knows where the end of the compressed\nfile is. Thus we must either ensure the decoder knows\nthe original file's length, or we have to use a special\nend-of-file character. A superior compressor would first\nencode the source file's length at the head of the compressed\nfile; then the decoder would be able to stop at the right\npoint and correct any truncation errors. I'll fix this\nin block2.py.\n\nThe package contains the following functions:\n\n findprobs(f=0.01,N=6):\n    Find probabilities of all the events\n    000000\n    000001\n     ...\n    111111\n    <-N ->\n\n Bencode(string,symbols,N):\n    Reads in a string of 0s and 1s, forms blocks, and encodes with Huffman.\n\n Bdecode(string,root,N):\n    Decode a binary string into blocks, then return appropriate 0s and 1s\n\n compress_it( inputfile, outputfile ):\n    Make Huffman code, and compress\n\n uncompress_it( inputfile, outputfile ):\n    Make Huffman code, and uncompress\n\n There are also three test functions.\n If block.py is run from a terminal, it invokes compress_it (using stdin)\n or uncompress_it (using stdin), respectively if there are zero arguments\n or one argument.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 0))

# Multiple import statement. import sys (1/2) (line 51)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/2) (line 51)
import os

import_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'os', os, module_type_store)

# Declaration of the 'node' class

class node:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'str', '')
        defaults = [str_5]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.__init__', ['count', 'index', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['count', 'index', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 56):
        
        # Assigning a Call to a Attribute (line 56):
        
        # Call to float(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'count' (line 56)
        count_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'count', False)
        # Processing the call keyword arguments (line 56)
        kwargs_8 = {}
        # Getting the type of 'float' (line 56)
        float_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'float', False)
        # Calling float(args, kwargs) (line 56)
        float_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), float_6, *[count_7], **kwargs_8)
        
        # Getting the type of 'self' (line 56)
        self_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'count' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_10, 'count', float_call_result_9)
        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'index' (line 57)
        index_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'index')
        # Getting the type of 'self' (line 57)
        self_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'index' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_12, 'index', index_11)
        
        # Assigning a Name to a Attribute (line 58):
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'name' (line 58)
        name_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'name')
        # Getting the type of 'self' (line 58)
        self_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'name' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_14, 'name', name_13)
        
        # Getting the type of 'self' (line 59)
        self_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'self')
        # Obtaining the member 'name' of a type (line 59)
        name_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), self_15, 'name')
        str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'str', '')
        # Applying the binary operator '==' (line 59)
        result_eq_18 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), '==', name_16, str_17)
        
        # Testing if the type of an if condition is none (line 59)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 8), result_eq_18):
            pass
        else:
            
            # Testing the type of an if condition (line 59)
            if_condition_19 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_eq_18)
            # Assigning a type to the variable 'if_condition_19' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_19', if_condition_19)
            # SSA begins for if statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Attribute (line 59):
            
            # Assigning a BinOp to a Attribute (line 59):
            str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 40), 'str', '_')
            
            # Call to str(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'index' (line 59)
            index_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 50), 'index', False)
            # Processing the call keyword arguments (line 59)
            kwargs_23 = {}
            # Getting the type of 'str' (line 59)
            str_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'str', False)
            # Calling str(args, kwargs) (line 59)
            str_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 59, 46), str_21, *[index_22], **kwargs_23)
            
            # Applying the binary operator '+' (line 59)
            result_add_25 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 40), '+', str_20, str_call_result_24)
            
            # Getting the type of 'self' (line 59)
            self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'self')
            # Setting the type of the member 'name' of a type (line 59)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 28), self_26, 'name', result_add_25)
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Attribute (line 60):
        
        # Assigning a Str to a Attribute (line 60):
        str_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'str', '')
        # Getting the type of 'self' (line 60)
        self_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member 'word' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_28, 'word', str_27)
        
        # Assigning a Num to a Attribute (line 61):
        
        # Assigning a Num to a Attribute (line 61):
        int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'int')
        # Getting the type of 'self' (line 61)
        self_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'isinternal' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_30, 'isinternal', int_29)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        node.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        node.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        node.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        node.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'node.stypy__cmp__')
        node.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        node.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        node.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        node.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        node.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        node.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        node.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.stypy__cmp__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__cmp__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__cmp__(...)' code ##################

        
        # Call to cmp(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'self', False)
        # Obtaining the member 'count' of a type (line 64)
        count_33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), self_32, 'count')
        # Getting the type of 'other' (line 64)
        other_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'other', False)
        # Obtaining the member 'count' of a type (line 64)
        count_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 31), other_34, 'count')
        # Processing the call keyword arguments (line 64)
        kwargs_36 = {}
        # Getting the type of 'cmp' (line 64)
        cmp_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'cmp', False)
        # Calling cmp(args, kwargs) (line 64)
        cmp_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), cmp_31, *[count_33, count_35], **kwargs_36)
        
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', cmp_call_result_37)
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_38


    @norecursion
    def report(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'report'
        module_type_store = module_type_store.open_function_context('report', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        node.report.__dict__.__setitem__('stypy_localization', localization)
        node.report.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        node.report.__dict__.__setitem__('stypy_type_store', module_type_store)
        node.report.__dict__.__setitem__('stypy_function_name', 'node.report')
        node.report.__dict__.__setitem__('stypy_param_names_list', [])
        node.report.__dict__.__setitem__('stypy_varargs_param_name', None)
        node.report.__dict__.__setitem__('stypy_kwargs_param_name', None)
        node.report.__dict__.__setitem__('stypy_call_defaults', defaults)
        node.report.__dict__.__setitem__('stypy_call_varargs', varargs)
        node.report.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        node.report.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.report', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'report', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'report(...)' code ##################

        
        # Getting the type of 'self' (line 67)
        self_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'self')
        # Obtaining the member 'index' of a type (line 67)
        index_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), self_39, 'index')
        int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'int')
        # Applying the binary operator '==' (line 67)
        result_eq_42 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 12), '==', index_40, int_41)
        
        # Testing if the type of an if condition is none (line 67)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_42):
            pass
        else:
            
            # Testing the type of an if condition (line 67)
            if_condition_43 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_42)
            # Assigning a type to the variable 'if_condition_43' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_43', if_condition_43)
            # SSA begins for if statement (line 67)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 67)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        
        # ################# End of 'report(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'report' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'report'
        return stypy_return_type_44


    @norecursion
    def associate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'associate'
        module_type_store = module_type_store.open_function_context('associate', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        node.associate.__dict__.__setitem__('stypy_localization', localization)
        node.associate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        node.associate.__dict__.__setitem__('stypy_type_store', module_type_store)
        node.associate.__dict__.__setitem__('stypy_function_name', 'node.associate')
        node.associate.__dict__.__setitem__('stypy_param_names_list', ['internalnode'])
        node.associate.__dict__.__setitem__('stypy_varargs_param_name', None)
        node.associate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        node.associate.__dict__.__setitem__('stypy_call_defaults', defaults)
        node.associate.__dict__.__setitem__('stypy_call_varargs', varargs)
        node.associate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        node.associate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.associate', ['internalnode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'associate', localization, ['internalnode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'associate(...)' code ##################

        
        # Assigning a Name to a Attribute (line 73):
        
        # Assigning a Name to a Attribute (line 73):
        # Getting the type of 'internalnode' (line 73)
        internalnode_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'internalnode')
        # Getting the type of 'self' (line 73)
        self_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'internalnode' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_46, 'internalnode', internalnode_45)
        
        # Assigning a Num to a Attribute (line 74):
        
        # Assigning a Num to a Attribute (line 74):
        int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'int')
        # Getting the type of 'internalnode' (line 74)
        internalnode_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'internalnode')
        # Setting the type of the member 'leaf' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), internalnode_48, 'leaf', int_47)
        
        # Assigning a Attribute to a Attribute (line 75):
        
        # Assigning a Attribute to a Attribute (line 75):
        # Getting the type of 'self' (line 75)
        self_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'self')
        # Obtaining the member 'name' of a type (line 75)
        name_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 28), self_49, 'name')
        # Getting the type of 'internalnode' (line 75)
        internalnode_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'internalnode')
        # Setting the type of the member 'name' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), internalnode_51, 'name', name_50)
        pass
        
        # ################# End of 'associate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'associate' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'associate'
        return stypy_return_type_52


# Assigning a type to the variable 'node' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'node', node)
# Declaration of the 'internalnode' class

class internalnode:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'internalnode.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 81):
        
        # Assigning a Num to a Attribute (line 81):
        int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'int')
        # Getting the type of 'self' (line 81)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self')
        # Setting the type of the member 'leaf' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_54, 'leaf', int_53)
        
        # Assigning a List to a Attribute (line 82):
        
        # Assigning a List to a Attribute (line 82):
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        
        # Getting the type of 'self' (line 82)
        self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'child' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_56, 'child', list_55)
        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def children(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'children'
        module_type_store = module_type_store.open_function_context('children', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        internalnode.children.__dict__.__setitem__('stypy_localization', localization)
        internalnode.children.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        internalnode.children.__dict__.__setitem__('stypy_type_store', module_type_store)
        internalnode.children.__dict__.__setitem__('stypy_function_name', 'internalnode.children')
        internalnode.children.__dict__.__setitem__('stypy_param_names_list', ['child0', 'child1'])
        internalnode.children.__dict__.__setitem__('stypy_varargs_param_name', None)
        internalnode.children.__dict__.__setitem__('stypy_kwargs_param_name', None)
        internalnode.children.__dict__.__setitem__('stypy_call_defaults', defaults)
        internalnode.children.__dict__.__setitem__('stypy_call_varargs', varargs)
        internalnode.children.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        internalnode.children.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'internalnode.children', ['child0', 'child1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'children', localization, ['child0', 'child1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'children(...)' code ##################

        
        # Assigning a Num to a Attribute (line 86):
        
        # Assigning a Num to a Attribute (line 86):
        int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'int')
        # Getting the type of 'self' (line 86)
        self_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member 'leaf' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_58, 'leaf', int_57)
        
        # Call to append(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'child0' (line 87)
        child0_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'child0', False)
        # Processing the call keyword arguments (line 87)
        kwargs_63 = {}
        # Getting the type of 'self' (line 87)
        self_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member 'child' of a type (line 87)
        child_60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_59, 'child')
        # Obtaining the member 'append' of a type (line 87)
        append_61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), child_60, 'append')
        # Calling append(args, kwargs) (line 87)
        append_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), append_61, *[child0_62], **kwargs_63)
        
        
        # Call to append(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'child1' (line 88)
        child1_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'child1', False)
        # Processing the call keyword arguments (line 88)
        kwargs_69 = {}
        # Getting the type of 'self' (line 88)
        self_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member 'child' of a type (line 88)
        child_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_65, 'child')
        # Obtaining the member 'append' of a type (line 88)
        append_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), child_66, 'append')
        # Calling append(args, kwargs) (line 88)
        append_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), append_67, *[child1_68], **kwargs_69)
        
        pass
        
        # ################# End of 'children(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'children' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'children'
        return stypy_return_type_71


# Assigning a type to the variable 'internalnode' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'internalnode', internalnode)

@norecursion
def find_idx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_idx'
    module_type_store = module_type_store.open_function_context('find_idx', 92, 0, False)
    
    # Passed parameters checking function
    find_idx.stypy_localization = localization
    find_idx.stypy_type_of_self = None
    find_idx.stypy_type_store = module_type_store
    find_idx.stypy_function_name = 'find_idx'
    find_idx.stypy_param_names_list = ['seq', 'index']
    find_idx.stypy_varargs_param_name = None
    find_idx.stypy_kwargs_param_name = None
    find_idx.stypy_call_defaults = defaults
    find_idx.stypy_call_varargs = varargs
    find_idx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_idx', ['seq', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_idx', localization, ['seq', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_idx(...)' code ##################

    
    # Getting the type of 'seq' (line 93)
    seq_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'seq')
    # Assigning a type to the variable 'seq_72' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'seq_72', seq_72)
    # Testing if the for loop is going to be iterated (line 93)
    # Testing the type of a for loop iterable (line 93)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 4), seq_72)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 93, 4), seq_72):
        # Getting the type of the for loop variable (line 93)
        for_loop_var_73 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 4), seq_72)
        # Assigning a type to the variable 'item' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'item', for_loop_var_73)
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'item' (line 94)
        item_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'item')
        # Obtaining the member 'index' of a type (line 94)
        index_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), item_74, 'index')
        # Getting the type of 'index' (line 94)
        index_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'index')
        # Applying the binary operator '==' (line 94)
        result_eq_77 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 11), '==', index_75, index_76)
        
        # Testing if the type of an if condition is none (line 94)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 94, 8), result_eq_77):
            pass
        else:
            
            # Testing the type of an if condition (line 94)
            if_condition_78 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), result_eq_77)
            # Assigning a type to the variable 'if_condition_78' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_78', if_condition_78)
            # SSA begins for if statement (line 94)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'item' (line 95)
            item_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'item')
            # Assigning a type to the variable 'stypy_return_type' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'stypy_return_type', item_79)
            # SSA join for if statement (line 94)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'find_idx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_idx' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_80)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_idx'
    return stypy_return_type_80

# Assigning a type to the variable 'find_idx' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'find_idx', find_idx)

@norecursion
def find_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_name'
    module_type_store = module_type_store.open_function_context('find_name', 98, 0, False)
    
    # Passed parameters checking function
    find_name.stypy_localization = localization
    find_name.stypy_type_of_self = None
    find_name.stypy_type_store = module_type_store
    find_name.stypy_function_name = 'find_name'
    find_name.stypy_param_names_list = ['seq', 'name']
    find_name.stypy_varargs_param_name = None
    find_name.stypy_kwargs_param_name = None
    find_name.stypy_call_defaults = defaults
    find_name.stypy_call_varargs = varargs
    find_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_name', ['seq', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_name', localization, ['seq', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_name(...)' code ##################

    
    # Getting the type of 'seq' (line 99)
    seq_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'seq')
    # Assigning a type to the variable 'seq_81' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'seq_81', seq_81)
    # Testing if the for loop is going to be iterated (line 99)
    # Testing the type of a for loop iterable (line 99)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 4), seq_81)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 99, 4), seq_81):
        # Getting the type of the for loop variable (line 99)
        for_loop_var_82 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 4), seq_81)
        # Assigning a type to the variable 'item' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'item', for_loop_var_82)
        # SSA begins for a for statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'item' (line 100)
        item_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'item')
        # Obtaining the member 'name' of a type (line 100)
        name_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 11), item_83, 'name')
        # Getting the type of 'name' (line 100)
        name_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'name')
        # Applying the binary operator '==' (line 100)
        result_eq_86 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), '==', name_84, name_85)
        
        # Testing if the type of an if condition is none (line 100)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 8), result_eq_86):
            pass
        else:
            
            # Testing the type of an if condition (line 100)
            if_condition_87 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_eq_86)
            # Assigning a type to the variable 'if_condition_87' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_87', if_condition_87)
            # SSA begins for if statement (line 100)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'item' (line 101)
            item_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'item')
            # Assigning a type to the variable 'stypy_return_type' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', item_88)
            # SSA join for if statement (line 100)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'find_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_name' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_name'
    return stypy_return_type_89

# Assigning a type to the variable 'find_name' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'find_name', find_name)

@norecursion
def iterate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterate'
    module_type_store = module_type_store.open_function_context('iterate', 104, 0, False)
    
    # Passed parameters checking function
    iterate.stypy_localization = localization
    iterate.stypy_type_of_self = None
    iterate.stypy_type_store = module_type_store
    iterate.stypy_function_name = 'iterate'
    iterate.stypy_param_names_list = ['c']
    iterate.stypy_varargs_param_name = None
    iterate.stypy_kwargs_param_name = None
    iterate.stypy_call_defaults = defaults
    iterate.stypy_call_varargs = varargs
    iterate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterate', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterate', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterate(...)' code ##################

    str_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, (-1)), 'str', '\n    Run the Huffman algorithm on the list of "nodes" c.\n    The list of nodes c is destroyed as we go, then recreated.\n    Codewords \'co.word\' are assigned to each node during the recreation of the list.\n    The order of the recreated list may well be different.\n    Use the list c for encoding.\n\n    The root of a new tree of "internalnodes" is returned.\n    This root should be used when decoding.\n\n    >>> c = [ node(0.5,1,\'a\'),                node(0.25,2,\'b\'),               node(0.125,3,\'c\'),              node(0.125,4,\'d\') ]   # my doctest query has been resolved\n    >>> root = iterate(c)           # "iterate(c)" returns a node, not nothing, and doctest cares!\n    >>> reportcode(c)               # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS\n    #Symbol   Count     Codeword\n    a         (0.5)     1\n    b         (0.25)    01\n    c         (0.12)    000\n    d         (0.12)    001\n    ')
    
    
    # Call to len(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'c' (line 127)
    c_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'c', False)
    # Processing the call keyword arguments (line 127)
    kwargs_93 = {}
    # Getting the type of 'len' (line 127)
    len_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'len', False)
    # Calling len(args, kwargs) (line 127)
    len_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), len_91, *[c_92], **kwargs_93)
    
    int_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'int')
    # Applying the binary operator '>' (line 127)
    result_gt_96 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 8), '>', len_call_result_94, int_95)
    
    # Testing if the type of an if condition is none (line 127)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 4), result_gt_96):
        
        # Assigning a Str to a Attribute (line 155):
        
        # Assigning a Str to a Attribute (line 155):
        str_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'str', '')
        
        # Obtaining the type of the subscript
        int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 10), 'int')
        # Getting the type of 'c' (line 155)
        c_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), c_187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), getitem___188, int_186)
        
        # Setting the type of the member 'word' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), subscript_call_result_189, 'word', str_185)
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to internalnode(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_191 = {}
        # Getting the type of 'internalnode' (line 156)
        internalnode_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 156)
        internalnode_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), internalnode_190, *[], **kwargs_191)
        
        # Assigning a type to the variable 'root' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'root', internalnode_call_result_192)
        
        # Call to associate(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'root' (line 157)
        root_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 23), 'root', False)
        # Processing the call keyword arguments (line 157)
        kwargs_199 = {}
        
        # Obtaining the type of the subscript
        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 10), 'int')
        # Getting the type of 'c' (line 157)
        c_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), c_194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___195, int_193)
        
        # Obtaining the member 'associate' of a type (line 157)
        associate_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), subscript_call_result_196, 'associate')
        # Calling associate(args, kwargs) (line 157)
        associate_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), associate_197, *[root_198], **kwargs_199)
        
        pass
    else:
        
        # Testing the type of an if condition (line 127)
        if_condition_97 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 4), result_gt_96)
        # Assigning a type to the variable 'if_condition_97' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'if_condition_97', if_condition_97)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to sort(...): (line 128)
        # Processing the call keyword arguments (line 128)
        kwargs_100 = {}
        # Getting the type of 'c' (line 128)
        c_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'c', False)
        # Obtaining the member 'sort' of a type (line 128)
        sort_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), c_98, 'sort')
        # Calling sort(args, kwargs) (line 128)
        sort_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), sort_99, *[], **kwargs_100)
        
        
        # Assigning a Subscript to a Name (line 129):
        
        # Assigning a Subscript to a Name (line 129):
        
        # Obtaining the type of the subscript
        int_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'int')
        # Getting the type of 'c' (line 129)
        c_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 22), 'c')
        # Obtaining the member '__getitem__' of a type (line 129)
        getitem___104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 22), c_103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 129)
        subscript_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 129, 22), getitem___104, int_102)
        
        # Assigning a type to the variable 'deletednode' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'deletednode', subscript_call_result_105)
        
        # Assigning a Attribute to a Name (line 130):
        
        # Assigning a Attribute to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'int')
        # Getting the type of 'c' (line 130)
        c_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'c')
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 17), c_107, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 130, 17), getitem___108, int_106)
        
        # Obtaining the member 'index' of a type (line 130)
        index_110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 17), subscript_call_result_109, 'index')
        # Assigning a type to the variable 'second' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'second', index_110)
        
        
        # Obtaining the type of the subscript
        int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 10), 'int')
        # Getting the type of 'c' (line 132)
        c_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), c_112, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), getitem___113, int_111)
        
        # Obtaining the member 'count' of a type (line 132)
        count_115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), subscript_call_result_114, 'count')
        
        # Obtaining the type of the subscript
        int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'int')
        # Getting the type of 'c' (line 132)
        c_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 22), 'c')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 22), c_117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 132, 22), getitem___118, int_116)
        
        # Obtaining the member 'count' of a type (line 132)
        count_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 22), subscript_call_result_119, 'count')
        # Applying the binary operator '+=' (line 132)
        result_iadd_121 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 8), '+=', count_115, count_120)
        
        # Obtaining the type of the subscript
        int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 10), 'int')
        # Getting the type of 'c' (line 132)
        c_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), c_123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), getitem___124, int_122)
        
        # Setting the type of the member 'count' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), subscript_call_result_125, 'count', result_iadd_121)
        
        # Deleting a member
        # Getting the type of 'c' (line 133)
        c_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'c')
        
        # Obtaining the type of the subscript
        int_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 14), 'int')
        # Getting the type of 'c' (line 133)
        c_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'c')
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), c_128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), getitem___129, int_127)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 8), c_126, subscript_call_result_130)
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to iterate(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'c' (line 135)
        c_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'c', False)
        # Processing the call keyword arguments (line 135)
        kwargs_133 = {}
        # Getting the type of 'iterate' (line 135)
        iterate_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'iterate', False)
        # Calling iterate(args, kwargs) (line 135)
        iterate_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), iterate_131, *[c_132], **kwargs_133)
        
        # Assigning a type to the variable 'root' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'root', iterate_call_result_134)
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to find_idx(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'c' (line 139)
        c_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'c', False)
        # Getting the type of 'second' (line 139)
        second_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'second', False)
        # Processing the call keyword arguments (line 139)
        kwargs_138 = {}
        # Getting the type of 'find_idx' (line 139)
        find_idx_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 13), 'find_idx', False)
        # Calling find_idx(args, kwargs) (line 139)
        find_idx_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 139, 13), find_idx_135, *[c_136, second_137], **kwargs_138)
        
        # Assigning a type to the variable 'co' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'co', find_idx_call_result_139)
        
        # Assigning a BinOp to a Attribute (line 141):
        
        # Assigning a BinOp to a Attribute (line 141):
        # Getting the type of 'co' (line 141)
        co_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'co')
        # Obtaining the member 'word' of a type (line 141)
        word_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 27), co_140, 'word')
        str_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 37), 'str', '0')
        # Applying the binary operator '+' (line 141)
        result_add_143 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 27), '+', word_141, str_142)
        
        # Getting the type of 'deletednode' (line 141)
        deletednode_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'deletednode')
        # Setting the type of the member 'word' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), deletednode_144, 'word', result_add_143)
        
        # Call to append(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'deletednode' (line 142)
        deletednode_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'deletednode', False)
        # Processing the call keyword arguments (line 142)
        kwargs_148 = {}
        # Getting the type of 'c' (line 142)
        c_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'c', False)
        # Obtaining the member 'append' of a type (line 142)
        append_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), c_145, 'append')
        # Calling append(args, kwargs) (line 142)
        append_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), append_146, *[deletednode_147], **kwargs_148)
        
        
        # Getting the type of 'co' (line 143)
        co_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'co')
        # Obtaining the member 'word' of a type (line 143)
        word_151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), co_150, 'word')
        str_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'str', '1')
        # Applying the binary operator '+=' (line 143)
        result_iadd_153 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 8), '+=', word_151, str_152)
        # Getting the type of 'co' (line 143)
        co_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'co')
        # Setting the type of the member 'word' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), co_154, 'word', result_iadd_153)
        
        
        # Getting the type of 'co' (line 144)
        co_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'co')
        # Obtaining the member 'count' of a type (line 144)
        count_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), co_155, 'count')
        # Getting the type of 'deletednode' (line 144)
        deletednode_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'deletednode')
        # Obtaining the member 'count' of a type (line 144)
        count_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), deletednode_157, 'count')
        # Applying the binary operator '-=' (line 144)
        result_isub_159 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 8), '-=', count_156, count_158)
        # Getting the type of 'co' (line 144)
        co_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'co')
        # Setting the type of the member 'count' of a type (line 144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), co_160, 'count', result_isub_159)
        
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to internalnode(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_162 = {}
        # Getting the type of 'internalnode' (line 147)
        internalnode_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 147)
        internalnode_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), internalnode_161, *[], **kwargs_162)
        
        # Assigning a type to the variable 'newnode0' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'newnode0', internalnode_call_result_163)
        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to internalnode(...): (line 148)
        # Processing the call keyword arguments (line 148)
        kwargs_165 = {}
        # Getting the type of 'internalnode' (line 148)
        internalnode_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 148)
        internalnode_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), internalnode_164, *[], **kwargs_165)
        
        # Assigning a type to the variable 'newnode1' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'newnode1', internalnode_call_result_166)
        
        # Assigning a Attribute to a Name (line 149):
        
        # Assigning a Attribute to a Name (line 149):
        # Getting the type of 'co' (line 149)
        co_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'co')
        # Obtaining the member 'internalnode' of a type (line 149)
        internalnode_168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), co_167, 'internalnode')
        # Assigning a type to the variable 'treenode' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'treenode', internalnode_168)
        
        # Call to children(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'newnode0' (line 150)
        newnode0_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'newnode0', False)
        # Getting the type of 'newnode1' (line 150)
        newnode1_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 36), 'newnode1', False)
        # Processing the call keyword arguments (line 150)
        kwargs_173 = {}
        # Getting the type of 'treenode' (line 150)
        treenode_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'treenode', False)
        # Obtaining the member 'children' of a type (line 150)
        children_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), treenode_169, 'children')
        # Calling children(args, kwargs) (line 150)
        children_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), children_170, *[newnode0_171, newnode1_172], **kwargs_173)
        
        
        # Call to associate(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'newnode0' (line 151)
        newnode0_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'newnode0', False)
        # Processing the call keyword arguments (line 151)
        kwargs_178 = {}
        # Getting the type of 'deletednode' (line 151)
        deletednode_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'deletednode', False)
        # Obtaining the member 'associate' of a type (line 151)
        associate_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), deletednode_175, 'associate')
        # Calling associate(args, kwargs) (line 151)
        associate_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), associate_176, *[newnode0_177], **kwargs_178)
        
        
        # Call to associate(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'newnode1' (line 152)
        newnode1_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'newnode1', False)
        # Processing the call keyword arguments (line 152)
        kwargs_183 = {}
        # Getting the type of 'co' (line 152)
        co_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'co', False)
        # Obtaining the member 'associate' of a type (line 152)
        associate_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), co_180, 'associate')
        # Calling associate(args, kwargs) (line 152)
        associate_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), associate_181, *[newnode1_182], **kwargs_183)
        
        pass
        # SSA branch for the else part of an if statement (line 127)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Attribute (line 155):
        
        # Assigning a Str to a Attribute (line 155):
        str_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'str', '')
        
        # Obtaining the type of the subscript
        int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 10), 'int')
        # Getting the type of 'c' (line 155)
        c_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), c_187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), getitem___188, int_186)
        
        # Setting the type of the member 'word' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), subscript_call_result_189, 'word', str_185)
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to internalnode(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_191 = {}
        # Getting the type of 'internalnode' (line 156)
        internalnode_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 156)
        internalnode_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), internalnode_190, *[], **kwargs_191)
        
        # Assigning a type to the variable 'root' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'root', internalnode_call_result_192)
        
        # Call to associate(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'root' (line 157)
        root_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 23), 'root', False)
        # Processing the call keyword arguments (line 157)
        kwargs_199 = {}
        
        # Obtaining the type of the subscript
        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 10), 'int')
        # Getting the type of 'c' (line 157)
        c_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), c_194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___195, int_193)
        
        # Obtaining the member 'associate' of a type (line 157)
        associate_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), subscript_call_result_196, 'associate')
        # Calling associate(args, kwargs) (line 157)
        associate_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), associate_197, *[root_198], **kwargs_199)
        
        pass
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'root' (line 159)
    root_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'root')
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type', root_201)
    
    # ################# End of 'iterate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterate' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_202)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterate'
    return stypy_return_type_202

# Assigning a type to the variable 'iterate' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'iterate', iterate)

@norecursion
def encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'encode'
    module_type_store = module_type_store.open_function_context('encode', 162, 0, False)
    
    # Passed parameters checking function
    encode.stypy_localization = localization
    encode.stypy_type_of_self = None
    encode.stypy_type_store = module_type_store
    encode.stypy_function_name = 'encode'
    encode.stypy_param_names_list = ['sourcelist', 'code']
    encode.stypy_varargs_param_name = None
    encode.stypy_kwargs_param_name = None
    encode.stypy_call_defaults = defaults
    encode.stypy_call_varargs = varargs
    encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode', ['sourcelist', 'code'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode', localization, ['sourcelist', 'code'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode(...)' code ##################

    str_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'str', '\n    Takes a list of source symbols. Returns a binary string.\n    ')
    
    # Assigning a Str to a Name (line 166):
    
    # Assigning a Str to a Name (line 166):
    str_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'str', '')
    # Assigning a type to the variable 'answer' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'answer', str_204)
    
    # Getting the type of 'sourcelist' (line 167)
    sourcelist_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'sourcelist')
    # Assigning a type to the variable 'sourcelist_205' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'sourcelist_205', sourcelist_205)
    # Testing if the for loop is going to be iterated (line 167)
    # Testing the type of a for loop iterable (line 167)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 167, 4), sourcelist_205)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 167, 4), sourcelist_205):
        # Getting the type of the for loop variable (line 167)
        for_loop_var_206 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 167, 4), sourcelist_205)
        # Assigning a type to the variable 's' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 's', for_loop_var_206)
        # SSA begins for a for statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to find_name(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'code' (line 168)
        code_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'code', False)
        # Getting the type of 's' (line 168)
        s_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 's', False)
        # Processing the call keyword arguments (line 168)
        kwargs_210 = {}
        # Getting the type of 'find_name' (line 168)
        find_name_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'find_name', False)
        # Calling find_name(args, kwargs) (line 168)
        find_name_call_result_211 = invoke(stypy.reporting.localization.Localization(__file__, 168, 13), find_name_207, *[code_208, s_209], **kwargs_210)
        
        # Assigning a type to the variable 'co' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'co', find_name_call_result_211)
        
        # Getting the type of 'co' (line 170)
        co_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'co')
        # Applying the 'not' unary operator (line 170)
        result_not__213 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 12), 'not', co_212)
        
        # Testing if the type of an if condition is none (line 170)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 8), result_not__213):
            
            # Assigning a BinOp to a Name (line 174):
            
            # Assigning a BinOp to a Name (line 174):
            # Getting the type of 'answer' (line 174)
            answer_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'answer')
            # Getting the type of 'co' (line 174)
            co_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'co')
            # Obtaining the member 'word' of a type (line 174)
            word_220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 30), co_219, 'word')
            # Applying the binary operator '+' (line 174)
            result_add_221 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 21), '+', answer_218, word_220)
            
            # Assigning a type to the variable 'answer' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'answer', result_add_221)
            pass
        else:
            
            # Testing the type of an if condition (line 170)
            if_condition_214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 8), result_not__213)
            # Assigning a type to the variable 'if_condition_214' (line 170)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'if_condition_214', if_condition_214)
            # SSA begins for if statement (line 170)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 33), 'str', 'Warning: symbol')
            # Getting the type of 's' (line 171)
            s_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 53), 's')
            str_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 57), 'str', 'has no encoding!')
            pass
            # SSA branch for the else part of an if statement (line 170)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 174):
            
            # Assigning a BinOp to a Name (line 174):
            # Getting the type of 'answer' (line 174)
            answer_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'answer')
            # Getting the type of 'co' (line 174)
            co_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'co')
            # Obtaining the member 'word' of a type (line 174)
            word_220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 30), co_219, 'word')
            # Applying the binary operator '+' (line 174)
            result_add_221 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 21), '+', answer_218, word_220)
            
            # Assigning a type to the variable 'answer' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'answer', result_add_221)
            pass
            # SSA join for if statement (line 170)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'answer' (line 176)
    answer_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'answer')
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type', answer_222)
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 162)
    stypy_return_type_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_223)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_223

# Assigning a type to the variable 'encode' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'encode', encode)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'decode'
    module_type_store = module_type_store.open_function_context('decode', 179, 0, False)
    
    # Passed parameters checking function
    decode.stypy_localization = localization
    decode.stypy_type_of_self = None
    decode.stypy_type_store = module_type_store
    decode.stypy_function_name = 'decode'
    decode.stypy_param_names_list = ['string', 'root']
    decode.stypy_varargs_param_name = None
    decode.stypy_kwargs_param_name = None
    decode.stypy_call_defaults = defaults
    decode.stypy_call_varargs = varargs
    decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode', ['string', 'root'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode', localization, ['string', 'root'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode(...)' code ##################

    str_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, (-1)), 'str', '\n    Decodes a binary string using the Huffman tree accessed via root\n    ')
    
    # Assigning a List to a Name (line 185):
    
    # Assigning a List to a Name (line 185):
    
    # Obtaining an instance of the builtin type 'list' (line 185)
    list_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 185)
    
    # Assigning a type to the variable 'answer' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'answer', list_225)
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to list(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'string' (line 186)
    string_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'string', False)
    # Processing the call keyword arguments (line 186)
    kwargs_228 = {}
    # Getting the type of 'list' (line 186)
    list_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'list', False)
    # Calling list(args, kwargs) (line 186)
    list_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), list_226, *[string_227], **kwargs_228)
    
    # Assigning a type to the variable 'clist' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'clist', list_call_result_229)
    
    # Assigning a Name to a Name (line 188):
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'root' (line 188)
    root_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'root')
    # Assigning a type to the variable 'currentnode' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'currentnode', root_230)
    
    # Getting the type of 'clist' (line 189)
    clist_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), 'clist')
    # Assigning a type to the variable 'clist_231' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'clist_231', clist_231)
    # Testing if the for loop is going to be iterated (line 189)
    # Testing the type of a for loop iterable (line 189)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 189, 4), clist_231)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 189, 4), clist_231):
        # Getting the type of the for loop variable (line 189)
        for_loop_var_232 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 189, 4), clist_231)
        # Assigning a type to the variable 'c' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'c', for_loop_var_232)
        # SSA begins for a for statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'c' (line 190)
        c_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'c')
        str_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 17), 'str', '\n')
        # Applying the binary operator '==' (line 190)
        result_eq_235 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 12), '==', c_233, str_234)
        
        # Testing if the type of an if condition is none (line 190)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 190, 8), result_eq_235):
            pass
        else:
            
            # Testing the type of an if condition (line 190)
            if_condition_236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), result_eq_235)
            # Assigning a type to the variable 'if_condition_236' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_236', if_condition_236)
            # SSA begins for if statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 190)
            module_type_store = module_type_store.join_ssa_context()
            

        # Evaluating assert statement condition
        
        # Evaluating a boolean operation
        
        # Getting the type of 'c' (line 191)
        c_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'c')
        str_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 21), 'str', '0')
        # Applying the binary operator '==' (line 191)
        result_eq_239 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '==', c_237, str_238)
        
        
        # Getting the type of 'c' (line 191)
        c_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'c')
        str_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 35), 'str', '1')
        # Applying the binary operator '==' (line 191)
        result_eq_242 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 30), '==', c_240, str_241)
        
        # Applying the binary operator 'or' (line 191)
        result_or_keyword_243 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), 'or', result_eq_239, result_eq_242)
        
        assert_244 = result_or_keyword_243
        # Assigning a type to the variable 'assert_244' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'assert_244', result_or_keyword_243)
        
        # Assigning a Subscript to a Name (line 192):
        
        # Assigning a Subscript to a Name (line 192):
        
        # Obtaining the type of the subscript
        
        # Call to int(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'c' (line 192)
        c_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'c', False)
        # Processing the call keyword arguments (line 192)
        kwargs_247 = {}
        # Getting the type of 'int' (line 192)
        int_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 40), 'int', False)
        # Calling int(args, kwargs) (line 192)
        int_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 192, 40), int_245, *[c_246], **kwargs_247)
        
        # Getting the type of 'currentnode' (line 192)
        currentnode_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 22), 'currentnode')
        # Obtaining the member 'child' of a type (line 192)
        child_250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 22), currentnode_249, 'child')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 22), child_250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 192, 22), getitem___251, int_call_result_248)
        
        # Assigning a type to the variable 'currentnode' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'currentnode', subscript_call_result_252)
        
        # Getting the type of 'currentnode' (line 193)
        currentnode_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'currentnode')
        # Obtaining the member 'leaf' of a type (line 193)
        leaf_254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 11), currentnode_253, 'leaf')
        int_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 31), 'int')
        # Applying the binary operator '!=' (line 193)
        result_ne_256 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), '!=', leaf_254, int_255)
        
        # Testing if the type of an if condition is none (line 193)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 193, 8), result_ne_256):
            pass
        else:
            
            # Testing the type of an if condition (line 193)
            if_condition_257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), result_ne_256)
            # Assigning a type to the variable 'if_condition_257' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_257', if_condition_257)
            # SSA begins for if statement (line 193)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 194)
            # Processing the call arguments (line 194)
            
            # Call to str(...): (line 194)
            # Processing the call arguments (line 194)
            # Getting the type of 'currentnode' (line 194)
            currentnode_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 30), 'currentnode', False)
            # Obtaining the member 'name' of a type (line 194)
            name_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 30), currentnode_261, 'name')
            # Processing the call keyword arguments (line 194)
            kwargs_263 = {}
            # Getting the type of 'str' (line 194)
            str_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'str', False)
            # Calling str(args, kwargs) (line 194)
            str_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 194, 26), str_260, *[name_262], **kwargs_263)
            
            # Processing the call keyword arguments (line 194)
            kwargs_265 = {}
            # Getting the type of 'answer' (line 194)
            answer_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'answer', False)
            # Obtaining the member 'append' of a type (line 194)
            append_259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), answer_258, 'append')
            # Calling append(args, kwargs) (line 194)
            append_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), append_259, *[str_call_result_264], **kwargs_265)
            
            
            # Assigning a Name to a Name (line 195):
            
            # Assigning a Name to a Name (line 195):
            # Getting the type of 'root' (line 195)
            root_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'root')
            # Assigning a type to the variable 'currentnode' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'currentnode', root_267)
            # SSA join for if statement (line 193)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Evaluating assert statement condition
    
    # Getting the type of 'currentnode' (line 198)
    currentnode_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'currentnode')
    # Getting the type of 'root' (line 198)
    root_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 31), 'root')
    # Applying the binary operator '==' (line 198)
    result_eq_270 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 16), '==', currentnode_268, root_269)
    
    assert_271 = result_eq_270
    # Assigning a type to the variable 'assert_271' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'assert_271', result_eq_270)
    # Getting the type of 'answer' (line 199)
    answer_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'answer')
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type', answer_272)
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 179)
    stypy_return_type_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_273

# Assigning a type to the variable 'decode' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'decode', decode)

@norecursion
def makenodes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'makenodes'
    module_type_store = module_type_store.open_function_context('makenodes', 210, 0, False)
    
    # Passed parameters checking function
    makenodes.stypy_localization = localization
    makenodes.stypy_type_of_self = None
    makenodes.stypy_type_store = module_type_store
    makenodes.stypy_function_name = 'makenodes'
    makenodes.stypy_param_names_list = ['probs']
    makenodes.stypy_varargs_param_name = None
    makenodes.stypy_kwargs_param_name = None
    makenodes.stypy_call_defaults = defaults
    makenodes.stypy_call_varargs = varargs
    makenodes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'makenodes', ['probs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'makenodes', localization, ['probs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'makenodes(...)' code ##################

    str_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, (-1)), 'str', '\n    Creates a list of nodes ready for the Huffman algorithm.\n    Each node will receive a codeword when Huffman algorithm "iterate" runs.\n\n    probs should be a list of pairs(\'<symbol>\', <value>).\n\n    >>> probs=[(\'a\',0.5), (\'b\',0.25), (\'c\',0.125), (\'d\',0.125)]\n    >>> symbols = makenodes(probs)\n    >>> root = iterate(symbols)\n    >>> zipped = encode([\'a\',\'a\',\'b\',\'a\',\'c\',\'b\',\'c\',\'d\'], symbols)\n    >>> print zipped\n    1101100001000001\n    >>> print decode( zipped, root )\n    [\'a\', \'a\', \'b\', \'a\', \'c\', \'b\', \'c\', \'d\']\n\n    See also the file Example.py for a python program that uses this package.\n    ')
    
    # Assigning a Num to a Name (line 228):
    
    # Assigning a Num to a Name (line 228):
    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
    # Assigning a type to the variable 'm' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'm', int_275)
    
    # Assigning a List to a Name (line 229):
    
    # Assigning a List to a Name (line 229):
    
    # Obtaining an instance of the builtin type 'list' (line 229)
    list_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 229)
    
    # Assigning a type to the variable 'c' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'c', list_276)
    
    # Getting the type of 'probs' (line 230)
    probs_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 13), 'probs')
    # Assigning a type to the variable 'probs_277' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'probs_277', probs_277)
    # Testing if the for loop is going to be iterated (line 230)
    # Testing the type of a for loop iterable (line 230)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 230, 4), probs_277)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 230, 4), probs_277):
        # Getting the type of the for loop variable (line 230)
        for_loop_var_278 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 230, 4), probs_277)
        # Assigning a type to the variable 'p' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'p', for_loop_var_278)
        # SSA begins for a for statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'm' (line 231)
        m_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'm')
        int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 13), 'int')
        # Applying the binary operator '+=' (line 231)
        result_iadd_281 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 8), '+=', m_279, int_280)
        # Assigning a type to the variable 'm' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'm', result_iadd_281)
        
        
        # Call to append(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to node(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Obtaining the type of the subscript
        int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 24), 'int')
        # Getting the type of 'p' (line 232)
        p_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 22), p_286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 232, 22), getitem___287, int_285)
        
        # Getting the type of 'm' (line 232)
        m_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'm', False)
        
        # Obtaining the type of the subscript
        int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 33), 'int')
        # Getting the type of 'p' (line 232)
        p_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 31), p_291, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 232, 31), getitem___292, int_290)
        
        # Processing the call keyword arguments (line 232)
        kwargs_294 = {}
        # Getting the type of 'node' (line 232)
        node_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'node', False)
        # Calling node(args, kwargs) (line 232)
        node_call_result_295 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), node_284, *[subscript_call_result_288, m_289, subscript_call_result_293], **kwargs_294)
        
        # Processing the call keyword arguments (line 232)
        kwargs_296 = {}
        # Getting the type of 'c' (line 232)
        c_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'c', False)
        # Obtaining the member 'append' of a type (line 232)
        append_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), c_282, 'append')
        # Calling append(args, kwargs) (line 232)
        append_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), append_283, *[node_call_result_295], **kwargs_296)
        
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'c' (line 234)
    c_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type', c_298)
    
    # ################# End of 'makenodes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'makenodes' in the type store
    # Getting the type of 'stypy_return_type' (line 210)
    stypy_return_type_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'makenodes'
    return stypy_return_type_299

# Assigning a type to the variable 'makenodes' (line 210)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'makenodes', makenodes)

@norecursion
def dec_to_bin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dec_to_bin'
    module_type_store = module_type_store.open_function_context('dec_to_bin', 237, 0, False)
    
    # Passed parameters checking function
    dec_to_bin.stypy_localization = localization
    dec_to_bin.stypy_type_of_self = None
    dec_to_bin.stypy_type_store = module_type_store
    dec_to_bin.stypy_function_name = 'dec_to_bin'
    dec_to_bin.stypy_param_names_list = ['n', 'digits']
    dec_to_bin.stypy_varargs_param_name = None
    dec_to_bin.stypy_kwargs_param_name = None
    dec_to_bin.stypy_call_defaults = defaults
    dec_to_bin.stypy_call_varargs = varargs
    dec_to_bin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dec_to_bin', ['n', 'digits'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dec_to_bin', localization, ['n', 'digits'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dec_to_bin(...)' code ##################

    str_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, (-1)), 'str', ' n is the number to convert to binary;  digits is the number of bits you want\n    Always prints full number of digits\n    >>> print dec_to_bin( 17 , 9)\n    000010001\n    >>> print dec_to_bin( 17 , 5)\n    10001\n\n    Will behead the standard binary number if requested\n    >>> print dec_to_bin( 17 , 4)\n    0001\n    ')
    
    # Getting the type of 'n' (line 249)
    n_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'n')
    int_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'int')
    # Applying the binary operator '<' (line 249)
    result_lt_303 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 8), '<', n_301, int_302)
    
    # Testing if the type of an if condition is none (line 249)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 4), result_lt_303):
        pass
    else:
        
        # Testing the type of an if condition (line 249)
        if_condition_304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 4), result_lt_303)
        # Assigning a type to the variable 'if_condition_304' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'if_condition_304', if_condition_304)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 250)
        # Processing the call arguments (line 250)
        str_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 25), 'str', 'warning, negative n not expected\n')
        # Processing the call keyword arguments (line 250)
        kwargs_309 = {}
        # Getting the type of 'sys' (line 250)
        sys_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 250)
        stderr_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), sys_305, 'stderr')
        # Obtaining the member 'write' of a type (line 250)
        write_307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), stderr_306, 'write')
        # Calling write(args, kwargs) (line 250)
        write_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), write_307, *[str_308], **kwargs_309)
        
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 251):
    
    # Assigning a BinOp to a Name (line 251):
    # Getting the type of 'digits' (line 251)
    digits_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'digits')
    int_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'int')
    # Applying the binary operator '-' (line 251)
    result_sub_313 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 8), '-', digits_311, int_312)
    
    # Assigning a type to the variable 'i' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'i', result_sub_313)
    
    # Assigning a Str to a Name (line 252):
    
    # Assigning a Str to a Name (line 252):
    str_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 10), 'str', '')
    # Assigning a type to the variable 'ans' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'ans', str_314)
    
    
    # Getting the type of 'i' (line 253)
    i_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 10), 'i')
    int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 15), 'int')
    # Applying the binary operator '>=' (line 253)
    result_ge_317 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 10), '>=', i_315, int_316)
    
    # Assigning a type to the variable 'result_ge_317' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'result_ge_317', result_ge_317)
    # Testing if the while is going to be iterated (line 253)
    # Testing the type of an if condition (line 253)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 4), result_ge_317)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 253, 4), result_ge_317):
        # SSA begins for while statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Compare to a Name (line 254):
        
        # Assigning a Compare to a Name (line 254):
        
        int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 15), 'int')
        # Getting the type of 'i' (line 254)
        i_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'i')
        # Applying the binary operator '<<' (line 254)
        result_lshift_320 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 15), '<<', int_318, i_319)
        
        # Getting the type of 'n' (line 254)
        n_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'n')
        # Applying the binary operator '&' (line 254)
        result_and__322 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 14), '&', result_lshift_320, n_321)
        
        int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 30), 'int')
        # Applying the binary operator '>' (line 254)
        result_gt_324 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 13), '>', result_and__322, int_323)
        
        # Assigning a type to the variable 'b' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'b', result_gt_324)
        
        # Getting the type of 'i' (line 255)
        i_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'i')
        int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 13), 'int')
        # Applying the binary operator '-=' (line 255)
        result_isub_327 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 8), '-=', i_325, int_326)
        # Assigning a type to the variable 'i' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'i', result_isub_327)
        
        
        # Assigning a BinOp to a Name (line 256):
        
        # Assigning a BinOp to a Name (line 256):
        # Getting the type of 'ans' (line 256)
        ans_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'ans')
        
        # Call to str(...): (line 256)
        # Processing the call arguments (line 256)
        
        # Call to int(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'b' (line 256)
        b_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'b', False)
        # Processing the call keyword arguments (line 256)
        kwargs_332 = {}
        # Getting the type of 'int' (line 256)
        int_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 24), 'int', False)
        # Calling int(args, kwargs) (line 256)
        int_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 256, 24), int_330, *[b_331], **kwargs_332)
        
        # Processing the call keyword arguments (line 256)
        kwargs_334 = {}
        # Getting the type of 'str' (line 256)
        str_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'str', False)
        # Calling str(args, kwargs) (line 256)
        str_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 256, 20), str_329, *[int_call_result_333], **kwargs_334)
        
        # Applying the binary operator '+' (line 256)
        result_add_336 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 14), '+', ans_328, str_call_result_335)
        
        # Assigning a type to the variable 'ans' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'ans', result_add_336)
        # SSA join for while statement (line 253)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'ans' (line 257)
    ans_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'ans')
    # Assigning a type to the variable 'stypy_return_type' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type', ans_337)
    
    # ################# End of 'dec_to_bin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dec_to_bin' in the type store
    # Getting the type of 'stypy_return_type' (line 237)
    stypy_return_type_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_338)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dec_to_bin'
    return stypy_return_type_338

# Assigning a type to the variable 'dec_to_bin' (line 237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'dec_to_bin', dec_to_bin)

# Assigning a Num to a Name (line 260):

# Assigning a Num to a Name (line 260):
int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 10), 'int')
# Assigning a type to the variable 'verbose' (line 260)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 0), 'verbose', int_339)

@norecursion
def weight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'weight'
    module_type_store = module_type_store.open_function_context('weight', 263, 0, False)
    
    # Passed parameters checking function
    weight.stypy_localization = localization
    weight.stypy_type_of_self = None
    weight.stypy_type_store = module_type_store
    weight.stypy_function_name = 'weight'
    weight.stypy_param_names_list = ['string']
    weight.stypy_varargs_param_name = None
    weight.stypy_kwargs_param_name = None
    weight.stypy_call_defaults = defaults
    weight.stypy_call_varargs = varargs
    weight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'weight', ['string'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'weight', localization, ['string'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'weight(...)' code ##################

    str_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, (-1)), 'str', '\n    ## returns number of 0s and number of 1s in the string\n    >>> print weight("00011")\n    (3, 2)\n    ')
    
    # Assigning a Num to a Name (line 269):
    
    # Assigning a Num to a Name (line 269):
    int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 9), 'int')
    # Assigning a type to the variable 'w0' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'w0', int_341)
    
    # Assigning a Num to a Name (line 270):
    
    # Assigning a Num to a Name (line 270):
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 9), 'int')
    # Assigning a type to the variable 'w1' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'w1', int_342)
    
    
    # Call to list(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'string' (line 271)
    string_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'string', False)
    # Processing the call keyword arguments (line 271)
    kwargs_345 = {}
    # Getting the type of 'list' (line 271)
    list_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'list', False)
    # Calling list(args, kwargs) (line 271)
    list_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 271, 13), list_343, *[string_344], **kwargs_345)
    
    # Assigning a type to the variable 'list_call_result_346' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'list_call_result_346', list_call_result_346)
    # Testing if the for loop is going to be iterated (line 271)
    # Testing the type of a for loop iterable (line 271)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 271, 4), list_call_result_346)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 271, 4), list_call_result_346):
        # Getting the type of the for loop variable (line 271)
        for_loop_var_347 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 271, 4), list_call_result_346)
        # Assigning a type to the variable 'c' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'c', for_loop_var_347)
        # SSA begins for a for statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'c' (line 272)
        c_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'c')
        str_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 17), 'str', '0')
        # Applying the binary operator '==' (line 272)
        result_eq_350 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 12), '==', c_348, str_349)
        
        # Testing if the type of an if condition is none (line 272)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 272, 8), result_eq_350):
            
            # Getting the type of 'c' (line 275)
            c_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 14), 'c')
            str_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 19), 'str', '1')
            # Applying the binary operator '==' (line 275)
            result_eq_357 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 14), '==', c_355, str_356)
            
            # Testing if the type of an if condition is none (line 275)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 275, 13), result_eq_357):
                pass
            else:
                
                # Testing the type of an if condition (line 275)
                if_condition_358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 13), result_eq_357)
                # Assigning a type to the variable 'if_condition_358' (line 275)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'if_condition_358', if_condition_358)
                # SSA begins for if statement (line 275)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'w1' (line 276)
                w1_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'w1')
                int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 18), 'int')
                # Applying the binary operator '+=' (line 276)
                result_iadd_361 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 12), '+=', w1_359, int_360)
                # Assigning a type to the variable 'w1' (line 276)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'w1', result_iadd_361)
                
                pass
                # SSA join for if statement (line 275)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 272)
            if_condition_351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 8), result_eq_350)
            # Assigning a type to the variable 'if_condition_351' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'if_condition_351', if_condition_351)
            # SSA begins for if statement (line 272)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'w0' (line 273)
            w0_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'w0')
            int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 18), 'int')
            # Applying the binary operator '+=' (line 273)
            result_iadd_354 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 12), '+=', w0_352, int_353)
            # Assigning a type to the variable 'w0' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'w0', result_iadd_354)
            
            pass
            # SSA branch for the else part of an if statement (line 272)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'c' (line 275)
            c_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 14), 'c')
            str_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 19), 'str', '1')
            # Applying the binary operator '==' (line 275)
            result_eq_357 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 14), '==', c_355, str_356)
            
            # Testing if the type of an if condition is none (line 275)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 275, 13), result_eq_357):
                pass
            else:
                
                # Testing the type of an if condition (line 275)
                if_condition_358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 13), result_eq_357)
                # Assigning a type to the variable 'if_condition_358' (line 275)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'if_condition_358', if_condition_358)
                # SSA begins for if statement (line 275)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'w1' (line 276)
                w1_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'w1')
                int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 18), 'int')
                # Applying the binary operator '+=' (line 276)
                result_iadd_361 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 12), '+=', w1_359, int_360)
                # Assigning a type to the variable 'w1' (line 276)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'w1', result_iadd_361)
                
                pass
                # SSA join for if statement (line 275)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 272)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 279)
    tuple_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 279)
    # Adding element type (line 279)
    # Getting the type of 'w0' (line 279)
    w0_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'w0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 12), tuple_362, w0_363)
    # Adding element type (line 279)
    # Getting the type of 'w1' (line 279)
    w1_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'w1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 12), tuple_362, w1_364)
    
    # Assigning a type to the variable 'stypy_return_type' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type', tuple_362)
    
    # ################# End of 'weight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'weight' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_365)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'weight'
    return stypy_return_type_365

# Assigning a type to the variable 'weight' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'weight', weight)

@norecursion
def findprobs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 16), 'float')
    int_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 24), 'int')
    defaults = [float_366, int_367]
    # Create a new context for function 'findprobs'
    module_type_store = module_type_store.open_function_context('findprobs', 282, 0, False)
    
    # Passed parameters checking function
    findprobs.stypy_localization = localization
    findprobs.stypy_type_of_self = None
    findprobs.stypy_type_store = module_type_store
    findprobs.stypy_function_name = 'findprobs'
    findprobs.stypy_param_names_list = ['f', 'N']
    findprobs.stypy_varargs_param_name = None
    findprobs.stypy_kwargs_param_name = None
    findprobs.stypy_call_defaults = defaults
    findprobs.stypy_call_varargs = varargs
    findprobs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findprobs', ['f', 'N'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findprobs', localization, ['f', 'N'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findprobs(...)' code ##################

    str_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, (-1)), 'str', " Find probabilities of all the events\n    000000\n    000001\n     ...\n    111111\n    <-N ->\n    >>> print findprobs(0.1,3)              # doctest:+ELLIPSIS\n    [('000', 0.7...),..., ('111', 0.001...)]\n    ")
    
    # Assigning a List to a Name (line 292):
    
    # Assigning a List to a Name (line 292):
    
    # Obtaining an instance of the builtin type 'list' (line 292)
    list_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 292)
    
    # Assigning a type to the variable 'answer' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'answer', list_369)
    
    
    # Call to range(...): (line 293)
    # Processing the call arguments (line 293)
    int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'int')
    # Getting the type of 'N' (line 293)
    N_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'N', False)
    # Applying the binary operator '**' (line 293)
    result_pow_373 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 19), '**', int_371, N_372)
    
    # Processing the call keyword arguments (line 293)
    kwargs_374 = {}
    # Getting the type of 'range' (line 293)
    range_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'range', False)
    # Calling range(args, kwargs) (line 293)
    range_call_result_375 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), range_370, *[result_pow_373], **kwargs_374)
    
    # Assigning a type to the variable 'range_call_result_375' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'range_call_result_375', range_call_result_375)
    # Testing if the for loop is going to be iterated (line 293)
    # Testing the type of a for loop iterable (line 293)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 293, 4), range_call_result_375)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 293, 4), range_call_result_375):
        # Getting the type of the for loop variable (line 293)
        for_loop_var_376 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 293, 4), range_call_result_375)
        # Assigning a type to the variable 'n' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'n', for_loop_var_376)
        # SSA begins for a for statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to dec_to_bin(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'n' (line 294)
        n_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'n', False)
        # Getting the type of 'N' (line 294)
        N_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'N', False)
        # Processing the call keyword arguments (line 294)
        kwargs_380 = {}
        # Getting the type of 'dec_to_bin' (line 294)
        dec_to_bin_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'dec_to_bin', False)
        # Calling dec_to_bin(args, kwargs) (line 294)
        dec_to_bin_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), dec_to_bin_377, *[n_378, N_379], **kwargs_380)
        
        # Assigning a type to the variable 's' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 's', dec_to_bin_call_result_381)
        
        # Assigning a Call to a Tuple (line 295):
        
        # Assigning a Call to a Name:
        
        # Call to weight(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 's' (line 295)
        s_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 's', False)
        # Processing the call keyword arguments (line 295)
        kwargs_384 = {}
        # Getting the type of 'weight' (line 295)
        weight_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'weight', False)
        # Calling weight(args, kwargs) (line 295)
        weight_call_result_385 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), weight_382, *[s_383], **kwargs_384)
        
        # Assigning a type to the variable 'call_assignment_1' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_1', weight_call_result_385)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 295)
        call_assignment_1_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_387 = stypy_get_value_from_tuple(call_assignment_1_386, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_2' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_2', stypy_get_value_from_tuple_call_result_387)
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'call_assignment_2' (line 295)
        call_assignment_2_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_2')
        # Assigning a type to the variable 'w0' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 9), 'w0', call_assignment_2_388)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 295)
        call_assignment_1_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_390 = stypy_get_value_from_tuple(call_assignment_1_389, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_3', stypy_get_value_from_tuple_call_result_390)
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'call_assignment_3' (line 295)
        call_assignment_3_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_3')
        # Assigning a type to the variable 'w1' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 13), 'w1', call_assignment_3_391)
        
        # Evaluating a boolean operation
        # Getting the type of 'verbose' (line 296)
        verbose_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'verbose')
        int_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 23), 'int')
        # Applying the binary operator 'and' (line 296)
        result_and_keyword_394 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 11), 'and', verbose_392, int_393)
        
        # Testing if the type of an if condition is none (line 296)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 296, 8), result_and_keyword_394):
            pass
        else:
            
            # Testing the type of an if condition (line 296)
            if_condition_395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 8), result_and_keyword_394)
            # Assigning a type to the variable 'if_condition_395' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'if_condition_395', if_condition_395)
            # SSA begins for if statement (line 296)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 296)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 298)
        # Processing the call arguments (line 298)
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        # Getting the type of 's' (line 298)
        s_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 23), tuple_398, s_399)
        # Adding element type (line 298)
        # Getting the type of 'f' (line 298)
        f_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 26), 'f', False)
        # Getting the type of 'w1' (line 298)
        w1_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 31), 'w1', False)
        # Applying the binary operator '**' (line 298)
        result_pow_402 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 26), '**', f_400, w1_401)
        
        int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 37), 'int')
        # Getting the type of 'f' (line 298)
        f_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'f', False)
        # Applying the binary operator '-' (line 298)
        result_sub_405 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 37), '-', int_403, f_404)
        
        # Getting the type of 'w0' (line 298)
        w0_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 47), 'w0', False)
        # Applying the binary operator '**' (line 298)
        result_pow_407 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 36), '**', result_sub_405, w0_406)
        
        # Applying the binary operator '*' (line 298)
        result_mul_408 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 26), '*', result_pow_402, result_pow_407)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 23), tuple_398, result_mul_408)
        
        # Processing the call keyword arguments (line 298)
        kwargs_409 = {}
        # Getting the type of 'answer' (line 298)
        answer_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'answer', False)
        # Obtaining the member 'append' of a type (line 298)
        append_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), answer_396, 'append')
        # Calling append(args, kwargs) (line 298)
        append_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), append_397, *[tuple_398], **kwargs_409)
        
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Evaluating assert statement condition
    
    
    # Call to len(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'answer' (line 300)
    answer_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'answer', False)
    # Processing the call keyword arguments (line 300)
    kwargs_413 = {}
    # Getting the type of 'len' (line 300)
    len_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'len', False)
    # Calling len(args, kwargs) (line 300)
    len_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), len_411, *[answer_412], **kwargs_413)
    
    int_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 27), 'int')
    # Getting the type of 'N' (line 300)
    N_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 32), 'N')
    # Applying the binary operator '**' (line 300)
    result_pow_417 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 27), '**', int_415, N_416)
    
    # Applying the binary operator '==' (line 300)
    result_eq_418 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 12), '==', len_call_result_414, result_pow_417)
    
    assert_419 = result_eq_418
    # Assigning a type to the variable 'assert_419' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'assert_419', result_eq_418)
    # Getting the type of 'answer' (line 301)
    answer_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'answer')
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type', answer_420)
    
    # ################# End of 'findprobs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findprobs' in the type store
    # Getting the type of 'stypy_return_type' (line 282)
    stypy_return_type_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findprobs'
    return stypy_return_type_421

# Assigning a type to the variable 'findprobs' (line 282)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'findprobs', findprobs)

@norecursion
def Bencode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Bencode'
    module_type_store = module_type_store.open_function_context('Bencode', 304, 0, False)
    
    # Passed parameters checking function
    Bencode.stypy_localization = localization
    Bencode.stypy_type_of_self = None
    Bencode.stypy_type_store = module_type_store
    Bencode.stypy_function_name = 'Bencode'
    Bencode.stypy_param_names_list = ['string', 'symbols', 'N']
    Bencode.stypy_varargs_param_name = None
    Bencode.stypy_kwargs_param_name = None
    Bencode.stypy_call_defaults = defaults
    Bencode.stypy_call_varargs = varargs
    Bencode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Bencode', ['string', 'symbols', 'N'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Bencode', localization, ['string', 'symbols', 'N'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Bencode(...)' code ##################

    str_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, (-1)), 'str', '\n    Reads in a string of 0s and 1s.\n    Creates a list of blocks of size N.\n    Sends this list to the general-purpose Huffman encoder\n    defined by the nodes in the list "symbols".\n    ')
    
    # Assigning a List to a Name (line 311):
    
    # Assigning a List to a Name (line 311):
    
    # Obtaining an instance of the builtin type 'list' (line 311)
    list_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 311)
    
    # Assigning a type to the variable 'blocks' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'blocks', list_423)
    
    # Assigning a Call to a Name (line 312):
    
    # Assigning a Call to a Name (line 312):
    
    # Call to list(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'string' (line 312)
    string_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'string', False)
    # Processing the call keyword arguments (line 312)
    kwargs_426 = {}
    # Getting the type of 'list' (line 312)
    list_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'list', False)
    # Calling list(args, kwargs) (line 312)
    list_call_result_427 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), list_424, *[string_425], **kwargs_426)
    
    # Assigning a type to the variable 'chars' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'chars', list_call_result_427)
    
    # Assigning a Str to a Name (line 314):
    
    # Assigning a Str to a Name (line 314):
    str_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 8), 'str', '')
    # Assigning a type to the variable 's' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 's', str_428)
    
    # Getting the type of 'chars' (line 315)
    chars_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'chars')
    # Assigning a type to the variable 'chars_429' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'chars_429', chars_429)
    # Testing if the for loop is going to be iterated (line 315)
    # Testing the type of a for loop iterable (line 315)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 315, 4), chars_429)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 315, 4), chars_429):
        # Getting the type of the for loop variable (line 315)
        for_loop_var_430 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 315, 4), chars_429)
        # Assigning a type to the variable 'c' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'c', for_loop_var_430)
        # SSA begins for a for statement (line 315)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 316):
        
        # Assigning a BinOp to a Name (line 316):
        # Getting the type of 's' (line 316)
        s_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 's')
        # Getting the type of 'c' (line 316)
        c_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'c')
        # Applying the binary operator '+' (line 316)
        result_add_433 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 12), '+', s_431, c_432)
        
        # Assigning a type to the variable 's' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 's', result_add_433)
        
        
        # Call to len(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 's' (line 317)
        s_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 's', False)
        # Processing the call keyword arguments (line 317)
        kwargs_436 = {}
        # Getting the type of 'len' (line 317)
        len_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'len', False)
        # Calling len(args, kwargs) (line 317)
        len_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), len_434, *[s_435], **kwargs_436)
        
        # Getting the type of 'N' (line 317)
        N_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 22), 'N')
        # Applying the binary operator '>=' (line 317)
        result_ge_439 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 12), '>=', len_call_result_437, N_438)
        
        # Testing if the type of an if condition is none (line 317)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 317, 8), result_ge_439):
            pass
        else:
            
            # Testing the type of an if condition (line 317)
            if_condition_440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), result_ge_439)
            # Assigning a type to the variable 'if_condition_440' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_440', if_condition_440)
            # SSA begins for if statement (line 317)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 's' (line 318)
            s_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 's', False)
            # Processing the call keyword arguments (line 318)
            kwargs_444 = {}
            # Getting the type of 'blocks' (line 318)
            blocks_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'blocks', False)
            # Obtaining the member 'append' of a type (line 318)
            append_442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), blocks_441, 'append')
            # Calling append(args, kwargs) (line 318)
            append_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), append_442, *[s_443], **kwargs_444)
            
            
            # Assigning a Str to a Name (line 319):
            
            # Assigning a Str to a Name (line 319):
            str_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 16), 'str', '')
            # Assigning a type to the variable 's' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 's', str_446)
            pass
            # SSA join for if statement (line 317)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to len(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 's' (line 322)
    s_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 's', False)
    # Processing the call keyword arguments (line 322)
    kwargs_449 = {}
    # Getting the type of 'len' (line 322)
    len_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'len', False)
    # Calling len(args, kwargs) (line 322)
    len_call_result_450 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), len_447, *[s_448], **kwargs_449)
    
    int_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 17), 'int')
    # Applying the binary operator '>' (line 322)
    result_gt_452 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 8), '>', len_call_result_450, int_451)
    
    # Testing if the type of an if condition is none (line 322)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 322, 4), result_gt_452):
        pass
    else:
        
        # Testing the type of an if condition (line 322)
        if_condition_453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), result_gt_452)
        # Assigning a type to the variable 'if_condition_453' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_453', if_condition_453)
        # SSA begins for if statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 29), 'str', 'warning, padding last block with 0s')
        
        
        
        # Call to len(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 's' (line 324)
        s_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 's', False)
        # Processing the call keyword arguments (line 324)
        kwargs_457 = {}
        # Getting the type of 'len' (line 324)
        len_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'len', False)
        # Calling len(args, kwargs) (line 324)
        len_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 324, 15), len_455, *[s_456], **kwargs_457)
        
        # Getting the type of 'N' (line 324)
        N_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'N')
        # Applying the binary operator '<' (line 324)
        result_lt_460 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 15), '<', len_call_result_458, N_459)
        
        # Assigning a type to the variable 'result_lt_460' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'result_lt_460', result_lt_460)
        # Testing if the while is going to be iterated (line 324)
        # Testing the type of an if condition (line 324)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 8), result_lt_460)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 324, 8), result_lt_460):
            # SSA begins for while statement (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a BinOp to a Name (line 325):
            
            # Assigning a BinOp to a Name (line 325):
            # Getting the type of 's' (line 325)
            s_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 's')
            str_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 20), 'str', '0')
            # Applying the binary operator '+' (line 325)
            result_add_463 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 16), '+', s_461, str_462)
            
            # Assigning a type to the variable 's' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 's', result_add_463)
            pass
            # SSA join for while statement (line 324)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 's' (line 327)
        s_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 22), 's', False)
        # Processing the call keyword arguments (line 327)
        kwargs_467 = {}
        # Getting the type of 'blocks' (line 327)
        blocks_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'blocks', False)
        # Obtaining the member 'append' of a type (line 327)
        append_465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), blocks_464, 'append')
        # Calling append(args, kwargs) (line 327)
        append_call_result_468 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), append_465, *[s_466], **kwargs_467)
        
        pass
        # SSA join for if statement (line 322)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'verbose' (line 330)
    verbose_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 7), 'verbose')
    # Testing if the type of an if condition is none (line 330)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 330, 4), verbose_469):
        pass
    else:
        
        # Testing the type of an if condition (line 330)
        if_condition_470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 4), verbose_469)
        # Assigning a type to the variable 'if_condition_470' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'if_condition_470', if_condition_470)
        # SSA begins for if statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a Call to a Name (line 334):
    
    # Call to encode(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'blocks' (line 334)
    blocks_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'blocks', False)
    # Getting the type of 'symbols' (line 334)
    symbols_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'symbols', False)
    # Processing the call keyword arguments (line 334)
    kwargs_474 = {}
    # Getting the type of 'encode' (line 334)
    encode_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'encode', False)
    # Calling encode(args, kwargs) (line 334)
    encode_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 334, 13), encode_471, *[blocks_472, symbols_473], **kwargs_474)
    
    # Assigning a type to the variable 'zipped' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'zipped', encode_call_result_475)
    # Getting the type of 'zipped' (line 335)
    zipped_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'zipped')
    # Assigning a type to the variable 'stypy_return_type' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type', zipped_476)
    
    # ################# End of 'Bencode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Bencode' in the type store
    # Getting the type of 'stypy_return_type' (line 304)
    stypy_return_type_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_477)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Bencode'
    return stypy_return_type_477

# Assigning a type to the variable 'Bencode' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'Bencode', Bencode)

@norecursion
def Bdecode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Bdecode'
    module_type_store = module_type_store.open_function_context('Bdecode', 338, 0, False)
    
    # Passed parameters checking function
    Bdecode.stypy_localization = localization
    Bdecode.stypy_type_of_self = None
    Bdecode.stypy_type_store = module_type_store
    Bdecode.stypy_function_name = 'Bdecode'
    Bdecode.stypy_param_names_list = ['string', 'root', 'N']
    Bdecode.stypy_varargs_param_name = None
    Bdecode.stypy_kwargs_param_name = None
    Bdecode.stypy_call_defaults = defaults
    Bdecode.stypy_call_varargs = varargs
    Bdecode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Bdecode', ['string', 'root', 'N'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Bdecode', localization, ['string', 'root', 'N'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Bdecode(...)' code ##################

    str_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', '\n    Decode a binary string into blocks.\n    ')
    
    # Assigning a Call to a Name (line 342):
    
    # Assigning a Call to a Name (line 342):
    
    # Call to decode(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'string' (line 342)
    string_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'string', False)
    # Getting the type of 'root' (line 342)
    root_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'root', False)
    # Processing the call keyword arguments (line 342)
    kwargs_482 = {}
    # Getting the type of 'decode' (line 342)
    decode_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'decode', False)
    # Calling decode(args, kwargs) (line 342)
    decode_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 342, 13), decode_479, *[string_480, root_481], **kwargs_482)
    
    # Assigning a type to the variable 'answer' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'answer', decode_call_result_483)
    # Getting the type of 'verbose' (line 343)
    verbose_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 7), 'verbose')
    # Testing if the type of an if condition is none (line 343)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 343, 4), verbose_484):
        pass
    else:
        
        # Testing the type of an if condition (line 343)
        if_condition_485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 4), verbose_484)
        # Assigning a type to the variable 'if_condition_485' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'if_condition_485', if_condition_485)
        # SSA begins for if statement (line 343)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 343)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to join(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'answer' (line 347)
    answer_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 21), 'answer', False)
    # Processing the call keyword arguments (line 347)
    kwargs_489 = {}
    str_486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 13), 'str', '')
    # Obtaining the member 'join' of a type (line 347)
    join_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 13), str_486, 'join')
    # Calling join(args, kwargs) (line 347)
    join_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 347, 13), join_487, *[answer_488], **kwargs_489)
    
    # Assigning a type to the variable 'output' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'output', join_call_result_490)
    # Getting the type of 'output' (line 349)
    output_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type', output_491)
    
    # ################# End of 'Bdecode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Bdecode' in the type store
    # Getting the type of 'stypy_return_type' (line 338)
    stypy_return_type_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Bdecode'
    return stypy_return_type_492

# Assigning a type to the variable 'Bdecode' (line 338)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 0), 'Bdecode', Bdecode)

@norecursion
def easytest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'easytest'
    module_type_store = module_type_store.open_function_context('easytest', 352, 0, False)
    
    # Passed parameters checking function
    easytest.stypy_localization = localization
    easytest.stypy_type_of_self = None
    easytest.stypy_type_store = module_type_store
    easytest.stypy_function_name = 'easytest'
    easytest.stypy_param_names_list = []
    easytest.stypy_varargs_param_name = None
    easytest.stypy_kwargs_param_name = None
    easytest.stypy_call_defaults = defaults
    easytest.stypy_call_varargs = varargs
    easytest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'easytest', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'easytest', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'easytest(...)' code ##################

    str_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, (-1)), 'str', "\n    Tests block code with N=3, f=0.01 on a tiny example.\n    >>> easytest()                 # doctest:+NORMALIZE_WHITESPACE\n    #Symbol     Count           Codeword\n    000         (0.97)          1\n    001         (0.0098)        001\n    010         (0.0098)        010\n    011         (9.9e-05)       00001\n    100         (0.0098)        011\n    101         (9.9e-05)       00010\n    110         (9.9e-05)       00011\n    111         (1e-06)         00000\n    zipped  = 1001010000010110111\n    decoded = ['000', '001', '010', '011', '100', '100', '000']\n    OK!\n    ")
    
    # Assigning a Num to a Name (line 369):
    
    # Assigning a Num to a Name (line 369):
    int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'int')
    # Assigning a type to the variable 'N' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'N', int_494)
    
    # Assigning a Num to a Name (line 370):
    
    # Assigning a Num to a Name (line 370):
    float_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 8), 'float')
    # Assigning a type to the variable 'f' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'f', float_495)
    
    # Assigning a Call to a Name (line 371):
    
    # Assigning a Call to a Name (line 371):
    
    # Call to findprobs(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'f' (line 371)
    f_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 22), 'f', False)
    # Getting the type of 'N' (line 371)
    N_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 25), 'N', False)
    # Processing the call keyword arguments (line 371)
    kwargs_499 = {}
    # Getting the type of 'findprobs' (line 371)
    findprobs_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'findprobs', False)
    # Calling findprobs(args, kwargs) (line 371)
    findprobs_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 371, 12), findprobs_496, *[f_497, N_498], **kwargs_499)
    
    # Assigning a type to the variable 'probs' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'probs', findprobs_call_result_500)
    
    # Assigning a Call to a Name (line 374):
    
    # Assigning a Call to a Name (line 374):
    
    # Call to makenodes(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'probs' (line 374)
    probs_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 24), 'probs', False)
    # Processing the call keyword arguments (line 374)
    kwargs_503 = {}
    # Getting the type of 'makenodes' (line 374)
    makenodes_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 14), 'makenodes', False)
    # Calling makenodes(args, kwargs) (line 374)
    makenodes_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 374, 14), makenodes_501, *[probs_502], **kwargs_503)
    
    # Assigning a type to the variable 'symbols' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'symbols', makenodes_call_result_504)
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to iterate(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'symbols' (line 376)
    symbols_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'symbols', False)
    # Processing the call keyword arguments (line 375)
    kwargs_507 = {}
    # Getting the type of 'iterate' (line 375)
    iterate_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'iterate', False)
    # Calling iterate(args, kwargs) (line 375)
    iterate_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 375, 11), iterate_505, *[symbols_506], **kwargs_507)
    
    # Assigning a type to the variable 'root' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'root', iterate_call_result_508)
    
    # Call to sort(...): (line 378)
    # Processing the call arguments (line 378)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 378, 17, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['x', 'y']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['x', 'y'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to cmp(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'x' (line 378)
        x_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 34), 'x', False)
        # Obtaining the member 'index' of a type (line 378)
        index_513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 34), x_512, 'index')
        # Getting the type of 'y' (line 378)
        y_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 43), 'y', False)
        # Obtaining the member 'index' of a type (line 378)
        index_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 43), y_514, 'index')
        # Processing the call keyword arguments (line 378)
        kwargs_516 = {}
        # Getting the type of 'cmp' (line 378)
        cmp_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 30), 'cmp', False)
        # Calling cmp(args, kwargs) (line 378)
        cmp_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 378, 30), cmp_511, *[index_513, index_515], **kwargs_516)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'stypy_return_type', cmp_call_result_517)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_518

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 378)
    _stypy_temp_lambda_1_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), '_stypy_temp_lambda_1')
    # Processing the call keyword arguments (line 378)
    kwargs_520 = {}
    # Getting the type of 'symbols' (line 378)
    symbols_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'symbols', False)
    # Obtaining the member 'sort' of a type (line 378)
    sort_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 4), symbols_509, 'sort')
    # Calling sort(args, kwargs) (line 378)
    sort_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 378, 4), sort_510, *[_stypy_temp_lambda_1_519], **kwargs_520)
    
    
    # Getting the type of 'symbols' (line 379)
    symbols_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 14), 'symbols')
    # Assigning a type to the variable 'symbols_522' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'symbols_522', symbols_522)
    # Testing if the for loop is going to be iterated (line 379)
    # Testing the type of a for loop iterable (line 379)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 379, 4), symbols_522)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 379, 4), symbols_522):
        # Getting the type of the for loop variable (line 379)
        for_loop_var_523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 379, 4), symbols_522)
        # Assigning a type to the variable 'co' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'co', for_loop_var_523)
        # SSA begins for a for statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to report(...): (line 380)
        # Processing the call keyword arguments (line 380)
        kwargs_526 = {}
        # Getting the type of 'co' (line 380)
        co_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'co', False)
        # Obtaining the member 'report' of a type (line 380)
        report_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), co_524, 'report')
        # Calling report(args, kwargs) (line 380)
        report_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), report_525, *[], **kwargs_526)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a List to a Name (line 382):
    
    # Assigning a List to a Name (line 382):
    
    # Obtaining an instance of the builtin type 'list' (line 382)
    list_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 382)
    # Adding element type (line 382)
    str_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 14), 'str', '000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), list_528, str_529)
    # Adding element type (line 382)
    str_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 21), 'str', '001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), list_528, str_530)
    # Adding element type (line 382)
    str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 28), 'str', '010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), list_528, str_531)
    # Adding element type (line 382)
    str_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 35), 'str', '011')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), list_528, str_532)
    # Adding element type (line 382)
    str_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 42), 'str', '100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), list_528, str_533)
    # Adding element type (line 382)
    str_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 49), 'str', '100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), list_528, str_534)
    # Adding element type (line 382)
    str_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 56), 'str', '000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), list_528, str_535)
    
    # Assigning a type to the variable 'source' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'source', list_528)
    
    # Assigning a Call to a Name (line 383):
    
    # Assigning a Call to a Name (line 383):
    
    # Call to encode(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'source' (line 383)
    source_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 20), 'source', False)
    # Getting the type of 'symbols' (line 383)
    symbols_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 28), 'symbols', False)
    # Processing the call keyword arguments (line 383)
    kwargs_539 = {}
    # Getting the type of 'encode' (line 383)
    encode_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 13), 'encode', False)
    # Calling encode(args, kwargs) (line 383)
    encode_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 383, 13), encode_536, *[source_537, symbols_538], **kwargs_539)
    
    # Assigning a type to the variable 'zipped' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'zipped', encode_call_result_540)
    str_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 10), 'str', 'zipped  =')
    # Getting the type of 'zipped' (line 384)
    zipped_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 22), 'zipped')
    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to decode(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'zipped' (line 385)
    zipped_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'zipped', False)
    # Getting the type of 'root' (line 385)
    root_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'root', False)
    # Processing the call keyword arguments (line 385)
    kwargs_546 = {}
    # Getting the type of 'decode' (line 385)
    decode_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 13), 'decode', False)
    # Calling decode(args, kwargs) (line 385)
    decode_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 385, 13), decode_543, *[zipped_544, root_545], **kwargs_546)
    
    # Assigning a type to the variable 'answer' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'answer', decode_call_result_547)
    str_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 10), 'str', 'decoded =')
    # Getting the type of 'answer' (line 386)
    answer_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 22), 'answer')
    
    # Getting the type of 'source' (line 387)
    source_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'source')
    # Getting the type of 'answer' (line 387)
    answer_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 18), 'answer')
    # Applying the binary operator '!=' (line 387)
    result_ne_552 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 8), '!=', source_550, answer_551)
    
    # Testing if the type of an if condition is none (line 387)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 387, 4), result_ne_552):
        str_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 14), 'str', 'OK!')
    else:
        
        # Testing the type of an if condition (line 387)
        if_condition_553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 4), result_ne_552)
        # Assigning a type to the variable 'if_condition_553' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'if_condition_553', if_condition_553)
        # SSA begins for if statement (line 387)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 14), 'str', 'ERROR')
        # SSA branch for the else part of an if statement (line 387)
        module_type_store.open_ssa_branch('else')
        str_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 14), 'str', 'OK!')
        # SSA join for if statement (line 387)
        module_type_store = module_type_store.join_ssa_context()
        

    pass
    
    # ################# End of 'easytest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'easytest' in the type store
    # Getting the type of 'stypy_return_type' (line 352)
    stypy_return_type_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_556)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'easytest'
    return stypy_return_type_556

# Assigning a type to the variable 'easytest' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'easytest', easytest)

@norecursion
def test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test'
    module_type_store = module_type_store.open_function_context('test', 394, 0, False)
    
    # Passed parameters checking function
    test.stypy_localization = localization
    test.stypy_type_of_self = None
    test.stypy_type_store = module_type_store
    test.stypy_function_name = 'test'
    test.stypy_param_names_list = []
    test.stypy_varargs_param_name = None
    test.stypy_kwargs_param_name = None
    test.stypy_call_defaults = defaults
    test.stypy_call_varargs = varargs
    test.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test(...)' code ##################

    
    # Call to easytest(...): (line 395)
    # Processing the call keyword arguments (line 395)
    kwargs_558 = {}
    # Getting the type of 'easytest' (line 395)
    easytest_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'easytest', False)
    # Calling easytest(args, kwargs) (line 395)
    easytest_call_result_559 = invoke(stypy.reporting.localization.Localization(__file__, 395, 4), easytest_557, *[], **kwargs_558)
    
    
    # Call to hardertest(...): (line 396)
    # Processing the call keyword arguments (line 396)
    kwargs_561 = {}
    # Getting the type of 'hardertest' (line 396)
    hardertest_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'hardertest', False)
    # Calling hardertest(args, kwargs) (line 396)
    hardertest_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 396, 4), hardertest_560, *[], **kwargs_561)
    
    
    # ################# End of 'test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test' in the type store
    # Getting the type of 'stypy_return_type' (line 394)
    stypy_return_type_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test'
    return stypy_return_type_563

# Assigning a type to the variable 'test' (line 394)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 0), 'test', test)

@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 399, 0, False)
    
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

    
    # Call to join(...): (line 400)
    # Processing the call arguments (line 400)
    
    # Call to dirname(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of '__file__' (line 400)
    file___570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 40), '__file__', False)
    # Processing the call keyword arguments (line 400)
    kwargs_571 = {}
    # Getting the type of 'os' (line 400)
    os_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 400)
    path_568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 24), os_567, 'path')
    # Obtaining the member 'dirname' of a type (line 400)
    dirname_569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 24), path_568, 'dirname')
    # Calling dirname(args, kwargs) (line 400)
    dirname_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 400, 24), dirname_569, *[file___570], **kwargs_571)
    
    # Getting the type of 'path' (line 400)
    path_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 51), 'path', False)
    # Processing the call keyword arguments (line 400)
    kwargs_574 = {}
    # Getting the type of 'os' (line 400)
    os_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 400)
    path_565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 11), os_564, 'path')
    # Obtaining the member 'join' of a type (line 400)
    join_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 11), path_565, 'join')
    # Calling join(args, kwargs) (line 400)
    join_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 400, 11), join_566, *[dirname_call_result_572, path_573], **kwargs_574)
    
    # Assigning a type to the variable 'stypy_return_type' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type', join_call_result_575)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 399)
    stypy_return_type_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_576)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_576

# Assigning a type to the variable 'Relative' (line 399)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'Relative', Relative)

@norecursion
def hardertest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hardertest'
    module_type_store = module_type_store.open_function_context('hardertest', 403, 0, False)
    
    # Passed parameters checking function
    hardertest.stypy_localization = localization
    hardertest.stypy_type_of_self = None
    hardertest.stypy_type_store = module_type_store
    hardertest.stypy_function_name = 'hardertest'
    hardertest.stypy_param_names_list = []
    hardertest.stypy_varargs_param_name = None
    hardertest.stypy_kwargs_param_name = None
    hardertest.stypy_call_defaults = defaults
    hardertest.stypy_call_varargs = varargs
    hardertest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hardertest', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hardertest', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hardertest(...)' code ##################

    
    # Assigning a Call to a Name (line 405):
    
    # Assigning a Call to a Name (line 405):
    
    # Call to open(...): (line 405)
    # Processing the call arguments (line 405)
    
    # Call to Relative(...): (line 405)
    # Processing the call arguments (line 405)
    str_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 30), 'str', 'testdata/BentCoinFile')
    # Processing the call keyword arguments (line 405)
    kwargs_580 = {}
    # Getting the type of 'Relative' (line 405)
    Relative_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'Relative', False)
    # Calling Relative(args, kwargs) (line 405)
    Relative_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 405, 21), Relative_578, *[str_579], **kwargs_580)
    
    str_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 56), 'str', 'r')
    # Processing the call keyword arguments (line 405)
    kwargs_583 = {}
    # Getting the type of 'open' (line 405)
    open_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'open', False)
    # Calling open(args, kwargs) (line 405)
    open_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 405, 16), open_577, *[Relative_call_result_581, str_582], **kwargs_583)
    
    # Assigning a type to the variable 'inputfile' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'inputfile', open_call_result_584)
    
    # Assigning a Call to a Name (line 406):
    
    # Assigning a Call to a Name (line 406):
    
    # Call to open(...): (line 406)
    # Processing the call arguments (line 406)
    
    # Call to Relative(...): (line 406)
    # Processing the call arguments (line 406)
    str_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 31), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 406)
    kwargs_588 = {}
    # Getting the type of 'Relative' (line 406)
    Relative_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 406)
    Relative_call_result_589 = invoke(stypy.reporting.localization.Localization(__file__, 406, 22), Relative_586, *[str_587], **kwargs_588)
    
    str_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 43), 'str', 'w')
    # Processing the call keyword arguments (line 406)
    kwargs_591 = {}
    # Getting the type of 'open' (line 406)
    open_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 17), 'open', False)
    # Calling open(args, kwargs) (line 406)
    open_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 406, 17), open_585, *[Relative_call_result_589, str_590], **kwargs_591)
    
    # Assigning a type to the variable 'outputfile' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'outputfile', open_call_result_592)
    
    # Call to compress_it(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'inputfile' (line 408)
    inputfile_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'inputfile', False)
    # Getting the type of 'outputfile' (line 408)
    outputfile_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 27), 'outputfile', False)
    # Processing the call keyword arguments (line 408)
    kwargs_596 = {}
    # Getting the type of 'compress_it' (line 408)
    compress_it_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'compress_it', False)
    # Calling compress_it(args, kwargs) (line 408)
    compress_it_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 408, 4), compress_it_593, *[inputfile_594, outputfile_595], **kwargs_596)
    
    
    # Call to close(...): (line 409)
    # Processing the call keyword arguments (line 409)
    kwargs_600 = {}
    # Getting the type of 'outputfile' (line 409)
    outputfile_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 409)
    close_599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 4), outputfile_598, 'close')
    # Calling close(args, kwargs) (line 409)
    close_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 409, 4), close_599, *[], **kwargs_600)
    
    
    # Call to close(...): (line 410)
    # Processing the call keyword arguments (line 410)
    kwargs_604 = {}
    # Getting the type of 'inputfile' (line 410)
    inputfile_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 410)
    close_603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 4), inputfile_602, 'close')
    # Calling close(args, kwargs) (line 410)
    close_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 410, 4), close_603, *[], **kwargs_604)
    
    
    # Assigning a Call to a Name (line 413):
    
    # Assigning a Call to a Name (line 413):
    
    # Call to open(...): (line 413)
    # Processing the call arguments (line 413)
    
    # Call to Relative(...): (line 413)
    # Processing the call arguments (line 413)
    str_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 30), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 413)
    kwargs_609 = {}
    # Getting the type of 'Relative' (line 413)
    Relative_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 21), 'Relative', False)
    # Calling Relative(args, kwargs) (line 413)
    Relative_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 413, 21), Relative_607, *[str_608], **kwargs_609)
    
    str_611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 42), 'str', 'r')
    # Processing the call keyword arguments (line 413)
    kwargs_612 = {}
    # Getting the type of 'open' (line 413)
    open_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'open', False)
    # Calling open(args, kwargs) (line 413)
    open_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 413, 16), open_606, *[Relative_call_result_610, str_611], **kwargs_612)
    
    # Assigning a type to the variable 'inputfile' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'inputfile', open_call_result_613)
    
    # Assigning a Call to a Name (line 414):
    
    # Assigning a Call to a Name (line 414):
    
    # Call to open(...): (line 414)
    # Processing the call arguments (line 414)
    
    # Call to Relative(...): (line 414)
    # Processing the call arguments (line 414)
    str_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 31), 'str', 'tmp2')
    # Processing the call keyword arguments (line 414)
    kwargs_617 = {}
    # Getting the type of 'Relative' (line 414)
    Relative_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 414)
    Relative_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 414, 22), Relative_615, *[str_616], **kwargs_617)
    
    str_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 40), 'str', 'w')
    # Processing the call keyword arguments (line 414)
    kwargs_620 = {}
    # Getting the type of 'open' (line 414)
    open_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 17), 'open', False)
    # Calling open(args, kwargs) (line 414)
    open_call_result_621 = invoke(stypy.reporting.localization.Localization(__file__, 414, 17), open_614, *[Relative_call_result_618, str_619], **kwargs_620)
    
    # Assigning a type to the variable 'outputfile' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'outputfile', open_call_result_621)
    
    # Call to uncompress_it(...): (line 416)
    # Processing the call arguments (line 416)
    # Getting the type of 'inputfile' (line 416)
    inputfile_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 18), 'inputfile', False)
    # Getting the type of 'outputfile' (line 416)
    outputfile_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 29), 'outputfile', False)
    # Processing the call keyword arguments (line 416)
    kwargs_625 = {}
    # Getting the type of 'uncompress_it' (line 416)
    uncompress_it_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'uncompress_it', False)
    # Calling uncompress_it(args, kwargs) (line 416)
    uncompress_it_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 416, 4), uncompress_it_622, *[inputfile_623, outputfile_624], **kwargs_625)
    
    
    # Call to close(...): (line 417)
    # Processing the call keyword arguments (line 417)
    kwargs_629 = {}
    # Getting the type of 'outputfile' (line 417)
    outputfile_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 417)
    close_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 4), outputfile_627, 'close')
    # Calling close(args, kwargs) (line 417)
    close_call_result_630 = invoke(stypy.reporting.localization.Localization(__file__, 417, 4), close_628, *[], **kwargs_629)
    
    
    # Call to close(...): (line 418)
    # Processing the call keyword arguments (line 418)
    kwargs_633 = {}
    # Getting the type of 'inputfile' (line 418)
    inputfile_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 418)
    close_632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 4), inputfile_631, 'close')
    # Calling close(args, kwargs) (line 418)
    close_call_result_634 = invoke(stypy.reporting.localization.Localization(__file__, 418, 4), close_632, *[], **kwargs_633)
    
    pass
    
    # ################# End of 'hardertest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hardertest' in the type store
    # Getting the type of 'stypy_return_type' (line 403)
    stypy_return_type_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hardertest'
    return stypy_return_type_635

# Assigning a type to the variable 'hardertest' (line 403)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'hardertest', hardertest)

# Assigning a Num to a Name (line 427):

# Assigning a Num to a Name (line 427):
float_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 4), 'float')
# Assigning a type to the variable 'f' (line 427)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'f', float_636)

# Assigning a Num to a Name (line 428):

# Assigning a Num to a Name (line 428):
int_637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 4), 'int')
# Assigning a type to the variable 'N' (line 428)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 0), 'N', int_637)

# Assigning a Num to a Name (line 429):

# Assigning a Num to a Name (line 429):
float_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 4), 'float')
# Assigning a type to the variable 'f' (line 429)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 0), 'f', float_638)

# Assigning a Num to a Name (line 430):

# Assigning a Num to a Name (line 430):
int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 4), 'int')
# Assigning a type to the variable 'N' (line 430)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'N', int_639)

# Assigning a Num to a Name (line 431):

# Assigning a Num to a Name (line 431):
float_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 4), 'float')
# Assigning a type to the variable 'f' (line 431)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 0), 'f', float_640)

# Assigning a Num to a Name (line 432):

# Assigning a Num to a Name (line 432):
int_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 4), 'int')
# Assigning a type to the variable 'N' (line 432)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), 'N', int_641)

@norecursion
def compress_it(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compress_it'
    module_type_store = module_type_store.open_function_context('compress_it', 435, 0, False)
    
    # Passed parameters checking function
    compress_it.stypy_localization = localization
    compress_it.stypy_type_of_self = None
    compress_it.stypy_type_store = module_type_store
    compress_it.stypy_function_name = 'compress_it'
    compress_it.stypy_param_names_list = ['inputfile', 'outputfile']
    compress_it.stypy_varargs_param_name = None
    compress_it.stypy_kwargs_param_name = None
    compress_it.stypy_call_defaults = defaults
    compress_it.stypy_call_varargs = varargs
    compress_it.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compress_it', ['inputfile', 'outputfile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compress_it', localization, ['inputfile', 'outputfile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compress_it(...)' code ##################

    str_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, (-1)), 'str', '\n    Make Huffman code for blocks, and\n    Compress from file (possibly stdin).\n    ')
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to findprobs(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'f' (line 440)
    f_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 22), 'f', False)
    # Getting the type of 'N' (line 440)
    N_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'N', False)
    # Processing the call keyword arguments (line 440)
    kwargs_646 = {}
    # Getting the type of 'findprobs' (line 440)
    findprobs_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'findprobs', False)
    # Calling findprobs(args, kwargs) (line 440)
    findprobs_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 440, 12), findprobs_643, *[f_644, N_645], **kwargs_646)
    
    # Assigning a type to the variable 'probs' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'probs', findprobs_call_result_647)
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to makenodes(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'probs' (line 441)
    probs_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 24), 'probs', False)
    # Processing the call keyword arguments (line 441)
    kwargs_650 = {}
    # Getting the type of 'makenodes' (line 441)
    makenodes_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 14), 'makenodes', False)
    # Calling makenodes(args, kwargs) (line 441)
    makenodes_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 441, 14), makenodes_648, *[probs_649], **kwargs_650)
    
    # Assigning a type to the variable 'symbols' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'symbols', makenodes_call_result_651)
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to iterate(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'symbols' (line 445)
    symbols_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'symbols', False)
    # Processing the call keyword arguments (line 444)
    kwargs_654 = {}
    # Getting the type of 'iterate' (line 444)
    iterate_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'iterate', False)
    # Calling iterate(args, kwargs) (line 444)
    iterate_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 444, 11), iterate_652, *[symbols_653], **kwargs_654)
    
    # Assigning a type to the variable 'root' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'root', iterate_call_result_655)
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to read(...): (line 447)
    # Processing the call keyword arguments (line 447)
    kwargs_658 = {}
    # Getting the type of 'inputfile' (line 447)
    inputfile_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 13), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 447)
    read_657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 13), inputfile_656, 'read')
    # Calling read(args, kwargs) (line 447)
    read_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 447, 13), read_657, *[], **kwargs_658)
    
    # Assigning a type to the variable 'string' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'string', read_call_result_659)
    
    # Call to write(...): (line 448)
    # Processing the call arguments (line 448)
    
    # Call to Bencode(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'string' (line 448)
    string_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 29), 'string', False)
    # Getting the type of 'symbols' (line 448)
    symbols_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 37), 'symbols', False)
    # Getting the type of 'N' (line 448)
    N_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 46), 'N', False)
    # Processing the call keyword arguments (line 448)
    kwargs_666 = {}
    # Getting the type of 'Bencode' (line 448)
    Bencode_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 21), 'Bencode', False)
    # Calling Bencode(args, kwargs) (line 448)
    Bencode_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 448, 21), Bencode_662, *[string_663, symbols_664, N_665], **kwargs_666)
    
    # Processing the call keyword arguments (line 448)
    kwargs_668 = {}
    # Getting the type of 'outputfile' (line 448)
    outputfile_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 448)
    write_661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 4), outputfile_660, 'write')
    # Calling write(args, kwargs) (line 448)
    write_call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 448, 4), write_661, *[Bencode_call_result_667], **kwargs_668)
    
    pass
    
    # ################# End of 'compress_it(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compress_it' in the type store
    # Getting the type of 'stypy_return_type' (line 435)
    stypy_return_type_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_670)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compress_it'
    return stypy_return_type_670

# Assigning a type to the variable 'compress_it' (line 435)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'compress_it', compress_it)

@norecursion
def uncompress_it(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'uncompress_it'
    module_type_store = module_type_store.open_function_context('uncompress_it', 452, 0, False)
    
    # Passed parameters checking function
    uncompress_it.stypy_localization = localization
    uncompress_it.stypy_type_of_self = None
    uncompress_it.stypy_type_store = module_type_store
    uncompress_it.stypy_function_name = 'uncompress_it'
    uncompress_it.stypy_param_names_list = ['inputfile', 'outputfile']
    uncompress_it.stypy_varargs_param_name = None
    uncompress_it.stypy_kwargs_param_name = None
    uncompress_it.stypy_call_defaults = defaults
    uncompress_it.stypy_call_varargs = varargs
    uncompress_it.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'uncompress_it', ['inputfile', 'outputfile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'uncompress_it', localization, ['inputfile', 'outputfile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'uncompress_it(...)' code ##################

    str_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, (-1)), 'str', '\n    Make Huffman code for blocks, and\n    UNCompress from file (possibly stdin).\n    ')
    
    # Assigning a Call to a Name (line 457):
    
    # Assigning a Call to a Name (line 457):
    
    # Call to findprobs(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'f' (line 457)
    f_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 22), 'f', False)
    # Getting the type of 'N' (line 457)
    N_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 25), 'N', False)
    # Processing the call keyword arguments (line 457)
    kwargs_675 = {}
    # Getting the type of 'findprobs' (line 457)
    findprobs_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'findprobs', False)
    # Calling findprobs(args, kwargs) (line 457)
    findprobs_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 457, 12), findprobs_672, *[f_673, N_674], **kwargs_675)
    
    # Assigning a type to the variable 'probs' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'probs', findprobs_call_result_676)
    
    # Assigning a Call to a Name (line 460):
    
    # Assigning a Call to a Name (line 460):
    
    # Call to makenodes(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'probs' (line 460)
    probs_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'probs', False)
    # Processing the call keyword arguments (line 460)
    kwargs_679 = {}
    # Getting the type of 'makenodes' (line 460)
    makenodes_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 14), 'makenodes', False)
    # Calling makenodes(args, kwargs) (line 460)
    makenodes_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 460, 14), makenodes_677, *[probs_678], **kwargs_679)
    
    # Assigning a type to the variable 'symbols' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'symbols', makenodes_call_result_680)
    
    # Assigning a Call to a Name (line 461):
    
    # Assigning a Call to a Name (line 461):
    
    # Call to iterate(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'symbols' (line 462)
    symbols_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'symbols', False)
    # Processing the call keyword arguments (line 461)
    kwargs_683 = {}
    # Getting the type of 'iterate' (line 461)
    iterate_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 11), 'iterate', False)
    # Calling iterate(args, kwargs) (line 461)
    iterate_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 461, 11), iterate_681, *[symbols_682], **kwargs_683)
    
    # Assigning a type to the variable 'root' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'root', iterate_call_result_684)
    
    # Assigning a Call to a Name (line 464):
    
    # Assigning a Call to a Name (line 464):
    
    # Call to read(...): (line 464)
    # Processing the call keyword arguments (line 464)
    kwargs_687 = {}
    # Getting the type of 'inputfile' (line 464)
    inputfile_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 13), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 464)
    read_686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 13), inputfile_685, 'read')
    # Calling read(args, kwargs) (line 464)
    read_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 464, 13), read_686, *[], **kwargs_687)
    
    # Assigning a type to the variable 'string' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'string', read_call_result_688)
    
    # Call to write(...): (line 465)
    # Processing the call arguments (line 465)
    
    # Call to Bdecode(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'string' (line 465)
    string_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 29), 'string', False)
    # Getting the type of 'root' (line 465)
    root_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 37), 'root', False)
    # Getting the type of 'N' (line 465)
    N_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 43), 'N', False)
    # Processing the call keyword arguments (line 465)
    kwargs_695 = {}
    # Getting the type of 'Bdecode' (line 465)
    Bdecode_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 21), 'Bdecode', False)
    # Calling Bdecode(args, kwargs) (line 465)
    Bdecode_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 465, 21), Bdecode_691, *[string_692, root_693, N_694], **kwargs_695)
    
    # Processing the call keyword arguments (line 465)
    kwargs_697 = {}
    # Getting the type of 'outputfile' (line 465)
    outputfile_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 465)
    write_690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 4), outputfile_689, 'write')
    # Calling write(args, kwargs) (line 465)
    write_call_result_698 = invoke(stypy.reporting.localization.Localization(__file__, 465, 4), write_690, *[Bdecode_call_result_696], **kwargs_697)
    
    pass
    
    # ################# End of 'uncompress_it(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'uncompress_it' in the type store
    # Getting the type of 'stypy_return_type' (line 452)
    stypy_return_type_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_699)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'uncompress_it'
    return stypy_return_type_699

# Assigning a type to the variable 'uncompress_it' (line 452)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 0), 'uncompress_it', uncompress_it)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 469, 0, False)
    
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

    
    # Call to setrecursionlimit(...): (line 470)
    # Processing the call arguments (line 470)
    int_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 26), 'int')
    # Processing the call keyword arguments (line 470)
    kwargs_703 = {}
    # Getting the type of 'sys' (line 470)
    sys_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'sys', False)
    # Obtaining the member 'setrecursionlimit' of a type (line 470)
    setrecursionlimit_701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 4), sys_700, 'setrecursionlimit')
    # Calling setrecursionlimit(args, kwargs) (line 470)
    setrecursionlimit_call_result_704 = invoke(stypy.reporting.localization.Localization(__file__, 470, 4), setrecursionlimit_701, *[int_702], **kwargs_703)
    
    
    # Call to test(...): (line 471)
    # Processing the call keyword arguments (line 471)
    kwargs_706 = {}
    # Getting the type of 'test' (line 471)
    test_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'test', False)
    # Calling test(args, kwargs) (line 471)
    test_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), test_705, *[], **kwargs_706)
    
    # Getting the type of 'True' (line 472)
    True_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'stypy_return_type', True_708)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 469)
    stypy_return_type_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_709)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_709

# Assigning a type to the variable 'run' (line 469)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'run', run)

# Call to run(...): (line 475)
# Processing the call keyword arguments (line 475)
kwargs_711 = {}
# Getting the type of 'run' (line 475)
run_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'run', False)
# Calling run(args, kwargs) (line 475)
run_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 475, 0), run_710, *[], **kwargs_711)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

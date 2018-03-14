
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
53: class node:
54:     def __init__(self, count, index , name=""):
55:         self.count = float(count)
56:         self.index = index
57:         self.name  = name ## optional argument
58:         if self.name=="" : self.name = '_'+str(index)
59:         self.word = "" ## codeword will go here
60:         self.isinternal = 0
61:     def __cmp__(self, other):
62:         return cmp(self.count, other.count)
63:     def report(self):
64:         if (self.index == 1 ) :
65:             pass#print '#Symbol\tCount\tCodeword'
66:         #print '%s\t(%2.2g)\t%s' % (self.name,self.count,self.word)
67:         pass
68:     def associate(self,internalnode):
69:         self.internalnode = internalnode
70:         internalnode.leaf = 1
71:         internalnode.name = self.name
72:         pass
73: 
74: class internalnode:
75:     def __init__(self):
76:         self.leaf = 0
77:         self.child = []
78:         pass
79:     def children(self,child0,child1):
80:         self.leaf = 0
81:         self.child.append(child0)
82:         self.child.append(child1)
83:         pass
84: 
85: def find_idx(seq, index):
86:     for item in seq:
87:         if item.index == index:
88:             return item
89: 
90: def find_name(seq, name):
91:     for item in seq:
92:         if item.name == name:
93:             return item
94: 
95: def iterate (c) :
96:     '''
97:     Run the Huffman algorithm on the list of "nodes" c.
98:     The list of nodes c is destroyed as we go, then recreated.
99:     Codewords 'co.word' are assigned to each node during the recreation of the list.
100:     The order of the recreated list may well be different.
101:     Use the list c for encoding.
102: 
103:     The root of a new tree of "internalnodes" is returned.
104:     This root should be used when decoding.
105: 
106:     >>> c = [ node(0.5,1,'a'),  \
107:               node(0.25,2,'b'), \
108:               node(0.125,3,'c'),\
109:               node(0.125,4,'d') ]   # my doctest query has been resolved
110:     >>> root = iterate(c)           # "iterate(c)" returns a node, not nothing, and doctest cares!
111:     >>> reportcode(c)               # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
112:     #Symbol   Count     Codeword
113:     a         (0.5)     1
114:     b         (0.25)    01
115:     c         (0.12)    000
116:     d         (0.12)    001
117:     '''
118:     if ( len(c) > 1 ) :
119:         c.sort() ## sort the nodes by count, using the __cmp__ function defined in the node class
120:         deletednode = c[0] ## keep copy of smallest node so that we can put it back in later
121:         second = c[1].index ## index of second smallest node
122:         # MERGE THE BOTTOM TWO
123:         c[1].count += c[0].count  ##  this merged node retains the name of the bigger leaf.
124:         del c[0]
125: 
126:         root = iterate ( c )
127: 
128:         ## Fill in the new information in the ENCODING list, c
129:         ## find the codeword that has been split/joined at this step
130:         co = find_idx(c, second)
131: #        co = find( lambda p: p.index == second , c )
132:         deletednode.word = co.word+'0'
133:         c.append( deletednode )  ## smaller one gets 0
134:         co.word += '1'
135:         co.count -= deletednode.count   ## restore correct count
136: 
137:         ## make the new branches in the DECODING tree
138:         newnode0 = internalnode()
139:         newnode1 = internalnode()
140:         treenode = co.internalnode # find the node that got two children
141:         treenode.children(newnode0,newnode1)
142:         deletednode.associate(newnode0)
143:         co.associate(newnode1)
144:         pass
145:     else :
146:         c[0].word = ""
147:         root = internalnode()
148:         c[0].associate(root)
149:         pass
150:     return root
151: 
152: def encode(sourcelist,code):
153:     '''
154:     Takes a list of source symbols. Returns a binary string.
155:     '''
156:     answer = ""
157:     for s in sourcelist:
158:         co = find_name(code, s)
159: #        co = find(lambda p: p.name == s, code)
160:         if ( not co  ):
161:             print >> sys.stderr, "Warning: symbol",`s`,"has no encoding!"
162:             pass
163:         else:
164:             answer = answer + co.word
165:             pass
166:     return answer
167: 
168: def decode(string,root):
169:     '''
170:     Decodes a binary string using the Huffman tree accessed via root
171:     '''
172:     ## split the string into a list
173:     ## then copy the elements of the list one by one.
174:     answer = []
175:     clist = list( string )
176:     ## start from root
177:     currentnode = root
178:     for c in clist:
179:         if ( c=='\n' ):  continue ## special case for newline characters
180:         assert ( c == '0' )or( c == '1')
181:         currentnode = currentnode.child[int(c)]
182:         if currentnode.leaf != 0:
183:             answer.append( str(currentnode.name) )
184:             currentnode = root
185:         pass
186:     assert (currentnode == root) ## if this is not true then we have run out of characters and are half-way through a codeword
187:     return answer
188: 
189: ## alternate way of calling huffman with a list of counts ## for use as package by other programs ######
190: ## two example ways of using it:
191: #>>> from Huffman3 import *
192: #>>> huffman([1, 2, 3, 4],1)
193: #>>> (c,root) = huffman([1, 2, 3, 4])
194: 
195: ## end ##########################################################################
196: 
197: def makenodes(probs):
198:     '''
199:     Creates a list of nodes ready for the Huffman algorithm.
200:     Each node will receive a codeword when Huffman algorithm "iterate" runs.
201: 
202:     probs should be a list of pairs('<symbol>', <value>).
203: 
204:     >>> probs=[('a',0.5), ('b',0.25), ('c',0.125), ('d',0.125)]
205:     >>> symbols = makenodes(probs)
206:     >>> root = iterate(symbols)
207:     >>> zipped = encode(['a','a','b','a','c','b','c','d'], symbols)
208:     >>> print zipped
209:     1101100001000001
210:     >>> print decode( zipped, root )
211:     ['a', 'a', 'b', 'a', 'c', 'b', 'c', 'd']
212: 
213:     See also the file Example.py for a python program that uses this package.
214:     '''
215:     m=0
216:     c=[]
217:     for p in probs:
218:         m += 1 ;
219:         c.append( node( p[1], m, p[0] ) )
220:         pass
221:     return c
222: 
223: def dec_to_bin( n , digits ):
224:     ''' n is the number to convert to binary;  digits is the number of bits you want
225:     Always prints full number of digits
226:     >>> print dec_to_bin( 17 , 9)
227:     000010001
228:     >>> print dec_to_bin( 17 , 5)
229:     10001
230: 
231:     Will behead the standard binary number if requested
232:     >>> print dec_to_bin( 17 , 4)
233:     0001
234:     '''
235:     if(n<0) :
236:         sys.stderr.write( "warning, negative n not expected\n")
237:     i=digits-1
238:     ans=""
239:     while i>=0 :
240:         b = (((1<<i)&n)>0)
241:         i -= 1
242:         ans = ans + str(int(b))
243:     return ans
244: 
245: verbose=0
246: 
247: def weight(string):
248:     '''
249:     ## returns number of 0s and number of 1s in the string
250:     >>> print weight("00011")
251:     (3, 2)
252:     '''
253:     w0=0;w1=0
254:     for c in list(string):
255:         if(c=='0'):
256:             w0+=1
257:             pass
258:         elif(c=='1'):
259:             w1+=1
260:             pass
261:         pass
262:     return (w0,w1)
263: 
264: 
265: def findprobs(f=0.01,N=6):
266:     ''' Find probabilities of all the events
267:     000000
268:     000001
269:      ...
270:     111111
271:     <-N ->
272:     >>> print findprobs(0.1,3)              # doctest:+ELLIPSIS
273:     [('000', 0.7...),..., ('111', 0.001...)]
274:     '''
275:     answer = []
276:     for n in range(2**N):
277:         s = dec_to_bin(n,N)
278:         (w0,w1) = weight(s)
279:         if verbose and 0 :
280:             pass#print s,w0,w1
281:         answer.append( (s, f**w1 * (1-f)**w0 ) )
282:         pass
283:     assert ( len(answer) == 2**N )
284:     return answer
285: 
286: def Bencode(string,symbols,N):
287:     '''
288:     Reads in a string of 0s and 1s.
289:     Creates a list of blocks of size N.
290:     Sends this list to the general-purpose Huffman encoder
291:     defined by the nodes in the list "symbols".
292:     '''
293:     blocks = []
294:     chars = list(string)
295: 
296:     s=""
297:     for c in chars:
298:         s = s+c
299:         if (len(s)>=N):
300:             blocks.append( s )
301:             s = ""
302:             pass
303:         pass
304:     if (len(s)>0):
305:         print >> sys.stderr, "warning, padding last block with 0s"
306:         while (len(s)<N):
307:             s = s+'0'
308:             pass
309:         blocks.append( s )
310:         pass
311: 
312:     if verbose:
313:         #print "blocks to be encoded:"
314:         #print blocks
315:         pass
316:     zipped = encode( blocks , symbols )
317:     return zipped
318: 
319: def Bdecode(string,root,N):
320:     '''
321:     Decode a binary string into blocks.
322:     '''
323:     answer = decode( string, root )
324:     if verbose:
325:         #print "blocks from decoder:"
326:         #print answer
327:         pass
328:     output = "".join( answer )
329:     ## this assumes that the source file was an exact multiple of the blocklength
330:     return output
331: 
332: def easytest():
333:     '''
334:     Tests block code with N=3, f=0.01 on a tiny example.
335:     >>> easytest()                 # doctest:+NORMALIZE_WHITESPACE
336:     #Symbol     Count           Codeword
337:     000         (0.97)          1
338:     001         (0.0098)        001
339:     010         (0.0098)        010
340:     011         (9.9e-05)       00001
341:     100         (0.0098)        011
342:     101         (9.9e-05)       00010
343:     110         (9.9e-05)       00011
344:     111         (1e-06)         00000
345:     zipped  = 1001010000010110111
346:     decoded = ['000', '001', '010', '011', '100', '100', '000']
347:     OK!
348:     '''
349:     N=3
350:     f=0.01
351:     probs = findprobs(f,N)
352: #    if len(probs) > 999 :
353: #        sys.setrecursionlimit( len(probs)+100 )
354:     symbols = makenodes(probs) # makenodes is defined at the bottom of Huffman3 package
355:     root = iterate(symbols) # make huffman code and put it into the symbols' nodes, and return the root of the decoding tree
356: 
357:     symbols.sort(lambda x, y: cmp(x.index, y.index)) # sort by index
358:     for co in symbols :                              # and write the answer
359:         co.report()
360: 
361:     source = ['000','001','010','011','100','100','000']
362:     zipped = encode(source, symbols)
363:     #print "zipped  =",zipped
364:     answer = decode( zipped, root )
365:     #print "decoded =",answer
366:     if ( source != answer ):
367:         pass#print "ERROR"
368:     else:
369:         pass#print "OK!"
370:     pass
371: 
372: def test():
373:     easytest()
374:     hardertest()
375: 
376: def Relative(path):
377:     return os.path.join(os.path.dirname(__file__), path)
378: 
379: def hardertest():
380:     #print "Reading the BentCoinFile"
381:     inputfile = open( Relative("testdata/BentCoinFile") , "r" )
382:     outputfile = open( Relative("tmp.zip") , "w" )
383:     #print  "Compressing to tmp.zip"
384:     compress_it(inputfile, outputfile)
385:     outputfile.close();     inputfile.close()
386: #    print "DONE compressing"
387: 
388:     inputfile = open( Relative("tmp.zip") , "r" )
389:     outputfile = open( Relative("tmp2") , "w" )
390:     #print  "Uncompressing to tmp2"
391:     uncompress_it(inputfile, outputfile)
392:     outputfile.close();     inputfile.close()
393: #    print "DONE uncompressing"
394: 
395: ##    print "Checking for differences..."
396: ##    os.system( "diff testdata/BentCoinFile tmp2" )
397: ##    os.system( "wc tmp.zip testdata/BentCoinFile tmp2" )
398:     pass
399: 
400: f=0.01; N=12   #  1244 bits if N==12
401: f=0.01; N=5   #  2266  bits if N==5
402: f=0.01; N=10   #  1379 bits if N==10
403: 
404: def compress_it( inputfile, outputfile ):
405:     '''
406:     Make Huffman code for blocks, and
407:     Compress from file (possibly stdin).
408:     '''
409:     probs = findprobs(f,N)
410:     symbols = makenodes(probs)
411: #    if len(probs) > 999 :
412: #        sys.setrecursionlimit( len(probs)+100 )
413:     root = iterate(symbols) # make huffman code and put it into the symbols' nodes, and return the root of the decoding tree
414: 
415:     string = inputfile.read()
416:     outputfile.write( Bencode(string, symbols, N) )
417:     pass
418: 
419: def uncompress_it( inputfile, outputfile ):
420:     '''
421:     Make Huffman code for blocks, and
422:     UNCompress from file (possibly stdin).
423:     '''
424:     probs = findprobs(f,N)
425: #    if len(probs) > 999 :
426: #        sys.setrecursionlimit( len(probs)+100 )
427:     symbols = makenodes(probs)
428:     root = iterate(symbols) # make huffman code and put it into the symbols' nodes, and return the root of the decoding tree
429: 
430:     string = inputfile.read()
431:     outputfile.write( Bdecode(string, root, N) )
432:     pass
433: 
434: def run():
435:     sys.setrecursionlimit( 10000 )
436:     test()
437:     return True
438: 
439: run()

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
        str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 43), 'str', '')
        defaults = [str_5]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Attribute (line 55):
        
        # Assigning a Call to a Attribute (line 55):
        
        # Call to float(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'count' (line 55)
        count_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'count', False)
        # Processing the call keyword arguments (line 55)
        kwargs_8 = {}
        # Getting the type of 'float' (line 55)
        float_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'float', False)
        # Calling float(args, kwargs) (line 55)
        float_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), float_6, *[count_7], **kwargs_8)
        
        # Getting the type of 'self' (line 55)
        self_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'count' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_10, 'count', float_call_result_9)
        
        # Assigning a Name to a Attribute (line 56):
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'index' (line 56)
        index_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'index')
        # Getting the type of 'self' (line 56)
        self_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'index' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_12, 'index', index_11)
        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'name' (line 57)
        name_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'name')
        # Getting the type of 'self' (line 57)
        self_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'name' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_14, 'name', name_13)
        
        # Getting the type of 'self' (line 58)
        self_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'self')
        # Obtaining the member 'name' of a type (line 58)
        name_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), self_15, 'name')
        str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'str', '')
        # Applying the binary operator '==' (line 58)
        result_eq_18 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), '==', name_16, str_17)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), result_eq_18):
            pass
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_19 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), result_eq_18)
            # Assigning a type to the variable 'if_condition_19' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_19', if_condition_19)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Attribute (line 58):
            
            # Assigning a BinOp to a Attribute (line 58):
            str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 39), 'str', '_')
            
            # Call to str(...): (line 58)
            # Processing the call arguments (line 58)
            # Getting the type of 'index' (line 58)
            index_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 47), 'index', False)
            # Processing the call keyword arguments (line 58)
            kwargs_23 = {}
            # Getting the type of 'str' (line 58)
            str_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'str', False)
            # Calling str(args, kwargs) (line 58)
            str_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 58, 43), str_21, *[index_22], **kwargs_23)
            
            # Applying the binary operator '+' (line 58)
            result_add_25 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 39), '+', str_20, str_call_result_24)
            
            # Getting the type of 'self' (line 58)
            self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'self')
            # Setting the type of the member 'name' of a type (line 58)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 27), self_26, 'name', result_add_25)
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Attribute (line 59):
        
        # Assigning a Str to a Attribute (line 59):
        str_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'str', '')
        # Getting the type of 'self' (line 59)
        self_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'word' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_28, 'word', str_27)
        
        # Assigning a Num to a Attribute (line 60):
        
        # Assigning a Num to a Attribute (line 60):
        int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'int')
        # Getting the type of 'self' (line 60)
        self_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member 'isinternal' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_30, 'isinternal', int_29)
        
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
        module_type_store = module_type_store.open_function_context('__cmp__', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
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

        
        # Call to cmp(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'self', False)
        # Obtaining the member 'count' of a type (line 62)
        count_33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), self_32, 'count')
        # Getting the type of 'other' (line 62)
        other_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'other', False)
        # Obtaining the member 'count' of a type (line 62)
        count_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 31), other_34, 'count')
        # Processing the call keyword arguments (line 62)
        kwargs_36 = {}
        # Getting the type of 'cmp' (line 62)
        cmp_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'cmp', False)
        # Calling cmp(args, kwargs) (line 62)
        cmp_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), cmp_31, *[count_33, count_35], **kwargs_36)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', cmp_call_result_37)
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
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
        module_type_store = module_type_store.open_function_context('report', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
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

        
        # Getting the type of 'self' (line 64)
        self_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self')
        # Obtaining the member 'index' of a type (line 64)
        index_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_39, 'index')
        int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'int')
        # Applying the binary operator '==' (line 64)
        result_eq_42 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '==', index_40, int_41)
        
        # Testing if the type of an if condition is none (line 64)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_42):
            pass
        else:
            
            # Testing the type of an if condition (line 64)
            if_condition_43 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_42)
            # Assigning a type to the variable 'if_condition_43' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_43', if_condition_43)
            # SSA begins for if statement (line 64)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 64)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        
        # ################# End of 'report(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'report' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
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
        module_type_store = module_type_store.open_function_context('associate', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
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

        
        # Assigning a Name to a Attribute (line 69):
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'internalnode' (line 69)
        internalnode_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'internalnode')
        # Getting the type of 'self' (line 69)
        self_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'internalnode' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_46, 'internalnode', internalnode_45)
        
        # Assigning a Num to a Attribute (line 70):
        
        # Assigning a Num to a Attribute (line 70):
        int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'int')
        # Getting the type of 'internalnode' (line 70)
        internalnode_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'internalnode')
        # Setting the type of the member 'leaf' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), internalnode_48, 'leaf', int_47)
        
        # Assigning a Attribute to a Attribute (line 71):
        
        # Assigning a Attribute to a Attribute (line 71):
        # Getting the type of 'self' (line 71)
        self_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'self')
        # Obtaining the member 'name' of a type (line 71)
        name_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 28), self_49, 'name')
        # Getting the type of 'internalnode' (line 71)
        internalnode_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'internalnode')
        # Setting the type of the member 'name' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), internalnode_51, 'name', name_50)
        pass
        
        # ################# End of 'associate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'associate' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'associate'
        return stypy_return_type_52


# Assigning a type to the variable 'node' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'node', node)
# Declaration of the 'internalnode' class

class internalnode:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
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

        
        # Assigning a Num to a Attribute (line 76):
        
        # Assigning a Num to a Attribute (line 76):
        int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'int')
        # Getting the type of 'self' (line 76)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'leaf' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_54, 'leaf', int_53)
        
        # Assigning a List to a Attribute (line 77):
        
        # Assigning a List to a Attribute (line 77):
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        
        # Getting the type of 'self' (line 77)
        self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'child' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_56, 'child', list_55)
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
        module_type_store = module_type_store.open_function_context('children', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
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

        
        # Assigning a Num to a Attribute (line 80):
        
        # Assigning a Num to a Attribute (line 80):
        int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'int')
        # Getting the type of 'self' (line 80)
        self_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member 'leaf' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_58, 'leaf', int_57)
        
        # Call to append(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'child0' (line 81)
        child0_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'child0', False)
        # Processing the call keyword arguments (line 81)
        kwargs_63 = {}
        # Getting the type of 'self' (line 81)
        self_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'child' of a type (line 81)
        child_60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_59, 'child')
        # Obtaining the member 'append' of a type (line 81)
        append_61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), child_60, 'append')
        # Calling append(args, kwargs) (line 81)
        append_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), append_61, *[child0_62], **kwargs_63)
        
        
        # Call to append(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'child1' (line 82)
        child1_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'child1', False)
        # Processing the call keyword arguments (line 82)
        kwargs_69 = {}
        # Getting the type of 'self' (line 82)
        self_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self', False)
        # Obtaining the member 'child' of a type (line 82)
        child_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_65, 'child')
        # Obtaining the member 'append' of a type (line 82)
        append_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), child_66, 'append')
        # Calling append(args, kwargs) (line 82)
        append_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), append_67, *[child1_68], **kwargs_69)
        
        pass
        
        # ################# End of 'children(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'children' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'children'
        return stypy_return_type_71


# Assigning a type to the variable 'internalnode' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'internalnode', internalnode)

@norecursion
def find_idx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_idx'
    module_type_store = module_type_store.open_function_context('find_idx', 85, 0, False)
    
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

    
    # Getting the type of 'seq' (line 86)
    seq_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'seq')
    # Assigning a type to the variable 'seq_72' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'seq_72', seq_72)
    # Testing if the for loop is going to be iterated (line 86)
    # Testing the type of a for loop iterable (line 86)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 4), seq_72)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 86, 4), seq_72):
        # Getting the type of the for loop variable (line 86)
        for_loop_var_73 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 4), seq_72)
        # Assigning a type to the variable 'item' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'item', for_loop_var_73)
        # SSA begins for a for statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'item' (line 87)
        item_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'item')
        # Obtaining the member 'index' of a type (line 87)
        index_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), item_74, 'index')
        # Getting the type of 'index' (line 87)
        index_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'index')
        # Applying the binary operator '==' (line 87)
        result_eq_77 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 11), '==', index_75, index_76)
        
        # Testing if the type of an if condition is none (line 87)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 8), result_eq_77):
            pass
        else:
            
            # Testing the type of an if condition (line 87)
            if_condition_78 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), result_eq_77)
            # Assigning a type to the variable 'if_condition_78' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'if_condition_78', if_condition_78)
            # SSA begins for if statement (line 87)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'item' (line 88)
            item_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'item')
            # Assigning a type to the variable 'stypy_return_type' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', item_79)
            # SSA join for if statement (line 87)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'find_idx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_idx' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_80)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_idx'
    return stypy_return_type_80

# Assigning a type to the variable 'find_idx' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'find_idx', find_idx)

@norecursion
def find_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_name'
    module_type_store = module_type_store.open_function_context('find_name', 90, 0, False)
    
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

    
    # Getting the type of 'seq' (line 91)
    seq_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'seq')
    # Assigning a type to the variable 'seq_81' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'seq_81', seq_81)
    # Testing if the for loop is going to be iterated (line 91)
    # Testing the type of a for loop iterable (line 91)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 4), seq_81)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 91, 4), seq_81):
        # Getting the type of the for loop variable (line 91)
        for_loop_var_82 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 4), seq_81)
        # Assigning a type to the variable 'item' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'item', for_loop_var_82)
        # SSA begins for a for statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'item' (line 92)
        item_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'item')
        # Obtaining the member 'name' of a type (line 92)
        name_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), item_83, 'name')
        # Getting the type of 'name' (line 92)
        name_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'name')
        # Applying the binary operator '==' (line 92)
        result_eq_86 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), '==', name_84, name_85)
        
        # Testing if the type of an if condition is none (line 92)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 8), result_eq_86):
            pass
        else:
            
            # Testing the type of an if condition (line 92)
            if_condition_87 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_eq_86)
            # Assigning a type to the variable 'if_condition_87' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_87', if_condition_87)
            # SSA begins for if statement (line 92)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'item' (line 93)
            item_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'item')
            # Assigning a type to the variable 'stypy_return_type' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'stypy_return_type', item_88)
            # SSA join for if statement (line 92)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'find_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_name' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_name'
    return stypy_return_type_89

# Assigning a type to the variable 'find_name' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'find_name', find_name)

@norecursion
def iterate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterate'
    module_type_store = module_type_store.open_function_context('iterate', 95, 0, False)
    
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

    str_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n    Run the Huffman algorithm on the list of "nodes" c.\n    The list of nodes c is destroyed as we go, then recreated.\n    Codewords \'co.word\' are assigned to each node during the recreation of the list.\n    The order of the recreated list may well be different.\n    Use the list c for encoding.\n\n    The root of a new tree of "internalnodes" is returned.\n    This root should be used when decoding.\n\n    >>> c = [ node(0.5,1,\'a\'),                node(0.25,2,\'b\'),               node(0.125,3,\'c\'),              node(0.125,4,\'d\') ]   # my doctest query has been resolved\n    >>> root = iterate(c)           # "iterate(c)" returns a node, not nothing, and doctest cares!\n    >>> reportcode(c)               # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS\n    #Symbol   Count     Codeword\n    a         (0.5)     1\n    b         (0.25)    01\n    c         (0.12)    000\n    d         (0.12)    001\n    ')
    
    
    # Call to len(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'c' (line 118)
    c_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 13), 'c', False)
    # Processing the call keyword arguments (line 118)
    kwargs_93 = {}
    # Getting the type of 'len' (line 118)
    len_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 9), 'len', False)
    # Calling len(args, kwargs) (line 118)
    len_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 118, 9), len_91, *[c_92], **kwargs_93)
    
    int_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'int')
    # Applying the binary operator '>' (line 118)
    result_gt_96 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 9), '>', len_call_result_94, int_95)
    
    # Testing if the type of an if condition is none (line 118)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 118, 4), result_gt_96):
        
        # Assigning a Str to a Attribute (line 146):
        
        # Assigning a Str to a Attribute (line 146):
        str_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'str', '')
        
        # Obtaining the type of the subscript
        int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 10), 'int')
        # Getting the type of 'c' (line 146)
        c_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), c_187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___188, int_186)
        
        # Setting the type of the member 'word' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_189, 'word', str_185)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to internalnode(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_191 = {}
        # Getting the type of 'internalnode' (line 147)
        internalnode_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 147)
        internalnode_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), internalnode_190, *[], **kwargs_191)
        
        # Assigning a type to the variable 'root' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'root', internalnode_call_result_192)
        
        # Call to associate(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'root' (line 148)
        root_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'root', False)
        # Processing the call keyword arguments (line 148)
        kwargs_199 = {}
        
        # Obtaining the type of the subscript
        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 10), 'int')
        # Getting the type of 'c' (line 148)
        c_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), c_194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), getitem___195, int_193)
        
        # Obtaining the member 'associate' of a type (line 148)
        associate_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), subscript_call_result_196, 'associate')
        # Calling associate(args, kwargs) (line 148)
        associate_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), associate_197, *[root_198], **kwargs_199)
        
        pass
    else:
        
        # Testing the type of an if condition (line 118)
        if_condition_97 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), result_gt_96)
        # Assigning a type to the variable 'if_condition_97' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_97', if_condition_97)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to sort(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_100 = {}
        # Getting the type of 'c' (line 119)
        c_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'c', False)
        # Obtaining the member 'sort' of a type (line 119)
        sort_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), c_98, 'sort')
        # Calling sort(args, kwargs) (line 119)
        sort_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), sort_99, *[], **kwargs_100)
        
        
        # Assigning a Subscript to a Name (line 120):
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 24), 'int')
        # Getting the type of 'c' (line 120)
        c_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'c')
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 22), c_103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 120, 22), getitem___104, int_102)
        
        # Assigning a type to the variable 'deletednode' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'deletednode', subscript_call_result_105)
        
        # Assigning a Attribute to a Name (line 121):
        
        # Assigning a Attribute to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'int')
        # Getting the type of 'c' (line 121)
        c_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'c')
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), c_107, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), getitem___108, int_106)
        
        # Obtaining the member 'index' of a type (line 121)
        index_110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), subscript_call_result_109, 'index')
        # Assigning a type to the variable 'second' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'second', index_110)
        
        
        # Obtaining the type of the subscript
        int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 10), 'int')
        # Getting the type of 'c' (line 123)
        c_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), c_112, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___113, int_111)
        
        # Obtaining the member 'count' of a type (line 123)
        count_115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), subscript_call_result_114, 'count')
        
        # Obtaining the type of the subscript
        int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'int')
        # Getting the type of 'c' (line 123)
        c_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'c')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 22), c_117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 123, 22), getitem___118, int_116)
        
        # Obtaining the member 'count' of a type (line 123)
        count_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 22), subscript_call_result_119, 'count')
        # Applying the binary operator '+=' (line 123)
        result_iadd_121 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 8), '+=', count_115, count_120)
        
        # Obtaining the type of the subscript
        int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 10), 'int')
        # Getting the type of 'c' (line 123)
        c_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), c_123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___124, int_122)
        
        # Setting the type of the member 'count' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), subscript_call_result_125, 'count', result_iadd_121)
        
        # Deleting a member
        # Getting the type of 'c' (line 124)
        c_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'c')
        
        # Obtaining the type of the subscript
        int_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'int')
        # Getting the type of 'c' (line 124)
        c_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'c')
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), c_128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), getitem___129, int_127)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), c_126, subscript_call_result_130)
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to iterate(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'c' (line 126)
        c_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'c', False)
        # Processing the call keyword arguments (line 126)
        kwargs_133 = {}
        # Getting the type of 'iterate' (line 126)
        iterate_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'iterate', False)
        # Calling iterate(args, kwargs) (line 126)
        iterate_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), iterate_131, *[c_132], **kwargs_133)
        
        # Assigning a type to the variable 'root' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'root', iterate_call_result_134)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to find_idx(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'c' (line 130)
        c_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'c', False)
        # Getting the type of 'second' (line 130)
        second_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'second', False)
        # Processing the call keyword arguments (line 130)
        kwargs_138 = {}
        # Getting the type of 'find_idx' (line 130)
        find_idx_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'find_idx', False)
        # Calling find_idx(args, kwargs) (line 130)
        find_idx_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 130, 13), find_idx_135, *[c_136, second_137], **kwargs_138)
        
        # Assigning a type to the variable 'co' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'co', find_idx_call_result_139)
        
        # Assigning a BinOp to a Attribute (line 132):
        
        # Assigning a BinOp to a Attribute (line 132):
        # Getting the type of 'co' (line 132)
        co_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'co')
        # Obtaining the member 'word' of a type (line 132)
        word_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 27), co_140, 'word')
        str_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'str', '0')
        # Applying the binary operator '+' (line 132)
        result_add_143 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 27), '+', word_141, str_142)
        
        # Getting the type of 'deletednode' (line 132)
        deletednode_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'deletednode')
        # Setting the type of the member 'word' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), deletednode_144, 'word', result_add_143)
        
        # Call to append(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'deletednode' (line 133)
        deletednode_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'deletednode', False)
        # Processing the call keyword arguments (line 133)
        kwargs_148 = {}
        # Getting the type of 'c' (line 133)
        c_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'c', False)
        # Obtaining the member 'append' of a type (line 133)
        append_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), c_145, 'append')
        # Calling append(args, kwargs) (line 133)
        append_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), append_146, *[deletednode_147], **kwargs_148)
        
        
        # Getting the type of 'co' (line 134)
        co_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'co')
        # Obtaining the member 'word' of a type (line 134)
        word_151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), co_150, 'word')
        str_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'str', '1')
        # Applying the binary operator '+=' (line 134)
        result_iadd_153 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 8), '+=', word_151, str_152)
        # Getting the type of 'co' (line 134)
        co_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'co')
        # Setting the type of the member 'word' of a type (line 134)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), co_154, 'word', result_iadd_153)
        
        
        # Getting the type of 'co' (line 135)
        co_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'co')
        # Obtaining the member 'count' of a type (line 135)
        count_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), co_155, 'count')
        # Getting the type of 'deletednode' (line 135)
        deletednode_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'deletednode')
        # Obtaining the member 'count' of a type (line 135)
        count_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 20), deletednode_157, 'count')
        # Applying the binary operator '-=' (line 135)
        result_isub_159 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 8), '-=', count_156, count_158)
        # Getting the type of 'co' (line 135)
        co_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'co')
        # Setting the type of the member 'count' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), co_160, 'count', result_isub_159)
        
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to internalnode(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_162 = {}
        # Getting the type of 'internalnode' (line 138)
        internalnode_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 138)
        internalnode_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), internalnode_161, *[], **kwargs_162)
        
        # Assigning a type to the variable 'newnode0' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'newnode0', internalnode_call_result_163)
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to internalnode(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_165 = {}
        # Getting the type of 'internalnode' (line 139)
        internalnode_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 139)
        internalnode_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), internalnode_164, *[], **kwargs_165)
        
        # Assigning a type to the variable 'newnode1' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'newnode1', internalnode_call_result_166)
        
        # Assigning a Attribute to a Name (line 140):
        
        # Assigning a Attribute to a Name (line 140):
        # Getting the type of 'co' (line 140)
        co_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'co')
        # Obtaining the member 'internalnode' of a type (line 140)
        internalnode_168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 19), co_167, 'internalnode')
        # Assigning a type to the variable 'treenode' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'treenode', internalnode_168)
        
        # Call to children(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'newnode0' (line 141)
        newnode0_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'newnode0', False)
        # Getting the type of 'newnode1' (line 141)
        newnode1_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 35), 'newnode1', False)
        # Processing the call keyword arguments (line 141)
        kwargs_173 = {}
        # Getting the type of 'treenode' (line 141)
        treenode_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'treenode', False)
        # Obtaining the member 'children' of a type (line 141)
        children_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), treenode_169, 'children')
        # Calling children(args, kwargs) (line 141)
        children_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), children_170, *[newnode0_171, newnode1_172], **kwargs_173)
        
        
        # Call to associate(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'newnode0' (line 142)
        newnode0_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'newnode0', False)
        # Processing the call keyword arguments (line 142)
        kwargs_178 = {}
        # Getting the type of 'deletednode' (line 142)
        deletednode_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'deletednode', False)
        # Obtaining the member 'associate' of a type (line 142)
        associate_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), deletednode_175, 'associate')
        # Calling associate(args, kwargs) (line 142)
        associate_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), associate_176, *[newnode0_177], **kwargs_178)
        
        
        # Call to associate(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'newnode1' (line 143)
        newnode1_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'newnode1', False)
        # Processing the call keyword arguments (line 143)
        kwargs_183 = {}
        # Getting the type of 'co' (line 143)
        co_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'co', False)
        # Obtaining the member 'associate' of a type (line 143)
        associate_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), co_180, 'associate')
        # Calling associate(args, kwargs) (line 143)
        associate_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), associate_181, *[newnode1_182], **kwargs_183)
        
        pass
        # SSA branch for the else part of an if statement (line 118)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Attribute (line 146):
        
        # Assigning a Str to a Attribute (line 146):
        str_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'str', '')
        
        # Obtaining the type of the subscript
        int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 10), 'int')
        # Getting the type of 'c' (line 146)
        c_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'c')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), c_187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___188, int_186)
        
        # Setting the type of the member 'word' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_189, 'word', str_185)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to internalnode(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_191 = {}
        # Getting the type of 'internalnode' (line 147)
        internalnode_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'internalnode', False)
        # Calling internalnode(args, kwargs) (line 147)
        internalnode_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), internalnode_190, *[], **kwargs_191)
        
        # Assigning a type to the variable 'root' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'root', internalnode_call_result_192)
        
        # Call to associate(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'root' (line 148)
        root_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'root', False)
        # Processing the call keyword arguments (line 148)
        kwargs_199 = {}
        
        # Obtaining the type of the subscript
        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 10), 'int')
        # Getting the type of 'c' (line 148)
        c_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), c_194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), getitem___195, int_193)
        
        # Obtaining the member 'associate' of a type (line 148)
        associate_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), subscript_call_result_196, 'associate')
        # Calling associate(args, kwargs) (line 148)
        associate_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), associate_197, *[root_198], **kwargs_199)
        
        pass
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'root' (line 150)
    root_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'root')
    # Assigning a type to the variable 'stypy_return_type' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type', root_201)
    
    # ################# End of 'iterate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterate' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_202)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterate'
    return stypy_return_type_202

# Assigning a type to the variable 'iterate' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'iterate', iterate)

@norecursion
def encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'encode'
    module_type_store = module_type_store.open_function_context('encode', 152, 0, False)
    
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

    str_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'str', '\n    Takes a list of source symbols. Returns a binary string.\n    ')
    
    # Assigning a Str to a Name (line 156):
    
    # Assigning a Str to a Name (line 156):
    str_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 13), 'str', '')
    # Assigning a type to the variable 'answer' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'answer', str_204)
    
    # Getting the type of 'sourcelist' (line 157)
    sourcelist_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'sourcelist')
    # Assigning a type to the variable 'sourcelist_205' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'sourcelist_205', sourcelist_205)
    # Testing if the for loop is going to be iterated (line 157)
    # Testing the type of a for loop iterable (line 157)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 4), sourcelist_205)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 4), sourcelist_205):
        # Getting the type of the for loop variable (line 157)
        for_loop_var_206 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 4), sourcelist_205)
        # Assigning a type to the variable 's' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 's', for_loop_var_206)
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to find_name(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'code' (line 158)
        code_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'code', False)
        # Getting the type of 's' (line 158)
        s_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 's', False)
        # Processing the call keyword arguments (line 158)
        kwargs_210 = {}
        # Getting the type of 'find_name' (line 158)
        find_name_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), 'find_name', False)
        # Calling find_name(args, kwargs) (line 158)
        find_name_call_result_211 = invoke(stypy.reporting.localization.Localization(__file__, 158, 13), find_name_207, *[code_208, s_209], **kwargs_210)
        
        # Assigning a type to the variable 'co' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'co', find_name_call_result_211)
        
        # Getting the type of 'co' (line 160)
        co_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'co')
        # Applying the 'not' unary operator (line 160)
        result_not__213 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 13), 'not', co_212)
        
        # Testing if the type of an if condition is none (line 160)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 8), result_not__213):
            
            # Assigning a BinOp to a Name (line 164):
            
            # Assigning a BinOp to a Name (line 164):
            # Getting the type of 'answer' (line 164)
            answer_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'answer')
            # Getting the type of 'co' (line 164)
            co_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'co')
            # Obtaining the member 'word' of a type (line 164)
            word_220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 30), co_219, 'word')
            # Applying the binary operator '+' (line 164)
            result_add_221 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 21), '+', answer_218, word_220)
            
            # Assigning a type to the variable 'answer' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'answer', result_add_221)
            pass
        else:
            
            # Testing the type of an if condition (line 160)
            if_condition_214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_not__213)
            # Assigning a type to the variable 'if_condition_214' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_214', if_condition_214)
            # SSA begins for if statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 33), 'str', 'Warning: symbol')
            # Getting the type of 's' (line 161)
            s_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 52), 's')
            str_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 55), 'str', 'has no encoding!')
            pass
            # SSA branch for the else part of an if statement (line 160)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 164):
            
            # Assigning a BinOp to a Name (line 164):
            # Getting the type of 'answer' (line 164)
            answer_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'answer')
            # Getting the type of 'co' (line 164)
            co_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'co')
            # Obtaining the member 'word' of a type (line 164)
            word_220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 30), co_219, 'word')
            # Applying the binary operator '+' (line 164)
            result_add_221 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 21), '+', answer_218, word_220)
            
            # Assigning a type to the variable 'answer' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'answer', result_add_221)
            pass
            # SSA join for if statement (line 160)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'answer' (line 166)
    answer_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'answer')
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type', answer_222)
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_223)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_223

# Assigning a type to the variable 'encode' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'encode', encode)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'decode'
    module_type_store = module_type_store.open_function_context('decode', 168, 0, False)
    
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

    str_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', '\n    Decodes a binary string using the Huffman tree accessed via root\n    ')
    
    # Assigning a List to a Name (line 174):
    
    # Assigning a List to a Name (line 174):
    
    # Obtaining an instance of the builtin type 'list' (line 174)
    list_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 174)
    
    # Assigning a type to the variable 'answer' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'answer', list_225)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to list(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'string' (line 175)
    string_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'string', False)
    # Processing the call keyword arguments (line 175)
    kwargs_228 = {}
    # Getting the type of 'list' (line 175)
    list_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'list', False)
    # Calling list(args, kwargs) (line 175)
    list_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), list_226, *[string_227], **kwargs_228)
    
    # Assigning a type to the variable 'clist' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'clist', list_call_result_229)
    
    # Assigning a Name to a Name (line 177):
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'root' (line 177)
    root_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'root')
    # Assigning a type to the variable 'currentnode' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'currentnode', root_230)
    
    # Getting the type of 'clist' (line 178)
    clist_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'clist')
    # Assigning a type to the variable 'clist_231' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'clist_231', clist_231)
    # Testing if the for loop is going to be iterated (line 178)
    # Testing the type of a for loop iterable (line 178)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 178, 4), clist_231)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 178, 4), clist_231):
        # Getting the type of the for loop variable (line 178)
        for_loop_var_232 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 178, 4), clist_231)
        # Assigning a type to the variable 'c' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'c', for_loop_var_232)
        # SSA begins for a for statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'c' (line 179)
        c_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'c')
        str_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'str', '\n')
        # Applying the binary operator '==' (line 179)
        result_eq_235 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 13), '==', c_233, str_234)
        
        # Testing if the type of an if condition is none (line 179)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 179, 8), result_eq_235):
            pass
        else:
            
            # Testing the type of an if condition (line 179)
            if_condition_236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 8), result_eq_235)
            # Assigning a type to the variable 'if_condition_236' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'if_condition_236', if_condition_236)
            # SSA begins for if statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 179)
            module_type_store = module_type_store.join_ssa_context()
            

        # Evaluating assert statement condition
        
        # Evaluating a boolean operation
        
        # Getting the type of 'c' (line 180)
        c_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'c')
        str_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 22), 'str', '0')
        # Applying the binary operator '==' (line 180)
        result_eq_239 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 17), '==', c_237, str_238)
        
        
        # Getting the type of 'c' (line 180)
        c_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'c')
        str_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 36), 'str', '1')
        # Applying the binary operator '==' (line 180)
        result_eq_242 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 31), '==', c_240, str_241)
        
        # Applying the binary operator 'or' (line 180)
        result_or_keyword_243 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 15), 'or', result_eq_239, result_eq_242)
        
        assert_244 = result_or_keyword_243
        # Assigning a type to the variable 'assert_244' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_244', result_or_keyword_243)
        
        # Assigning a Subscript to a Name (line 181):
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        
        # Call to int(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'c' (line 181)
        c_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 44), 'c', False)
        # Processing the call keyword arguments (line 181)
        kwargs_247 = {}
        # Getting the type of 'int' (line 181)
        int_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 40), 'int', False)
        # Calling int(args, kwargs) (line 181)
        int_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 181, 40), int_245, *[c_246], **kwargs_247)
        
        # Getting the type of 'currentnode' (line 181)
        currentnode_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'currentnode')
        # Obtaining the member 'child' of a type (line 181)
        child_250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 22), currentnode_249, 'child')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 22), child_250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 181, 22), getitem___251, int_call_result_248)
        
        # Assigning a type to the variable 'currentnode' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'currentnode', subscript_call_result_252)
        
        # Getting the type of 'currentnode' (line 182)
        currentnode_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'currentnode')
        # Obtaining the member 'leaf' of a type (line 182)
        leaf_254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), currentnode_253, 'leaf')
        int_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 31), 'int')
        # Applying the binary operator '!=' (line 182)
        result_ne_256 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), '!=', leaf_254, int_255)
        
        # Testing if the type of an if condition is none (line 182)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 182, 8), result_ne_256):
            pass
        else:
            
            # Testing the type of an if condition (line 182)
            if_condition_257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), result_ne_256)
            # Assigning a type to the variable 'if_condition_257' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_257', if_condition_257)
            # SSA begins for if statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 183)
            # Processing the call arguments (line 183)
            
            # Call to str(...): (line 183)
            # Processing the call arguments (line 183)
            # Getting the type of 'currentnode' (line 183)
            currentnode_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'currentnode', False)
            # Obtaining the member 'name' of a type (line 183)
            name_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 31), currentnode_261, 'name')
            # Processing the call keyword arguments (line 183)
            kwargs_263 = {}
            # Getting the type of 'str' (line 183)
            str_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 27), 'str', False)
            # Calling str(args, kwargs) (line 183)
            str_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 183, 27), str_260, *[name_262], **kwargs_263)
            
            # Processing the call keyword arguments (line 183)
            kwargs_265 = {}
            # Getting the type of 'answer' (line 183)
            answer_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'answer', False)
            # Obtaining the member 'append' of a type (line 183)
            append_259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), answer_258, 'append')
            # Calling append(args, kwargs) (line 183)
            append_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), append_259, *[str_call_result_264], **kwargs_265)
            
            
            # Assigning a Name to a Name (line 184):
            
            # Assigning a Name to a Name (line 184):
            # Getting the type of 'root' (line 184)
            root_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'root')
            # Assigning a type to the variable 'currentnode' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'currentnode', root_267)
            # SSA join for if statement (line 182)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Evaluating assert statement condition
    
    # Getting the type of 'currentnode' (line 186)
    currentnode_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'currentnode')
    # Getting the type of 'root' (line 186)
    root_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'root')
    # Applying the binary operator '==' (line 186)
    result_eq_270 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 12), '==', currentnode_268, root_269)
    
    assert_271 = result_eq_270
    # Assigning a type to the variable 'assert_271' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'assert_271', result_eq_270)
    # Getting the type of 'answer' (line 187)
    answer_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'answer')
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', answer_272)
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_273

# Assigning a type to the variable 'decode' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'decode', decode)

@norecursion
def makenodes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'makenodes'
    module_type_store = module_type_store.open_function_context('makenodes', 197, 0, False)
    
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

    str_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '\n    Creates a list of nodes ready for the Huffman algorithm.\n    Each node will receive a codeword when Huffman algorithm "iterate" runs.\n\n    probs should be a list of pairs(\'<symbol>\', <value>).\n\n    >>> probs=[(\'a\',0.5), (\'b\',0.25), (\'c\',0.125), (\'d\',0.125)]\n    >>> symbols = makenodes(probs)\n    >>> root = iterate(symbols)\n    >>> zipped = encode([\'a\',\'a\',\'b\',\'a\',\'c\',\'b\',\'c\',\'d\'], symbols)\n    >>> print zipped\n    1101100001000001\n    >>> print decode( zipped, root )\n    [\'a\', \'a\', \'b\', \'a\', \'c\', \'b\', \'c\', \'d\']\n\n    See also the file Example.py for a python program that uses this package.\n    ')
    
    # Assigning a Num to a Name (line 215):
    
    # Assigning a Num to a Name (line 215):
    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 6), 'int')
    # Assigning a type to the variable 'm' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'm', int_275)
    
    # Assigning a List to a Name (line 216):
    
    # Assigning a List to a Name (line 216):
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 6), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    
    # Assigning a type to the variable 'c' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'c', list_276)
    
    # Getting the type of 'probs' (line 217)
    probs_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'probs')
    # Assigning a type to the variable 'probs_277' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'probs_277', probs_277)
    # Testing if the for loop is going to be iterated (line 217)
    # Testing the type of a for loop iterable (line 217)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 217, 4), probs_277)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 217, 4), probs_277):
        # Getting the type of the for loop variable (line 217)
        for_loop_var_278 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 217, 4), probs_277)
        # Assigning a type to the variable 'p' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'p', for_loop_var_278)
        # SSA begins for a for statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'm' (line 218)
        m_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'm')
        int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 13), 'int')
        # Applying the binary operator '+=' (line 218)
        result_iadd_281 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 8), '+=', m_279, int_280)
        # Assigning a type to the variable 'm' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'm', result_iadd_281)
        
        
        # Call to append(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Call to node(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Obtaining the type of the subscript
        int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 26), 'int')
        # Getting the type of 'p' (line 219)
        p_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 24), p_286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 219, 24), getitem___287, int_285)
        
        # Getting the type of 'm' (line 219)
        m_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'm', False)
        
        # Obtaining the type of the subscript
        int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 35), 'int')
        # Getting the type of 'p' (line 219)
        p_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 33), p_291, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 219, 33), getitem___292, int_290)
        
        # Processing the call keyword arguments (line 219)
        kwargs_294 = {}
        # Getting the type of 'node' (line 219)
        node_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 18), 'node', False)
        # Calling node(args, kwargs) (line 219)
        node_call_result_295 = invoke(stypy.reporting.localization.Localization(__file__, 219, 18), node_284, *[subscript_call_result_288, m_289, subscript_call_result_293], **kwargs_294)
        
        # Processing the call keyword arguments (line 219)
        kwargs_296 = {}
        # Getting the type of 'c' (line 219)
        c_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'c', False)
        # Obtaining the member 'append' of a type (line 219)
        append_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), c_282, 'append')
        # Calling append(args, kwargs) (line 219)
        append_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), append_283, *[node_call_result_295], **kwargs_296)
        
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'c' (line 221)
    c_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', c_298)
    
    # ################# End of 'makenodes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'makenodes' in the type store
    # Getting the type of 'stypy_return_type' (line 197)
    stypy_return_type_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'makenodes'
    return stypy_return_type_299

# Assigning a type to the variable 'makenodes' (line 197)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'makenodes', makenodes)

@norecursion
def dec_to_bin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dec_to_bin'
    module_type_store = module_type_store.open_function_context('dec_to_bin', 223, 0, False)
    
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

    str_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', ' n is the number to convert to binary;  digits is the number of bits you want\n    Always prints full number of digits\n    >>> print dec_to_bin( 17 , 9)\n    000010001\n    >>> print dec_to_bin( 17 , 5)\n    10001\n\n    Will behead the standard binary number if requested\n    >>> print dec_to_bin( 17 , 4)\n    0001\n    ')
    
    # Getting the type of 'n' (line 235)
    n_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'n')
    int_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 9), 'int')
    # Applying the binary operator '<' (line 235)
    result_lt_303 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 7), '<', n_301, int_302)
    
    # Testing if the type of an if condition is none (line 235)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 235, 4), result_lt_303):
        pass
    else:
        
        # Testing the type of an if condition (line 235)
        if_condition_304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), result_lt_303)
        # Assigning a type to the variable 'if_condition_304' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_304', if_condition_304)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 236)
        # Processing the call arguments (line 236)
        str_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'str', 'warning, negative n not expected\n')
        # Processing the call keyword arguments (line 236)
        kwargs_309 = {}
        # Getting the type of 'sys' (line 236)
        sys_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 236)
        stderr_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), sys_305, 'stderr')
        # Obtaining the member 'write' of a type (line 236)
        write_307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), stderr_306, 'write')
        # Calling write(args, kwargs) (line 236)
        write_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), write_307, *[str_308], **kwargs_309)
        
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 237):
    
    # Assigning a BinOp to a Name (line 237):
    # Getting the type of 'digits' (line 237)
    digits_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 6), 'digits')
    int_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 13), 'int')
    # Applying the binary operator '-' (line 237)
    result_sub_313 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 6), '-', digits_311, int_312)
    
    # Assigning a type to the variable 'i' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'i', result_sub_313)
    
    # Assigning a Str to a Name (line 238):
    
    # Assigning a Str to a Name (line 238):
    str_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 8), 'str', '')
    # Assigning a type to the variable 'ans' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'ans', str_314)
    
    
    # Getting the type of 'i' (line 239)
    i_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 10), 'i')
    int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 13), 'int')
    # Applying the binary operator '>=' (line 239)
    result_ge_317 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 10), '>=', i_315, int_316)
    
    # Assigning a type to the variable 'result_ge_317' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'result_ge_317', result_ge_317)
    # Testing if the while is going to be iterated (line 239)
    # Testing the type of an if condition (line 239)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 4), result_ge_317)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 239, 4), result_ge_317):
        # SSA begins for while statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Compare to a Name (line 240):
        
        # Assigning a Compare to a Name (line 240):
        
        int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 15), 'int')
        # Getting the type of 'i' (line 240)
        i_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'i')
        # Applying the binary operator '<<' (line 240)
        result_lshift_320 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), '<<', int_318, i_319)
        
        # Getting the type of 'n' (line 240)
        n_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'n')
        # Applying the binary operator '&' (line 240)
        result_and__322 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 14), '&', result_lshift_320, n_321)
        
        int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 24), 'int')
        # Applying the binary operator '>' (line 240)
        result_gt_324 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 13), '>', result_and__322, int_323)
        
        # Assigning a type to the variable 'b' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'b', result_gt_324)
        
        # Getting the type of 'i' (line 241)
        i_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'i')
        int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 13), 'int')
        # Applying the binary operator '-=' (line 241)
        result_isub_327 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 8), '-=', i_325, int_326)
        # Assigning a type to the variable 'i' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'i', result_isub_327)
        
        
        # Assigning a BinOp to a Name (line 242):
        
        # Assigning a BinOp to a Name (line 242):
        # Getting the type of 'ans' (line 242)
        ans_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'ans')
        
        # Call to str(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Call to int(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'b' (line 242)
        b_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'b', False)
        # Processing the call keyword arguments (line 242)
        kwargs_332 = {}
        # Getting the type of 'int' (line 242)
        int_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'int', False)
        # Calling int(args, kwargs) (line 242)
        int_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 242, 24), int_330, *[b_331], **kwargs_332)
        
        # Processing the call keyword arguments (line 242)
        kwargs_334 = {}
        # Getting the type of 'str' (line 242)
        str_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'str', False)
        # Calling str(args, kwargs) (line 242)
        str_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), str_329, *[int_call_result_333], **kwargs_334)
        
        # Applying the binary operator '+' (line 242)
        result_add_336 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 14), '+', ans_328, str_call_result_335)
        
        # Assigning a type to the variable 'ans' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'ans', result_add_336)
        # SSA join for while statement (line 239)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'ans' (line 243)
    ans_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'ans')
    # Assigning a type to the variable 'stypy_return_type' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type', ans_337)
    
    # ################# End of 'dec_to_bin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dec_to_bin' in the type store
    # Getting the type of 'stypy_return_type' (line 223)
    stypy_return_type_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_338)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dec_to_bin'
    return stypy_return_type_338

# Assigning a type to the variable 'dec_to_bin' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'dec_to_bin', dec_to_bin)

# Assigning a Num to a Name (line 245):

# Assigning a Num to a Name (line 245):
int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 8), 'int')
# Assigning a type to the variable 'verbose' (line 245)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'verbose', int_339)

@norecursion
def weight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'weight'
    module_type_store = module_type_store.open_function_context('weight', 247, 0, False)
    
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

    str_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'str', '\n    ## returns number of 0s and number of 1s in the string\n    >>> print weight("00011")\n    (3, 2)\n    ')
    
    # Assigning a Num to a Name (line 253):
    
    # Assigning a Num to a Name (line 253):
    int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 7), 'int')
    # Assigning a type to the variable 'w0' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'w0', int_341)
    
    # Assigning a Num to a Name (line 253):
    
    # Assigning a Num to a Name (line 253):
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 12), 'int')
    # Assigning a type to the variable 'w1' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 9), 'w1', int_342)
    
    
    # Call to list(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'string' (line 254)
    string_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 18), 'string', False)
    # Processing the call keyword arguments (line 254)
    kwargs_345 = {}
    # Getting the type of 'list' (line 254)
    list_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'list', False)
    # Calling list(args, kwargs) (line 254)
    list_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 254, 13), list_343, *[string_344], **kwargs_345)
    
    # Assigning a type to the variable 'list_call_result_346' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'list_call_result_346', list_call_result_346)
    # Testing if the for loop is going to be iterated (line 254)
    # Testing the type of a for loop iterable (line 254)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 4), list_call_result_346)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 254, 4), list_call_result_346):
        # Getting the type of the for loop variable (line 254)
        for_loop_var_347 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 4), list_call_result_346)
        # Assigning a type to the variable 'c' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'c', for_loop_var_347)
        # SSA begins for a for statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'c' (line 255)
        c_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'c')
        str_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 14), 'str', '0')
        # Applying the binary operator '==' (line 255)
        result_eq_350 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 11), '==', c_348, str_349)
        
        # Testing if the type of an if condition is none (line 255)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 255, 8), result_eq_350):
            
            # Getting the type of 'c' (line 258)
            c_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'c')
            str_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'str', '1')
            # Applying the binary operator '==' (line 258)
            result_eq_357 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 13), '==', c_355, str_356)
            
            # Testing if the type of an if condition is none (line 258)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 258, 12), result_eq_357):
                pass
            else:
                
                # Testing the type of an if condition (line 258)
                if_condition_358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 12), result_eq_357)
                # Assigning a type to the variable 'if_condition_358' (line 258)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'if_condition_358', if_condition_358)
                # SSA begins for if statement (line 258)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'w1' (line 259)
                w1_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'w1')
                int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 16), 'int')
                # Applying the binary operator '+=' (line 259)
                result_iadd_361 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 12), '+=', w1_359, int_360)
                # Assigning a type to the variable 'w1' (line 259)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'w1', result_iadd_361)
                
                pass
                # SSA join for if statement (line 258)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 255)
            if_condition_351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_eq_350)
            # Assigning a type to the variable 'if_condition_351' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_351', if_condition_351)
            # SSA begins for if statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'w0' (line 256)
            w0_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'w0')
            int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 16), 'int')
            # Applying the binary operator '+=' (line 256)
            result_iadd_354 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 12), '+=', w0_352, int_353)
            # Assigning a type to the variable 'w0' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'w0', result_iadd_354)
            
            pass
            # SSA branch for the else part of an if statement (line 255)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'c' (line 258)
            c_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'c')
            str_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'str', '1')
            # Applying the binary operator '==' (line 258)
            result_eq_357 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 13), '==', c_355, str_356)
            
            # Testing if the type of an if condition is none (line 258)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 258, 12), result_eq_357):
                pass
            else:
                
                # Testing the type of an if condition (line 258)
                if_condition_358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 12), result_eq_357)
                # Assigning a type to the variable 'if_condition_358' (line 258)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'if_condition_358', if_condition_358)
                # SSA begins for if statement (line 258)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'w1' (line 259)
                w1_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'w1')
                int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 16), 'int')
                # Applying the binary operator '+=' (line 259)
                result_iadd_361 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 12), '+=', w1_359, int_360)
                # Assigning a type to the variable 'w1' (line 259)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'w1', result_iadd_361)
                
                pass
                # SSA join for if statement (line 258)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    # Getting the type of 'w0' (line 262)
    w0_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'w0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 12), tuple_362, w0_363)
    # Adding element type (line 262)
    # Getting the type of 'w1' (line 262)
    w1_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'w1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 12), tuple_362, w1_364)
    
    # Assigning a type to the variable 'stypy_return_type' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type', tuple_362)
    
    # ################# End of 'weight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'weight' in the type store
    # Getting the type of 'stypy_return_type' (line 247)
    stypy_return_type_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_365)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'weight'
    return stypy_return_type_365

# Assigning a type to the variable 'weight' (line 247)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'weight', weight)

@norecursion
def findprobs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 16), 'float')
    int_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 23), 'int')
    defaults = [float_366, int_367]
    # Create a new context for function 'findprobs'
    module_type_store = module_type_store.open_function_context('findprobs', 265, 0, False)
    
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

    str_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, (-1)), 'str', " Find probabilities of all the events\n    000000\n    000001\n     ...\n    111111\n    <-N ->\n    >>> print findprobs(0.1,3)              # doctest:+ELLIPSIS\n    [('000', 0.7...),..., ('111', 0.001...)]\n    ")
    
    # Assigning a List to a Name (line 275):
    
    # Assigning a List to a Name (line 275):
    
    # Obtaining an instance of the builtin type 'list' (line 275)
    list_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 275)
    
    # Assigning a type to the variable 'answer' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'answer', list_369)
    
    
    # Call to range(...): (line 276)
    # Processing the call arguments (line 276)
    int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'int')
    # Getting the type of 'N' (line 276)
    N_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'N', False)
    # Applying the binary operator '**' (line 276)
    result_pow_373 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 19), '**', int_371, N_372)
    
    # Processing the call keyword arguments (line 276)
    kwargs_374 = {}
    # Getting the type of 'range' (line 276)
    range_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 13), 'range', False)
    # Calling range(args, kwargs) (line 276)
    range_call_result_375 = invoke(stypy.reporting.localization.Localization(__file__, 276, 13), range_370, *[result_pow_373], **kwargs_374)
    
    # Assigning a type to the variable 'range_call_result_375' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'range_call_result_375', range_call_result_375)
    # Testing if the for loop is going to be iterated (line 276)
    # Testing the type of a for loop iterable (line 276)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 276, 4), range_call_result_375)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 276, 4), range_call_result_375):
        # Getting the type of the for loop variable (line 276)
        for_loop_var_376 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 276, 4), range_call_result_375)
        # Assigning a type to the variable 'n' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'n', for_loop_var_376)
        # SSA begins for a for statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 277):
        
        # Assigning a Call to a Name (line 277):
        
        # Call to dec_to_bin(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'n' (line 277)
        n_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'n', False)
        # Getting the type of 'N' (line 277)
        N_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'N', False)
        # Processing the call keyword arguments (line 277)
        kwargs_380 = {}
        # Getting the type of 'dec_to_bin' (line 277)
        dec_to_bin_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'dec_to_bin', False)
        # Calling dec_to_bin(args, kwargs) (line 277)
        dec_to_bin_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), dec_to_bin_377, *[n_378, N_379], **kwargs_380)
        
        # Assigning a type to the variable 's' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 's', dec_to_bin_call_result_381)
        
        # Assigning a Call to a Tuple (line 278):
        
        # Assigning a Call to a Name:
        
        # Call to weight(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 's' (line 278)
        s_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 25), 's', False)
        # Processing the call keyword arguments (line 278)
        kwargs_384 = {}
        # Getting the type of 'weight' (line 278)
        weight_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'weight', False)
        # Calling weight(args, kwargs) (line 278)
        weight_call_result_385 = invoke(stypy.reporting.localization.Localization(__file__, 278, 18), weight_382, *[s_383], **kwargs_384)
        
        # Assigning a type to the variable 'call_assignment_1' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_1', weight_call_result_385)
        
        # Assigning a Call to a Name (line 278):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 278)
        call_assignment_1_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_387 = stypy_get_value_from_tuple(call_assignment_1_386, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_2' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_2', stypy_get_value_from_tuple_call_result_387)
        
        # Assigning a Name to a Name (line 278):
        # Getting the type of 'call_assignment_2' (line 278)
        call_assignment_2_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_2')
        # Assigning a type to the variable 'w0' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 9), 'w0', call_assignment_2_388)
        
        # Assigning a Call to a Name (line 278):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 278)
        call_assignment_1_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_390 = stypy_get_value_from_tuple(call_assignment_1_389, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_3', stypy_get_value_from_tuple_call_result_390)
        
        # Assigning a Name to a Name (line 278):
        # Getting the type of 'call_assignment_3' (line 278)
        call_assignment_3_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_3')
        # Assigning a type to the variable 'w1' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'w1', call_assignment_3_391)
        
        # Evaluating a boolean operation
        # Getting the type of 'verbose' (line 279)
        verbose_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'verbose')
        int_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 23), 'int')
        # Applying the binary operator 'and' (line 279)
        result_and_keyword_394 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), 'and', verbose_392, int_393)
        
        # Testing if the type of an if condition is none (line 279)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 279, 8), result_and_keyword_394):
            pass
        else:
            
            # Testing the type of an if condition (line 279)
            if_condition_395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_and_keyword_394)
            # Assigning a type to the variable 'if_condition_395' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_395', if_condition_395)
            # SSA begins for if statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 279)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 281)
        # Processing the call arguments (line 281)
        
        # Obtaining an instance of the builtin type 'tuple' (line 281)
        tuple_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 281)
        # Adding element type (line 281)
        # Getting the type of 's' (line 281)
        s_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 24), tuple_398, s_399)
        # Adding element type (line 281)
        # Getting the type of 'f' (line 281)
        f_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'f', False)
        # Getting the type of 'w1' (line 281)
        w1_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 30), 'w1', False)
        # Applying the binary operator '**' (line 281)
        result_pow_402 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 27), '**', f_400, w1_401)
        
        int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 36), 'int')
        # Getting the type of 'f' (line 281)
        f_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 38), 'f', False)
        # Applying the binary operator '-' (line 281)
        result_sub_405 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 36), '-', int_403, f_404)
        
        # Getting the type of 'w0' (line 281)
        w0_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 42), 'w0', False)
        # Applying the binary operator '**' (line 281)
        result_pow_407 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 35), '**', result_sub_405, w0_406)
        
        # Applying the binary operator '*' (line 281)
        result_mul_408 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 27), '*', result_pow_402, result_pow_407)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 24), tuple_398, result_mul_408)
        
        # Processing the call keyword arguments (line 281)
        kwargs_409 = {}
        # Getting the type of 'answer' (line 281)
        answer_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'answer', False)
        # Obtaining the member 'append' of a type (line 281)
        append_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), answer_396, 'append')
        # Calling append(args, kwargs) (line 281)
        append_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), append_397, *[tuple_398], **kwargs_409)
        
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Evaluating assert statement condition
    
    
    # Call to len(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'answer' (line 283)
    answer_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 17), 'answer', False)
    # Processing the call keyword arguments (line 283)
    kwargs_413 = {}
    # Getting the type of 'len' (line 283)
    len_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 13), 'len', False)
    # Calling len(args, kwargs) (line 283)
    len_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 283, 13), len_411, *[answer_412], **kwargs_413)
    
    int_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'int')
    # Getting the type of 'N' (line 283)
    N_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 31), 'N')
    # Applying the binary operator '**' (line 283)
    result_pow_417 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 28), '**', int_415, N_416)
    
    # Applying the binary operator '==' (line 283)
    result_eq_418 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 13), '==', len_call_result_414, result_pow_417)
    
    assert_419 = result_eq_418
    # Assigning a type to the variable 'assert_419' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'assert_419', result_eq_418)
    # Getting the type of 'answer' (line 284)
    answer_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'answer')
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type', answer_420)
    
    # ################# End of 'findprobs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findprobs' in the type store
    # Getting the type of 'stypy_return_type' (line 265)
    stypy_return_type_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findprobs'
    return stypy_return_type_421

# Assigning a type to the variable 'findprobs' (line 265)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'findprobs', findprobs)

@norecursion
def Bencode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Bencode'
    module_type_store = module_type_store.open_function_context('Bencode', 286, 0, False)
    
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

    str_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, (-1)), 'str', '\n    Reads in a string of 0s and 1s.\n    Creates a list of blocks of size N.\n    Sends this list to the general-purpose Huffman encoder\n    defined by the nodes in the list "symbols".\n    ')
    
    # Assigning a List to a Name (line 293):
    
    # Assigning a List to a Name (line 293):
    
    # Obtaining an instance of the builtin type 'list' (line 293)
    list_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 293)
    
    # Assigning a type to the variable 'blocks' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'blocks', list_423)
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to list(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'string' (line 294)
    string_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'string', False)
    # Processing the call keyword arguments (line 294)
    kwargs_426 = {}
    # Getting the type of 'list' (line 294)
    list_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'list', False)
    # Calling list(args, kwargs) (line 294)
    list_call_result_427 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), list_424, *[string_425], **kwargs_426)
    
    # Assigning a type to the variable 'chars' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'chars', list_call_result_427)
    
    # Assigning a Str to a Name (line 296):
    
    # Assigning a Str to a Name (line 296):
    str_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 6), 'str', '')
    # Assigning a type to the variable 's' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 's', str_428)
    
    # Getting the type of 'chars' (line 297)
    chars_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'chars')
    # Assigning a type to the variable 'chars_429' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'chars_429', chars_429)
    # Testing if the for loop is going to be iterated (line 297)
    # Testing the type of a for loop iterable (line 297)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 297, 4), chars_429)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 297, 4), chars_429):
        # Getting the type of the for loop variable (line 297)
        for_loop_var_430 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 297, 4), chars_429)
        # Assigning a type to the variable 'c' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'c', for_loop_var_430)
        # SSA begins for a for statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 298):
        
        # Assigning a BinOp to a Name (line 298):
        # Getting the type of 's' (line 298)
        s_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 's')
        # Getting the type of 'c' (line 298)
        c_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 14), 'c')
        # Applying the binary operator '+' (line 298)
        result_add_433 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 12), '+', s_431, c_432)
        
        # Assigning a type to the variable 's' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 's', result_add_433)
        
        
        # Call to len(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 's' (line 299)
        s_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 's', False)
        # Processing the call keyword arguments (line 299)
        kwargs_436 = {}
        # Getting the type of 'len' (line 299)
        len_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'len', False)
        # Calling len(args, kwargs) (line 299)
        len_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), len_434, *[s_435], **kwargs_436)
        
        # Getting the type of 'N' (line 299)
        N_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'N')
        # Applying the binary operator '>=' (line 299)
        result_ge_439 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 12), '>=', len_call_result_437, N_438)
        
        # Testing if the type of an if condition is none (line 299)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 299, 8), result_ge_439):
            pass
        else:
            
            # Testing the type of an if condition (line 299)
            if_condition_440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 8), result_ge_439)
            # Assigning a type to the variable 'if_condition_440' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'if_condition_440', if_condition_440)
            # SSA begins for if statement (line 299)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 300)
            # Processing the call arguments (line 300)
            # Getting the type of 's' (line 300)
            s_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 's', False)
            # Processing the call keyword arguments (line 300)
            kwargs_444 = {}
            # Getting the type of 'blocks' (line 300)
            blocks_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'blocks', False)
            # Obtaining the member 'append' of a type (line 300)
            append_442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), blocks_441, 'append')
            # Calling append(args, kwargs) (line 300)
            append_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), append_442, *[s_443], **kwargs_444)
            
            
            # Assigning a Str to a Name (line 301):
            
            # Assigning a Str to a Name (line 301):
            str_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 16), 'str', '')
            # Assigning a type to the variable 's' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 's', str_446)
            pass
            # SSA join for if statement (line 299)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to len(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 's' (line 304)
    s_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 's', False)
    # Processing the call keyword arguments (line 304)
    kwargs_449 = {}
    # Getting the type of 'len' (line 304)
    len_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'len', False)
    # Calling len(args, kwargs) (line 304)
    len_call_result_450 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), len_447, *[s_448], **kwargs_449)
    
    int_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 15), 'int')
    # Applying the binary operator '>' (line 304)
    result_gt_452 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 8), '>', len_call_result_450, int_451)
    
    # Testing if the type of an if condition is none (line 304)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 304, 4), result_gt_452):
        pass
    else:
        
        # Testing the type of an if condition (line 304)
        if_condition_453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 4), result_gt_452)
        # Assigning a type to the variable 'if_condition_453' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'if_condition_453', if_condition_453)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 29), 'str', 'warning, padding last block with 0s')
        
        
        
        # Call to len(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 's' (line 306)
        s_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 's', False)
        # Processing the call keyword arguments (line 306)
        kwargs_457 = {}
        # Getting the type of 'len' (line 306)
        len_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'len', False)
        # Calling len(args, kwargs) (line 306)
        len_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 306, 15), len_455, *[s_456], **kwargs_457)
        
        # Getting the type of 'N' (line 306)
        N_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'N')
        # Applying the binary operator '<' (line 306)
        result_lt_460 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 15), '<', len_call_result_458, N_459)
        
        # Assigning a type to the variable 'result_lt_460' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'result_lt_460', result_lt_460)
        # Testing if the while is going to be iterated (line 306)
        # Testing the type of an if condition (line 306)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), result_lt_460)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 306, 8), result_lt_460):
            # SSA begins for while statement (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a BinOp to a Name (line 307):
            
            # Assigning a BinOp to a Name (line 307):
            # Getting the type of 's' (line 307)
            s_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 's')
            str_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 18), 'str', '0')
            # Applying the binary operator '+' (line 307)
            result_add_463 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 16), '+', s_461, str_462)
            
            # Assigning a type to the variable 's' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 's', result_add_463)
            pass
            # SSA join for while statement (line 306)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 's' (line 309)
        s_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 's', False)
        # Processing the call keyword arguments (line 309)
        kwargs_467 = {}
        # Getting the type of 'blocks' (line 309)
        blocks_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'blocks', False)
        # Obtaining the member 'append' of a type (line 309)
        append_465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), blocks_464, 'append')
        # Calling append(args, kwargs) (line 309)
        append_call_result_468 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), append_465, *[s_466], **kwargs_467)
        
        pass
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'verbose' (line 312)
    verbose_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 7), 'verbose')
    # Testing if the type of an if condition is none (line 312)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 312, 4), verbose_469):
        pass
    else:
        
        # Testing the type of an if condition (line 312)
        if_condition_470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 4), verbose_469)
        # Assigning a type to the variable 'if_condition_470' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'if_condition_470', if_condition_470)
        # SSA begins for if statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 312)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to encode(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'blocks' (line 316)
    blocks_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 21), 'blocks', False)
    # Getting the type of 'symbols' (line 316)
    symbols_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 30), 'symbols', False)
    # Processing the call keyword arguments (line 316)
    kwargs_474 = {}
    # Getting the type of 'encode' (line 316)
    encode_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'encode', False)
    # Calling encode(args, kwargs) (line 316)
    encode_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), encode_471, *[blocks_472, symbols_473], **kwargs_474)
    
    # Assigning a type to the variable 'zipped' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'zipped', encode_call_result_475)
    # Getting the type of 'zipped' (line 317)
    zipped_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'zipped')
    # Assigning a type to the variable 'stypy_return_type' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type', zipped_476)
    
    # ################# End of 'Bencode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Bencode' in the type store
    # Getting the type of 'stypy_return_type' (line 286)
    stypy_return_type_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_477)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Bencode'
    return stypy_return_type_477

# Assigning a type to the variable 'Bencode' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'Bencode', Bencode)

@norecursion
def Bdecode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Bdecode'
    module_type_store = module_type_store.open_function_context('Bdecode', 319, 0, False)
    
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

    str_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, (-1)), 'str', '\n    Decode a binary string into blocks.\n    ')
    
    # Assigning a Call to a Name (line 323):
    
    # Assigning a Call to a Name (line 323):
    
    # Call to decode(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'string' (line 323)
    string_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'string', False)
    # Getting the type of 'root' (line 323)
    root_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 29), 'root', False)
    # Processing the call keyword arguments (line 323)
    kwargs_482 = {}
    # Getting the type of 'decode' (line 323)
    decode_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'decode', False)
    # Calling decode(args, kwargs) (line 323)
    decode_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 323, 13), decode_479, *[string_480, root_481], **kwargs_482)
    
    # Assigning a type to the variable 'answer' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'answer', decode_call_result_483)
    # Getting the type of 'verbose' (line 324)
    verbose_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 7), 'verbose')
    # Testing if the type of an if condition is none (line 324)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 324, 4), verbose_484):
        pass
    else:
        
        # Testing the type of an if condition (line 324)
        if_condition_485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 4), verbose_484)
        # Assigning a type to the variable 'if_condition_485' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'if_condition_485', if_condition_485)
        # SSA begins for if statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 324)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to join(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'answer' (line 328)
    answer_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 22), 'answer', False)
    # Processing the call keyword arguments (line 328)
    kwargs_489 = {}
    str_486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 13), 'str', '')
    # Obtaining the member 'join' of a type (line 328)
    join_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 13), str_486, 'join')
    # Calling join(args, kwargs) (line 328)
    join_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 328, 13), join_487, *[answer_488], **kwargs_489)
    
    # Assigning a type to the variable 'output' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'output', join_call_result_490)
    # Getting the type of 'output' (line 330)
    output_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type', output_491)
    
    # ################# End of 'Bdecode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Bdecode' in the type store
    # Getting the type of 'stypy_return_type' (line 319)
    stypy_return_type_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Bdecode'
    return stypy_return_type_492

# Assigning a type to the variable 'Bdecode' (line 319)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'Bdecode', Bdecode)

@norecursion
def easytest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'easytest'
    module_type_store = module_type_store.open_function_context('easytest', 332, 0, False)
    
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

    str_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, (-1)), 'str', "\n    Tests block code with N=3, f=0.01 on a tiny example.\n    >>> easytest()                 # doctest:+NORMALIZE_WHITESPACE\n    #Symbol     Count           Codeword\n    000         (0.97)          1\n    001         (0.0098)        001\n    010         (0.0098)        010\n    011         (9.9e-05)       00001\n    100         (0.0098)        011\n    101         (9.9e-05)       00010\n    110         (9.9e-05)       00011\n    111         (1e-06)         00000\n    zipped  = 1001010000010110111\n    decoded = ['000', '001', '010', '011', '100', '100', '000']\n    OK!\n    ")
    
    # Assigning a Num to a Name (line 349):
    
    # Assigning a Num to a Name (line 349):
    int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 6), 'int')
    # Assigning a type to the variable 'N' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'N', int_494)
    
    # Assigning a Num to a Name (line 350):
    
    # Assigning a Num to a Name (line 350):
    float_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 6), 'float')
    # Assigning a type to the variable 'f' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'f', float_495)
    
    # Assigning a Call to a Name (line 351):
    
    # Assigning a Call to a Name (line 351):
    
    # Call to findprobs(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'f' (line 351)
    f_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 22), 'f', False)
    # Getting the type of 'N' (line 351)
    N_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'N', False)
    # Processing the call keyword arguments (line 351)
    kwargs_499 = {}
    # Getting the type of 'findprobs' (line 351)
    findprobs_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'findprobs', False)
    # Calling findprobs(args, kwargs) (line 351)
    findprobs_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), findprobs_496, *[f_497, N_498], **kwargs_499)
    
    # Assigning a type to the variable 'probs' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'probs', findprobs_call_result_500)
    
    # Assigning a Call to a Name (line 354):
    
    # Assigning a Call to a Name (line 354):
    
    # Call to makenodes(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'probs' (line 354)
    probs_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'probs', False)
    # Processing the call keyword arguments (line 354)
    kwargs_503 = {}
    # Getting the type of 'makenodes' (line 354)
    makenodes_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 14), 'makenodes', False)
    # Calling makenodes(args, kwargs) (line 354)
    makenodes_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 354, 14), makenodes_501, *[probs_502], **kwargs_503)
    
    # Assigning a type to the variable 'symbols' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'symbols', makenodes_call_result_504)
    
    # Assigning a Call to a Name (line 355):
    
    # Assigning a Call to a Name (line 355):
    
    # Call to iterate(...): (line 355)
    # Processing the call arguments (line 355)
    # Getting the type of 'symbols' (line 355)
    symbols_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 'symbols', False)
    # Processing the call keyword arguments (line 355)
    kwargs_507 = {}
    # Getting the type of 'iterate' (line 355)
    iterate_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'iterate', False)
    # Calling iterate(args, kwargs) (line 355)
    iterate_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 355, 11), iterate_505, *[symbols_506], **kwargs_507)
    
    # Assigning a type to the variable 'root' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'root', iterate_call_result_508)
    
    # Call to sort(...): (line 357)
    # Processing the call arguments (line 357)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 357, 17, True)
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

        
        # Call to cmp(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'x' (line 357)
        x_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 34), 'x', False)
        # Obtaining the member 'index' of a type (line 357)
        index_513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 34), x_512, 'index')
        # Getting the type of 'y' (line 357)
        y_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 43), 'y', False)
        # Obtaining the member 'index' of a type (line 357)
        index_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 43), y_514, 'index')
        # Processing the call keyword arguments (line 357)
        kwargs_516 = {}
        # Getting the type of 'cmp' (line 357)
        cmp_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 30), 'cmp', False)
        # Calling cmp(args, kwargs) (line 357)
        cmp_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 357, 30), cmp_511, *[index_513, index_515], **kwargs_516)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'stypy_return_type', cmp_call_result_517)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 357)
        stypy_return_type_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_518

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 357)
    _stypy_temp_lambda_1_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), '_stypy_temp_lambda_1')
    # Processing the call keyword arguments (line 357)
    kwargs_520 = {}
    # Getting the type of 'symbols' (line 357)
    symbols_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'symbols', False)
    # Obtaining the member 'sort' of a type (line 357)
    sort_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 4), symbols_509, 'sort')
    # Calling sort(args, kwargs) (line 357)
    sort_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 357, 4), sort_510, *[_stypy_temp_lambda_1_519], **kwargs_520)
    
    
    # Getting the type of 'symbols' (line 358)
    symbols_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 14), 'symbols')
    # Assigning a type to the variable 'symbols_522' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'symbols_522', symbols_522)
    # Testing if the for loop is going to be iterated (line 358)
    # Testing the type of a for loop iterable (line 358)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 358, 4), symbols_522)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 358, 4), symbols_522):
        # Getting the type of the for loop variable (line 358)
        for_loop_var_523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 358, 4), symbols_522)
        # Assigning a type to the variable 'co' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'co', for_loop_var_523)
        # SSA begins for a for statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to report(...): (line 359)
        # Processing the call keyword arguments (line 359)
        kwargs_526 = {}
        # Getting the type of 'co' (line 359)
        co_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'co', False)
        # Obtaining the member 'report' of a type (line 359)
        report_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), co_524, 'report')
        # Calling report(args, kwargs) (line 359)
        report_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), report_525, *[], **kwargs_526)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a List to a Name (line 361):
    
    # Assigning a List to a Name (line 361):
    
    # Obtaining an instance of the builtin type 'list' (line 361)
    list_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 361)
    # Adding element type (line 361)
    str_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 14), 'str', '000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 13), list_528, str_529)
    # Adding element type (line 361)
    str_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 20), 'str', '001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 13), list_528, str_530)
    # Adding element type (line 361)
    str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 26), 'str', '010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 13), list_528, str_531)
    # Adding element type (line 361)
    str_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 32), 'str', '011')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 13), list_528, str_532)
    # Adding element type (line 361)
    str_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 38), 'str', '100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 13), list_528, str_533)
    # Adding element type (line 361)
    str_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 44), 'str', '100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 13), list_528, str_534)
    # Adding element type (line 361)
    str_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 50), 'str', '000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 13), list_528, str_535)
    
    # Assigning a type to the variable 'source' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'source', list_528)
    
    # Assigning a Call to a Name (line 362):
    
    # Assigning a Call to a Name (line 362):
    
    # Call to encode(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of 'source' (line 362)
    source_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 20), 'source', False)
    # Getting the type of 'symbols' (line 362)
    symbols_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 28), 'symbols', False)
    # Processing the call keyword arguments (line 362)
    kwargs_539 = {}
    # Getting the type of 'encode' (line 362)
    encode_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 13), 'encode', False)
    # Calling encode(args, kwargs) (line 362)
    encode_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 362, 13), encode_536, *[source_537, symbols_538], **kwargs_539)
    
    # Assigning a type to the variable 'zipped' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'zipped', encode_call_result_540)
    
    # Assigning a Call to a Name (line 364):
    
    # Assigning a Call to a Name (line 364):
    
    # Call to decode(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'zipped' (line 364)
    zipped_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 21), 'zipped', False)
    # Getting the type of 'root' (line 364)
    root_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 29), 'root', False)
    # Processing the call keyword arguments (line 364)
    kwargs_544 = {}
    # Getting the type of 'decode' (line 364)
    decode_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 13), 'decode', False)
    # Calling decode(args, kwargs) (line 364)
    decode_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 364, 13), decode_541, *[zipped_542, root_543], **kwargs_544)
    
    # Assigning a type to the variable 'answer' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'answer', decode_call_result_545)
    
    # Getting the type of 'source' (line 366)
    source_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 9), 'source')
    # Getting the type of 'answer' (line 366)
    answer_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'answer')
    # Applying the binary operator '!=' (line 366)
    result_ne_548 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 9), '!=', source_546, answer_547)
    
    # Testing if the type of an if condition is none (line 366)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 366, 4), result_ne_548):
        pass
    else:
        
        # Testing the type of an if condition (line 366)
        if_condition_549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 4), result_ne_548)
        # Assigning a type to the variable 'if_condition_549' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'if_condition_549', if_condition_549)
        # SSA begins for if statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 366)
        module_type_store.open_ssa_branch('else')
        pass
        # SSA join for if statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        

    pass
    
    # ################# End of 'easytest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'easytest' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'easytest'
    return stypy_return_type_550

# Assigning a type to the variable 'easytest' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'easytest', easytest)

@norecursion
def test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test'
    module_type_store = module_type_store.open_function_context('test', 372, 0, False)
    
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

    
    # Call to easytest(...): (line 373)
    # Processing the call keyword arguments (line 373)
    kwargs_552 = {}
    # Getting the type of 'easytest' (line 373)
    easytest_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'easytest', False)
    # Calling easytest(args, kwargs) (line 373)
    easytest_call_result_553 = invoke(stypy.reporting.localization.Localization(__file__, 373, 4), easytest_551, *[], **kwargs_552)
    
    
    # Call to hardertest(...): (line 374)
    # Processing the call keyword arguments (line 374)
    kwargs_555 = {}
    # Getting the type of 'hardertest' (line 374)
    hardertest_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'hardertest', False)
    # Calling hardertest(args, kwargs) (line 374)
    hardertest_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 374, 4), hardertest_554, *[], **kwargs_555)
    
    
    # ################# End of 'test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test' in the type store
    # Getting the type of 'stypy_return_type' (line 372)
    stypy_return_type_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_557)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test'
    return stypy_return_type_557

# Assigning a type to the variable 'test' (line 372)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 0), 'test', test)

@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 376, 0, False)
    
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

    
    # Call to join(...): (line 377)
    # Processing the call arguments (line 377)
    
    # Call to dirname(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of '__file__' (line 377)
    file___564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 40), '__file__', False)
    # Processing the call keyword arguments (line 377)
    kwargs_565 = {}
    # Getting the type of 'os' (line 377)
    os_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 377)
    path_562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 24), os_561, 'path')
    # Obtaining the member 'dirname' of a type (line 377)
    dirname_563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 24), path_562, 'dirname')
    # Calling dirname(args, kwargs) (line 377)
    dirname_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 377, 24), dirname_563, *[file___564], **kwargs_565)
    
    # Getting the type of 'path' (line 377)
    path_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 51), 'path', False)
    # Processing the call keyword arguments (line 377)
    kwargs_568 = {}
    # Getting the type of 'os' (line 377)
    os_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 377)
    path_559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 11), os_558, 'path')
    # Obtaining the member 'join' of a type (line 377)
    join_560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 11), path_559, 'join')
    # Calling join(args, kwargs) (line 377)
    join_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 377, 11), join_560, *[dirname_call_result_566, path_567], **kwargs_568)
    
    # Assigning a type to the variable 'stypy_return_type' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type', join_call_result_569)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 376)
    stypy_return_type_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_570)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_570

# Assigning a type to the variable 'Relative' (line 376)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'Relative', Relative)

@norecursion
def hardertest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hardertest'
    module_type_store = module_type_store.open_function_context('hardertest', 379, 0, False)
    
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

    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to open(...): (line 381)
    # Processing the call arguments (line 381)
    
    # Call to Relative(...): (line 381)
    # Processing the call arguments (line 381)
    str_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 31), 'str', 'testdata/BentCoinFile')
    # Processing the call keyword arguments (line 381)
    kwargs_574 = {}
    # Getting the type of 'Relative' (line 381)
    Relative_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 381)
    Relative_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 381, 22), Relative_572, *[str_573], **kwargs_574)
    
    str_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 58), 'str', 'r')
    # Processing the call keyword arguments (line 381)
    kwargs_577 = {}
    # Getting the type of 'open' (line 381)
    open_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'open', False)
    # Calling open(args, kwargs) (line 381)
    open_call_result_578 = invoke(stypy.reporting.localization.Localization(__file__, 381, 16), open_571, *[Relative_call_result_575, str_576], **kwargs_577)
    
    # Assigning a type to the variable 'inputfile' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'inputfile', open_call_result_578)
    
    # Assigning a Call to a Name (line 382):
    
    # Assigning a Call to a Name (line 382):
    
    # Call to open(...): (line 382)
    # Processing the call arguments (line 382)
    
    # Call to Relative(...): (line 382)
    # Processing the call arguments (line 382)
    str_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 32), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 382)
    kwargs_582 = {}
    # Getting the type of 'Relative' (line 382)
    Relative_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'Relative', False)
    # Calling Relative(args, kwargs) (line 382)
    Relative_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 382, 23), Relative_580, *[str_581], **kwargs_582)
    
    str_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 45), 'str', 'w')
    # Processing the call keyword arguments (line 382)
    kwargs_585 = {}
    # Getting the type of 'open' (line 382)
    open_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), 'open', False)
    # Calling open(args, kwargs) (line 382)
    open_call_result_586 = invoke(stypy.reporting.localization.Localization(__file__, 382, 17), open_579, *[Relative_call_result_583, str_584], **kwargs_585)
    
    # Assigning a type to the variable 'outputfile' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'outputfile', open_call_result_586)
    
    # Call to compress_it(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'inputfile' (line 384)
    inputfile_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'inputfile', False)
    # Getting the type of 'outputfile' (line 384)
    outputfile_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'outputfile', False)
    # Processing the call keyword arguments (line 384)
    kwargs_590 = {}
    # Getting the type of 'compress_it' (line 384)
    compress_it_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'compress_it', False)
    # Calling compress_it(args, kwargs) (line 384)
    compress_it_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 384, 4), compress_it_587, *[inputfile_588, outputfile_589], **kwargs_590)
    
    
    # Call to close(...): (line 385)
    # Processing the call keyword arguments (line 385)
    kwargs_594 = {}
    # Getting the type of 'outputfile' (line 385)
    outputfile_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 385)
    close_593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 4), outputfile_592, 'close')
    # Calling close(args, kwargs) (line 385)
    close_call_result_595 = invoke(stypy.reporting.localization.Localization(__file__, 385, 4), close_593, *[], **kwargs_594)
    
    
    # Call to close(...): (line 385)
    # Processing the call keyword arguments (line 385)
    kwargs_598 = {}
    # Getting the type of 'inputfile' (line 385)
    inputfile_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 385)
    close_597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 28), inputfile_596, 'close')
    # Calling close(args, kwargs) (line 385)
    close_call_result_599 = invoke(stypy.reporting.localization.Localization(__file__, 385, 28), close_597, *[], **kwargs_598)
    
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to open(...): (line 388)
    # Processing the call arguments (line 388)
    
    # Call to Relative(...): (line 388)
    # Processing the call arguments (line 388)
    str_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 31), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 388)
    kwargs_603 = {}
    # Getting the type of 'Relative' (line 388)
    Relative_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 388)
    Relative_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 388, 22), Relative_601, *[str_602], **kwargs_603)
    
    str_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 44), 'str', 'r')
    # Processing the call keyword arguments (line 388)
    kwargs_606 = {}
    # Getting the type of 'open' (line 388)
    open_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'open', False)
    # Calling open(args, kwargs) (line 388)
    open_call_result_607 = invoke(stypy.reporting.localization.Localization(__file__, 388, 16), open_600, *[Relative_call_result_604, str_605], **kwargs_606)
    
    # Assigning a type to the variable 'inputfile' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'inputfile', open_call_result_607)
    
    # Assigning a Call to a Name (line 389):
    
    # Assigning a Call to a Name (line 389):
    
    # Call to open(...): (line 389)
    # Processing the call arguments (line 389)
    
    # Call to Relative(...): (line 389)
    # Processing the call arguments (line 389)
    str_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 32), 'str', 'tmp2')
    # Processing the call keyword arguments (line 389)
    kwargs_611 = {}
    # Getting the type of 'Relative' (line 389)
    Relative_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 23), 'Relative', False)
    # Calling Relative(args, kwargs) (line 389)
    Relative_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 389, 23), Relative_609, *[str_610], **kwargs_611)
    
    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 42), 'str', 'w')
    # Processing the call keyword arguments (line 389)
    kwargs_614 = {}
    # Getting the type of 'open' (line 389)
    open_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'open', False)
    # Calling open(args, kwargs) (line 389)
    open_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 389, 17), open_608, *[Relative_call_result_612, str_613], **kwargs_614)
    
    # Assigning a type to the variable 'outputfile' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'outputfile', open_call_result_615)
    
    # Call to uncompress_it(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'inputfile' (line 391)
    inputfile_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 18), 'inputfile', False)
    # Getting the type of 'outputfile' (line 391)
    outputfile_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'outputfile', False)
    # Processing the call keyword arguments (line 391)
    kwargs_619 = {}
    # Getting the type of 'uncompress_it' (line 391)
    uncompress_it_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'uncompress_it', False)
    # Calling uncompress_it(args, kwargs) (line 391)
    uncompress_it_call_result_620 = invoke(stypy.reporting.localization.Localization(__file__, 391, 4), uncompress_it_616, *[inputfile_617, outputfile_618], **kwargs_619)
    
    
    # Call to close(...): (line 392)
    # Processing the call keyword arguments (line 392)
    kwargs_623 = {}
    # Getting the type of 'outputfile' (line 392)
    outputfile_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 392)
    close_622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 4), outputfile_621, 'close')
    # Calling close(args, kwargs) (line 392)
    close_call_result_624 = invoke(stypy.reporting.localization.Localization(__file__, 392, 4), close_622, *[], **kwargs_623)
    
    
    # Call to close(...): (line 392)
    # Processing the call keyword arguments (line 392)
    kwargs_627 = {}
    # Getting the type of 'inputfile' (line 392)
    inputfile_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 28), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 392)
    close_626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 28), inputfile_625, 'close')
    # Calling close(args, kwargs) (line 392)
    close_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 392, 28), close_626, *[], **kwargs_627)
    
    pass
    
    # ################# End of 'hardertest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hardertest' in the type store
    # Getting the type of 'stypy_return_type' (line 379)
    stypy_return_type_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_629)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hardertest'
    return stypy_return_type_629

# Assigning a type to the variable 'hardertest' (line 379)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 0), 'hardertest', hardertest)

# Assigning a Num to a Name (line 400):

# Assigning a Num to a Name (line 400):
float_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 2), 'float')
# Assigning a type to the variable 'f' (line 400)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'f', float_630)

# Assigning a Num to a Name (line 400):

# Assigning a Num to a Name (line 400):
int_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 10), 'int')
# Assigning a type to the variable 'N' (line 400)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'N', int_631)

# Assigning a Num to a Name (line 401):

# Assigning a Num to a Name (line 401):
float_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 2), 'float')
# Assigning a type to the variable 'f' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'f', float_632)

# Assigning a Num to a Name (line 401):

# Assigning a Num to a Name (line 401):
int_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 10), 'int')
# Assigning a type to the variable 'N' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'N', int_633)

# Assigning a Num to a Name (line 402):

# Assigning a Num to a Name (line 402):
float_634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 2), 'float')
# Assigning a type to the variable 'f' (line 402)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'f', float_634)

# Assigning a Num to a Name (line 402):

# Assigning a Num to a Name (line 402):
int_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 10), 'int')
# Assigning a type to the variable 'N' (line 402)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'N', int_635)

@norecursion
def compress_it(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compress_it'
    module_type_store = module_type_store.open_function_context('compress_it', 404, 0, False)
    
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

    str_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, (-1)), 'str', '\n    Make Huffman code for blocks, and\n    Compress from file (possibly stdin).\n    ')
    
    # Assigning a Call to a Name (line 409):
    
    # Assigning a Call to a Name (line 409):
    
    # Call to findprobs(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'f' (line 409)
    f_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'f', False)
    # Getting the type of 'N' (line 409)
    N_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'N', False)
    # Processing the call keyword arguments (line 409)
    kwargs_640 = {}
    # Getting the type of 'findprobs' (line 409)
    findprobs_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'findprobs', False)
    # Calling findprobs(args, kwargs) (line 409)
    findprobs_call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 409, 12), findprobs_637, *[f_638, N_639], **kwargs_640)
    
    # Assigning a type to the variable 'probs' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'probs', findprobs_call_result_641)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to makenodes(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'probs' (line 410)
    probs_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 24), 'probs', False)
    # Processing the call keyword arguments (line 410)
    kwargs_644 = {}
    # Getting the type of 'makenodes' (line 410)
    makenodes_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'makenodes', False)
    # Calling makenodes(args, kwargs) (line 410)
    makenodes_call_result_645 = invoke(stypy.reporting.localization.Localization(__file__, 410, 14), makenodes_642, *[probs_643], **kwargs_644)
    
    # Assigning a type to the variable 'symbols' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'symbols', makenodes_call_result_645)
    
    # Assigning a Call to a Name (line 413):
    
    # Assigning a Call to a Name (line 413):
    
    # Call to iterate(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'symbols' (line 413)
    symbols_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 'symbols', False)
    # Processing the call keyword arguments (line 413)
    kwargs_648 = {}
    # Getting the type of 'iterate' (line 413)
    iterate_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'iterate', False)
    # Calling iterate(args, kwargs) (line 413)
    iterate_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 413, 11), iterate_646, *[symbols_647], **kwargs_648)
    
    # Assigning a type to the variable 'root' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'root', iterate_call_result_649)
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to read(...): (line 415)
    # Processing the call keyword arguments (line 415)
    kwargs_652 = {}
    # Getting the type of 'inputfile' (line 415)
    inputfile_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 415)
    read_651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 13), inputfile_650, 'read')
    # Calling read(args, kwargs) (line 415)
    read_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 415, 13), read_651, *[], **kwargs_652)
    
    # Assigning a type to the variable 'string' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'string', read_call_result_653)
    
    # Call to write(...): (line 416)
    # Processing the call arguments (line 416)
    
    # Call to Bencode(...): (line 416)
    # Processing the call arguments (line 416)
    # Getting the type of 'string' (line 416)
    string_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'string', False)
    # Getting the type of 'symbols' (line 416)
    symbols_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 38), 'symbols', False)
    # Getting the type of 'N' (line 416)
    N_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 47), 'N', False)
    # Processing the call keyword arguments (line 416)
    kwargs_660 = {}
    # Getting the type of 'Bencode' (line 416)
    Bencode_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'Bencode', False)
    # Calling Bencode(args, kwargs) (line 416)
    Bencode_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 416, 22), Bencode_656, *[string_657, symbols_658, N_659], **kwargs_660)
    
    # Processing the call keyword arguments (line 416)
    kwargs_662 = {}
    # Getting the type of 'outputfile' (line 416)
    outputfile_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 416)
    write_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 4), outputfile_654, 'write')
    # Calling write(args, kwargs) (line 416)
    write_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 416, 4), write_655, *[Bencode_call_result_661], **kwargs_662)
    
    pass
    
    # ################# End of 'compress_it(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compress_it' in the type store
    # Getting the type of 'stypy_return_type' (line 404)
    stypy_return_type_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_664)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compress_it'
    return stypy_return_type_664

# Assigning a type to the variable 'compress_it' (line 404)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'compress_it', compress_it)

@norecursion
def uncompress_it(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'uncompress_it'
    module_type_store = module_type_store.open_function_context('uncompress_it', 419, 0, False)
    
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

    str_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, (-1)), 'str', '\n    Make Huffman code for blocks, and\n    UNCompress from file (possibly stdin).\n    ')
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to findprobs(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'f' (line 424)
    f_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 22), 'f', False)
    # Getting the type of 'N' (line 424)
    N_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'N', False)
    # Processing the call keyword arguments (line 424)
    kwargs_669 = {}
    # Getting the type of 'findprobs' (line 424)
    findprobs_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'findprobs', False)
    # Calling findprobs(args, kwargs) (line 424)
    findprobs_call_result_670 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), findprobs_666, *[f_667, N_668], **kwargs_669)
    
    # Assigning a type to the variable 'probs' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'probs', findprobs_call_result_670)
    
    # Assigning a Call to a Name (line 427):
    
    # Assigning a Call to a Name (line 427):
    
    # Call to makenodes(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'probs' (line 427)
    probs_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), 'probs', False)
    # Processing the call keyword arguments (line 427)
    kwargs_673 = {}
    # Getting the type of 'makenodes' (line 427)
    makenodes_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 14), 'makenodes', False)
    # Calling makenodes(args, kwargs) (line 427)
    makenodes_call_result_674 = invoke(stypy.reporting.localization.Localization(__file__, 427, 14), makenodes_671, *[probs_672], **kwargs_673)
    
    # Assigning a type to the variable 'symbols' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'symbols', makenodes_call_result_674)
    
    # Assigning a Call to a Name (line 428):
    
    # Assigning a Call to a Name (line 428):
    
    # Call to iterate(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'symbols' (line 428)
    symbols_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 'symbols', False)
    # Processing the call keyword arguments (line 428)
    kwargs_677 = {}
    # Getting the type of 'iterate' (line 428)
    iterate_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'iterate', False)
    # Calling iterate(args, kwargs) (line 428)
    iterate_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 428, 11), iterate_675, *[symbols_676], **kwargs_677)
    
    # Assigning a type to the variable 'root' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'root', iterate_call_result_678)
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to read(...): (line 430)
    # Processing the call keyword arguments (line 430)
    kwargs_681 = {}
    # Getting the type of 'inputfile' (line 430)
    inputfile_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 430)
    read_680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 13), inputfile_679, 'read')
    # Calling read(args, kwargs) (line 430)
    read_call_result_682 = invoke(stypy.reporting.localization.Localization(__file__, 430, 13), read_680, *[], **kwargs_681)
    
    # Assigning a type to the variable 'string' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'string', read_call_result_682)
    
    # Call to write(...): (line 431)
    # Processing the call arguments (line 431)
    
    # Call to Bdecode(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'string' (line 431)
    string_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 30), 'string', False)
    # Getting the type of 'root' (line 431)
    root_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 38), 'root', False)
    # Getting the type of 'N' (line 431)
    N_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 44), 'N', False)
    # Processing the call keyword arguments (line 431)
    kwargs_689 = {}
    # Getting the type of 'Bdecode' (line 431)
    Bdecode_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 22), 'Bdecode', False)
    # Calling Bdecode(args, kwargs) (line 431)
    Bdecode_call_result_690 = invoke(stypy.reporting.localization.Localization(__file__, 431, 22), Bdecode_685, *[string_686, root_687, N_688], **kwargs_689)
    
    # Processing the call keyword arguments (line 431)
    kwargs_691 = {}
    # Getting the type of 'outputfile' (line 431)
    outputfile_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 431)
    write_684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 4), outputfile_683, 'write')
    # Calling write(args, kwargs) (line 431)
    write_call_result_692 = invoke(stypy.reporting.localization.Localization(__file__, 431, 4), write_684, *[Bdecode_call_result_690], **kwargs_691)
    
    pass
    
    # ################# End of 'uncompress_it(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'uncompress_it' in the type store
    # Getting the type of 'stypy_return_type' (line 419)
    stypy_return_type_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'uncompress_it'
    return stypy_return_type_693

# Assigning a type to the variable 'uncompress_it' (line 419)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 0), 'uncompress_it', uncompress_it)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 434, 0, False)
    
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

    
    # Call to setrecursionlimit(...): (line 435)
    # Processing the call arguments (line 435)
    int_696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'int')
    # Processing the call keyword arguments (line 435)
    kwargs_697 = {}
    # Getting the type of 'sys' (line 435)
    sys_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'sys', False)
    # Obtaining the member 'setrecursionlimit' of a type (line 435)
    setrecursionlimit_695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 4), sys_694, 'setrecursionlimit')
    # Calling setrecursionlimit(args, kwargs) (line 435)
    setrecursionlimit_call_result_698 = invoke(stypy.reporting.localization.Localization(__file__, 435, 4), setrecursionlimit_695, *[int_696], **kwargs_697)
    
    
    # Call to test(...): (line 436)
    # Processing the call keyword arguments (line 436)
    kwargs_700 = {}
    # Getting the type of 'test' (line 436)
    test_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'test', False)
    # Calling test(args, kwargs) (line 436)
    test_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 436, 4), test_699, *[], **kwargs_700)
    
    # Getting the type of 'True' (line 437)
    True_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'stypy_return_type', True_702)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 434)
    stypy_return_type_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_703)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_703

# Assigning a type to the variable 'run' (line 434)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'run', run)

# Call to run(...): (line 439)
# Processing the call keyword arguments (line 439)
kwargs_705 = {}
# Getting the type of 'run' (line 439)
run_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'run', False)
# Calling run(args, kwargs) (line 439)
run_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 439, 0), run_704, *[], **kwargs_705)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: Lempel-Ziv code (2) for compression
4: 
5: This version of Lempel-Ziv looks back in the sequence read so far
6: for a match to the incoming substring; then sends a pointer.
7: 
8: It is a self-delimiting code that sends a special pointer indicating
9: end of file.
10: 
11: http://www.aims.ac.za/~mackay/itila/
12: 
13:   LZ2.py is free software (c) David MacKay December 2005. License: GPL
14: '''
15: ## For license statement see  http://www.gnu.org/copyleft/gpl.html
16: 
17: import sys, os
18: 
19: 
20: def Relative(path):
21:     return os.path.join(os.path.dirname(__file__), path)
22: 
23: 
24: def dec_to_bin(n, digits):
25:     ''' n is the number to convert to binary;  digits is the number of bits you want
26:     Always prints full number of digits
27:     >>> print dec_to_bin( 17 , 9)
28:     000010001
29:     >>> print dec_to_bin( 17 , 5)
30:     10001
31: 
32:     Will behead the standard binary number if requested
33:     >>> print dec_to_bin( 17 , 4)
34:     0001
35:     '''
36:     if (n < 0):
37:         sys.stderr.write("warning, negative n not expected\n")
38:     i = digits - 1
39:     ans = ""
40:     while i >= 0:
41:         b = (((1 << i) & n) > 0)
42:         i -= 1
43:         ans = ans + str(int(b))
44:     return ans
45: 
46: 
47: def bin_to_dec(clist, c, tot=0):
48:     '''Implements ordinary binary to integer conversion if tot=0
49:     and HEADLESS binary to integer if tot=1
50:     clist is a list of bits; read c of them and turn into an integer.
51:     The bits that are read from the list are popped from it, i.e., deleted
52: 
53:     Regular binary to decimal 1001 is 9...
54:     >>> bin_to_dec(  ['1', '0', '0', '1']  ,  4  , 0 )
55:     9
56: 
57:     Headless binary to decimal [1] 1001 is 25...
58:     >>> bin_to_dec(  ['1', '0', '0', '1']  ,  4  , 1 )
59:     25
60:     '''
61:     while (c > 0):
62:         assert (len(clist) > 0)  ## else we have been fed insufficient bits.
63:         tot = tot * 2 + int(clist.pop(0))
64:         c -= 1
65:         pass
66:     return tot
67: 
68: 
69: def ceillog(n):  ## ceil( log_2 ( n ))   [Used by LZ.py]
70:     '''
71:     >>> print ceillog(3), ceillog(4), ceillog(5)
72:     2 2 3
73:     '''
74:     assert n >= 1
75:     c = 0
76:     while 2 ** c < n:
77:         c += 1
78:     return c
79: 
80: 
81: def status(fr, to, oldpointer, digits, digits2, length, c, maxlength):
82:     '''
83:     Print out the current pointer location, the current matching
84:     string, and the current locations of "from" and "to".
85:     Also report the range of conceivable values for the pointer and
86:     length quantities.
87:     '''
88:     pass
89:     # print "fr,to = %d,%d; oldpointer=%d; d=%d, d2=%d, l=%d" % \
90:     #      (fr,to,oldpointer, digits,digits2, length)
91: 
92: 
93: ##    print "|%s\n|%s%s\t(pointer at %d/0..%d)\n|%s%s\t(maximal match of length %d/0..%d)\n|%s%s\n|%s%s" %\
94: ##          (c,\
95: ##           '.'*oldpointer,'p', oldpointer, fr-1, \
96: ##           ' '*oldpointer, '-'*length, length, maxlength,\
97: ##           ' '*fr, 'f',\
98: ##           ' '*to, 't')
99: 
100: def searchstatus(fr, to, L, c):
101:     '''
102:     Show the current string (fr,to) that is being searched for.
103:     '''
104:     pass
105: 
106: 
107: ##    print "L=%d, fr=%d, to=%d" % (L,fr, to)
108: ##    print "|%s\n|%s%s\n|%s%s" %\
109: ##          (c,\
110: ##           ' '*fr, 'f',\
111: ##           ' '*to, 't')
112: ##    # find where this substring occurs
113: ##    print "looking for '%s' inside '%s'. " % (c[fr:to],c[0:fr]) ,
114: 
115: def encode(c, pretty=1, verbose=0):  ## c is STRING of characters (0/1) ; p is whether to print out prettily
116:     '''
117:     Encode using Lempel-Ziv (2), which sends pointers and lengths
118:     Pretty printing
119:     >>> print encode("000000000000100000000000",1)
120:     0(0,1)(00,10)(000,100)(0000,0100)(1101,0)(0000,1011)
121: 
122:     Normal printing
123:     >>> print encode("000000000000100000000000",0)
124:     0010010000100000001001101000001011
125: 
126:     To understand the encoding procedure it might be
127:     best to read the decoder.
128: 
129:     Termination rule:
130:     We have a special reserved "impossible pointer" string
131:     and always have space for one extra one in the pointer area, to allow us to
132:     send termination information, including whether the last bit needed removing or not.
133:     If the last thing that happened was a perfect match without mismatch event,
134:     which is the most common event, do nothing, send EOF. The decoder will sees the special
135:     '0' bit after the special pointer and so does not include the final character.
136:     If instead there happened to be a mismatch event at the exact moment
137:     when the final char arrived, we do a standard decoding action.
138: 
139:     ::-----SEQUENCE SO FAR-------::|--SEQUENCE STILL to be sent--
140:          ^pointer                   ^fr         ^to
141:          ------------               ------------
142:          <- length ->               <- length ->
143:     Once we have found the maximum length (to-fr) that matches, send
144:          the values of pointer and length.
145: 
146:     '''
147:     output = []
148:     L = len(c)
149:     assert L > 1  ## sorry, cannot compress the files "0" or "1" or ""
150:     output.append(c[0])  # to get started we must send one bit
151:     fr = 1;
152:     eof_sent = 0
153:     while (eof_sent == 0):  # start a new substring search
154:         to = fr  # Always Start by finding a match of the empty string
155:         oldpointer = -2  # indicates that no match has been found. Used for debugging.
156:         while (eof_sent == 0) and (to <= L):  # extend the search
157:             if verbose > 2:  searchstatus(fr, to, L, c);  pass
158:             pointer = c[0:fr].find(c[fr:to])
159:             if verbose > 2: pass  # print "result:",pointer , to ; pass
160:             if (pointer == -1) or (to >= L):
161:                 if (pointer != -1): oldpointer = pointer;  pass
162:                 digits = ceillog(
163:                     fr + 1)  # digits=ceillog ( fr ) would be enough space for oldpointer, which is in range (0,fr-1).
164:                 # we give ourselves extra space so as to be able to convey a termination event
165:                 maxlength = fr - oldpointer  # from-oldpointer is maximum possible sequence length
166:                 digits2 = ceillog(maxlength + 1)
167:                 if (pointer == -1):    to -= 1; pass  # the matched string was shorter than to-fr; need to step back.
168:                 length = to - fr
169:                 if length < maxlength:  # then the receiver can deduce the next bit
170:                     to += 1;
171:                     pass
172:                 if (to >= L):  # Special termination message precedes the last (pointer,length) message.
173:                     if (pointer != -1):
174:                         specialbit = 0; pass
175:                     else:
176:                         specialbit = 1; pass
177:                     output.append(printout(dec_to_bin(fr, digits),
178:                                            str(specialbit), pretty))
179:                     eof_sent = 1
180:                     pass
181:                 assert length <= maxlength
182:                 output.append(printout(dec_to_bin(oldpointer, digits),
183:                                        dec_to_bin(length, digits2), pretty))
184:                 if verbose:
185:                     status(fr, to, oldpointer, digits, digits2, length, c, maxlength)
186:                     # print "".join(output)
187:                     pass
188:                 oldpointer = -2
189:                 fr = to
190:                 break
191:             else:
192:                 to += 1;
193:                 oldpointer = pointer;
194:                 pass
195:             pass
196:         pass
197:     if verbose: pass  # print "DONE Encoding"
198:     return "".join(output)
199: 
200: 
201: def printout(pointerstring, lengthstring, pretty=1):
202:     if pretty:
203:         return "(" + pointerstring + "," + lengthstring + ")"
204:     else:
205:         return pointerstring + lengthstring
206: 
207: 
208: def decode(li, verbose=0):
209:     '''
210:     >>> print decode(list("0010010000100000001001101000001010"))
211:     00000000000010000000000
212:     '''
213:     assert (len(li) > 0)  # need to get first bit! The compressor cannot compress the empty string.
214:     c = li.pop(0)
215:     fr = 1;
216:     to = fr
217: 
218:     not_eof = 1;
219:     specialbit = 0
220:     while not_eof:
221:         assert (len(li) > 0)  # self-delimiting file
222:         digits = ceillog(fr + 1)
223:         pointer = bin_to_dec(li, digits)  # get the pointer
224:         maxlength = fr - pointer
225:         if pointer == fr:  # special end of file signal!
226:             specialbit = int(li.pop(0))
227:             pointer = bin_to_dec(li, digits)
228:             maxlength = fr - pointer
229:             not_eof = 0
230:             pass
231:         digits2 = ceillog(maxlength + 1)
232:         length = bin_to_dec(li, digits2)
233:         addition = c[pointer:pointer + length];
234:         assert len(addition) == length
235:         if ((not_eof == 0) and (specialbit == 1)) or (not_eof and (length < maxlength)):
236:             opposite = str(1 - int(c[pointer + length]))
237:         else:
238:             opposite = ''
239:         c = c + addition + opposite
240:         if verbose:
241:             to = length + fr + 1
242:             status(fr, to, pointer, digits, digits2, length, c, maxlength)
243:             pass
244:         fr = len(c)
245:     return c
246: 
247: 
248: def test():
249:     # print "pretty encoding examples:"
250:     examples = ["0010000000001000000000001", "00000000000010000000000"]
251:     examples2 = ["1010101010101010101010101010101010101010", \
252:                  "011", \
253:                  "01", "10", "11", "00", "000", "001", "010", "011", "100", "101", "110", \
254:                  "1010100000000000000000000101010101010101000000000000101010101010101010", \
255:                  "10101010101010101010101010101010101010101", \
256:                  "00000", "000000", "0000000", "00000000", "000000000", "0000000000", \
257:                  "00001", "000001", "0000001", "00000001", "000000001", "0000000001", \
258:                  "0000", "0001", "0010", "0011", "0100", "0101", "0110", \
259:                  "0111", "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111", \
260:                  "111", "110010010101000000000001110100100100000000000000", \
261:                  "00000000000010000000000", "1100100", "100100"]
262:     pretty = 1;
263:     verbose = 1
264:     for ex in examples:
265:         ##        print
266:         ##        print "Encoding", ex
267:         zip = encode(ex, pretty, verbose)
268:         if verbose > 2: print zip
269:         zip2 = encode(ex, 0, 0)
270:         ##        print "Decoding", zip2
271:         unc = decode(list(zip2), verbose)
272:         ##        print "-> ", unc
273:         if unc == ex:
274:             ##            print "OK!"
275:             pass
276:         else:
277:             ##            print "ERROR!!!!!!!!!!!!!!!!!!!!!!!!!"
278:             assert False
279: 
280:     if (0):
281:         pretty = 1;
282:         verbose = 1
283:         for ex in examples2:
284:             zip = encode(ex, pretty, verbose)
285:             ##            print zip
286:             zip2 = encode(ex, 0, 0)
287:             ##            print "Decoding", zip2
288:             unc = decode(list(zip2), verbose)
289:             ##            print "-> ", unc
290:             if unc == ex:
291:                 pass  # print "OK!"
292:             else:
293:                 ##                print "ERROR!!!!!!!!!!!!!!!!!!!!!!!!!"
294:                 assert False
295:         ##        print "decoding examples:"
296:         examples = ["0010010000100000001001101000001001"]
297:         for ex in examples:
298:             ##            print ex, decode( list(ex) , verbose )
299:             decode(list(ex), verbose)
300: 
301: 
302: def hardertest():
303:     ##    print "Reading the BentCoinFile"
304:     inputfile = open(Relative("testdata/BentCoinFile"), "r")
305:     outputfile = open(Relative("tmp.zip"), "w")
306:     ##    print  "Compressing to tmp.zip"
307: 
308:     zip = encode(inputfile.read(), 0, 0)
309:     outputfile.write(zip)
310:     outputfile.close();
311:     inputfile.close()
312:     ##    print "DONE compressing"
313: 
314:     inputfile = open(Relative("tmp.zip"), "r")
315:     outputfile = open(Relative("tmp2"), "w")
316:     ##    print  "Uncompressing to tmp2"
317:     unc = decode(list(inputfile.read()), 0)
318:     outputfile.write(unc)
319:     outputfile.close();
320:     inputfile.close()
321: 
322: 
323: ##    print "DONE uncompressing"
324: 
325: ##    print "Checking for differences..."
326: ##    os.system( "diff testdata/BentCoinFile tmp2" )
327: ##    os.system( "wc tmp.zip testdata/BentCoinFile tmp2" )
328: 
329: def run():
330:     test()
331:     hardertest()
332:     return True
333: 
334: 
335: run()
336: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\nLempel-Ziv code (2) for compression\n\nThis version of Lempel-Ziv looks back in the sequence read so far\nfor a match to the incoming substring; then sends a pointer.\n\nIt is a self-delimiting code that sends a special pointer indicating\nend of file.\n\nhttp://www.aims.ac.za/~mackay/itila/\n\n  LZ2.py is free software (c) David MacKay December 2005. License: GPL\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# Multiple import statement. import sys (1/2) (line 17)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/2) (line 17)
import os

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 20, 0, False)
    
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

    
    # Call to join(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to dirname(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of '__file__' (line 21)
    file___8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 40), '__file__', False)
    # Processing the call keyword arguments (line 21)
    kwargs_9 = {}
    # Getting the type of 'os' (line 21)
    os_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 21)
    path_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 24), os_5, 'path')
    # Obtaining the member 'dirname' of a type (line 21)
    dirname_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 24), path_6, 'dirname')
    # Calling dirname(args, kwargs) (line 21)
    dirname_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 21, 24), dirname_7, *[file___8], **kwargs_9)
    
    # Getting the type of 'path' (line 21)
    path_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 51), 'path', False)
    # Processing the call keyword arguments (line 21)
    kwargs_12 = {}
    # Getting the type of 'os' (line 21)
    os_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 21)
    path_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), os_2, 'path')
    # Obtaining the member 'join' of a type (line 21)
    join_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), path_3, 'join')
    # Calling join(args, kwargs) (line 21)
    join_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), join_4, *[dirname_call_result_10, path_11], **kwargs_12)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', join_call_result_13)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_14

# Assigning a type to the variable 'Relative' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'Relative', Relative)

@norecursion
def dec_to_bin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dec_to_bin'
    module_type_store = module_type_store.open_function_context('dec_to_bin', 24, 0, False)
    
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

    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', ' n is the number to convert to binary;  digits is the number of bits you want\n    Always prints full number of digits\n    >>> print dec_to_bin( 17 , 9)\n    000010001\n    >>> print dec_to_bin( 17 , 5)\n    10001\n\n    Will behead the standard binary number if requested\n    >>> print dec_to_bin( 17 , 4)\n    0001\n    ')
    
    # Getting the type of 'n' (line 36)
    n_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'n')
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 12), 'int')
    # Applying the binary operator '<' (line 36)
    result_lt_18 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 8), '<', n_16, int_17)
    
    # Testing if the type of an if condition is none (line 36)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 4), result_lt_18):
        pass
    else:
        
        # Testing the type of an if condition (line 36)
        if_condition_19 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 4), result_lt_18)
        # Assigning a type to the variable 'if_condition_19' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'if_condition_19', if_condition_19)
        # SSA begins for if statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 37)
        # Processing the call arguments (line 37)
        str_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'str', 'warning, negative n not expected\n')
        # Processing the call keyword arguments (line 37)
        kwargs_24 = {}
        # Getting the type of 'sys' (line 37)
        sys_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 37)
        stderr_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), sys_20, 'stderr')
        # Obtaining the member 'write' of a type (line 37)
        write_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), stderr_21, 'write')
        # Calling write(args, kwargs) (line 37)
        write_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), write_22, *[str_23], **kwargs_24)
        
        # SSA join for if statement (line 36)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 38):
    # Getting the type of 'digits' (line 38)
    digits_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'digits')
    int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
    # Applying the binary operator '-' (line 38)
    result_sub_28 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 8), '-', digits_26, int_27)
    
    # Assigning a type to the variable 'i' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'i', result_sub_28)
    
    # Assigning a Str to a Name (line 39):
    str_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 10), 'str', '')
    # Assigning a type to the variable 'ans' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'ans', str_29)
    
    
    # Getting the type of 'i' (line 40)
    i_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'i')
    int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'int')
    # Applying the binary operator '>=' (line 40)
    result_ge_32 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 10), '>=', i_30, int_31)
    
    # Testing if the while is going to be iterated (line 40)
    # Testing the type of an if condition (line 40)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 4), result_ge_32)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 4), result_ge_32):
        # SSA begins for while statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Compare to a Name (line 41):
        
        int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'int')
        # Getting the type of 'i' (line 41)
        i_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'i')
        # Applying the binary operator '<<' (line 41)
        result_lshift_35 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '<<', int_33, i_34)
        
        # Getting the type of 'n' (line 41)
        n_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'n')
        # Applying the binary operator '&' (line 41)
        result_and__37 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 14), '&', result_lshift_35, n_36)
        
        int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
        # Applying the binary operator '>' (line 41)
        result_gt_39 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 13), '>', result_and__37, int_38)
        
        # Assigning a type to the variable 'b' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'b', result_gt_39)
        
        # Getting the type of 'i' (line 42)
        i_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'i')
        int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'int')
        # Applying the binary operator '-=' (line 42)
        result_isub_42 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 8), '-=', i_40, int_41)
        # Assigning a type to the variable 'i' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'i', result_isub_42)
        
        
        # Assigning a BinOp to a Name (line 43):
        # Getting the type of 'ans' (line 43)
        ans_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'ans')
        
        # Call to str(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to int(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'b' (line 43)
        b_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'b', False)
        # Processing the call keyword arguments (line 43)
        kwargs_47 = {}
        # Getting the type of 'int' (line 43)
        int_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'int', False)
        # Calling int(args, kwargs) (line 43)
        int_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), int_45, *[b_46], **kwargs_47)
        
        # Processing the call keyword arguments (line 43)
        kwargs_49 = {}
        # Getting the type of 'str' (line 43)
        str_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'str', False)
        # Calling str(args, kwargs) (line 43)
        str_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), str_44, *[int_call_result_48], **kwargs_49)
        
        # Applying the binary operator '+' (line 43)
        result_add_51 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 14), '+', ans_43, str_call_result_50)
        
        # Assigning a type to the variable 'ans' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'ans', result_add_51)
        # SSA join for while statement (line 40)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'ans' (line 44)
    ans_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'ans')
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', ans_52)
    
    # ################# End of 'dec_to_bin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dec_to_bin' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_53)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dec_to_bin'
    return stypy_return_type_53

# Assigning a type to the variable 'dec_to_bin' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'dec_to_bin', dec_to_bin)

@norecursion
def bin_to_dec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'int')
    defaults = [int_54]
    # Create a new context for function 'bin_to_dec'
    module_type_store = module_type_store.open_function_context('bin_to_dec', 47, 0, False)
    
    # Passed parameters checking function
    bin_to_dec.stypy_localization = localization
    bin_to_dec.stypy_type_of_self = None
    bin_to_dec.stypy_type_store = module_type_store
    bin_to_dec.stypy_function_name = 'bin_to_dec'
    bin_to_dec.stypy_param_names_list = ['clist', 'c', 'tot']
    bin_to_dec.stypy_varargs_param_name = None
    bin_to_dec.stypy_kwargs_param_name = None
    bin_to_dec.stypy_call_defaults = defaults
    bin_to_dec.stypy_call_varargs = varargs
    bin_to_dec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bin_to_dec', ['clist', 'c', 'tot'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bin_to_dec', localization, ['clist', 'c', 'tot'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bin_to_dec(...)' code ##################

    str_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', "Implements ordinary binary to integer conversion if tot=0\n    and HEADLESS binary to integer if tot=1\n    clist is a list of bits; read c of them and turn into an integer.\n    The bits that are read from the list are popped from it, i.e., deleted\n\n    Regular binary to decimal 1001 is 9...\n    >>> bin_to_dec(  ['1', '0', '0', '1']  ,  4  , 0 )\n    9\n\n    Headless binary to decimal [1] 1001 is 25...\n    >>> bin_to_dec(  ['1', '0', '0', '1']  ,  4  , 1 )\n    25\n    ")
    
    
    # Getting the type of 'c' (line 61)
    c_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'c')
    int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
    # Applying the binary operator '>' (line 61)
    result_gt_58 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), '>', c_56, int_57)
    
    # Testing if the while is going to be iterated (line 61)
    # Testing the type of an if condition (line 61)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), result_gt_58)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 4), result_gt_58):
        # SSA begins for while statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'clist' (line 62)
        clist_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'clist', False)
        # Processing the call keyword arguments (line 62)
        kwargs_61 = {}
        # Getting the type of 'len' (line 62)
        len_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'len', False)
        # Calling len(args, kwargs) (line 62)
        len_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), len_59, *[clist_60], **kwargs_61)
        
        int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'int')
        # Applying the binary operator '>' (line 62)
        result_gt_64 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 16), '>', len_call_result_62, int_63)
        
        
        # Assigning a BinOp to a Name (line 63):
        # Getting the type of 'tot' (line 63)
        tot_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'tot')
        int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'int')
        # Applying the binary operator '*' (line 63)
        result_mul_67 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 14), '*', tot_65, int_66)
        
        
        # Call to int(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to pop(...): (line 63)
        # Processing the call arguments (line 63)
        int_71 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_72 = {}
        # Getting the type of 'clist' (line 63)
        clist_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'clist', False)
        # Obtaining the member 'pop' of a type (line 63)
        pop_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 28), clist_69, 'pop')
        # Calling pop(args, kwargs) (line 63)
        pop_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 63, 28), pop_70, *[int_71], **kwargs_72)
        
        # Processing the call keyword arguments (line 63)
        kwargs_74 = {}
        # Getting the type of 'int' (line 63)
        int_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'int', False)
        # Calling int(args, kwargs) (line 63)
        int_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), int_68, *[pop_call_result_73], **kwargs_74)
        
        # Applying the binary operator '+' (line 63)
        result_add_76 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 14), '+', result_mul_67, int_call_result_75)
        
        # Assigning a type to the variable 'tot' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tot', result_add_76)
        
        # Getting the type of 'c' (line 64)
        c_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'c')
        int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'int')
        # Applying the binary operator '-=' (line 64)
        result_isub_79 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 8), '-=', c_77, int_78)
        # Assigning a type to the variable 'c' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'c', result_isub_79)
        
        pass
        # SSA join for while statement (line 61)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'tot' (line 66)
    tot_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'tot')
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', tot_80)
    
    # ################# End of 'bin_to_dec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bin_to_dec' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bin_to_dec'
    return stypy_return_type_81

# Assigning a type to the variable 'bin_to_dec' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'bin_to_dec', bin_to_dec)

@norecursion
def ceillog(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ceillog'
    module_type_store = module_type_store.open_function_context('ceillog', 69, 0, False)
    
    # Passed parameters checking function
    ceillog.stypy_localization = localization
    ceillog.stypy_type_of_self = None
    ceillog.stypy_type_store = module_type_store
    ceillog.stypy_function_name = 'ceillog'
    ceillog.stypy_param_names_list = ['n']
    ceillog.stypy_varargs_param_name = None
    ceillog.stypy_kwargs_param_name = None
    ceillog.stypy_call_defaults = defaults
    ceillog.stypy_call_varargs = varargs
    ceillog.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ceillog', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ceillog', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ceillog(...)' code ##################

    str_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', '\n    >>> print ceillog(3), ceillog(4), ceillog(5)\n    2 2 3\n    ')
    # Evaluating assert statement condition
    
    # Getting the type of 'n' (line 74)
    n_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'n')
    int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'int')
    # Applying the binary operator '>=' (line 74)
    result_ge_85 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), '>=', n_83, int_84)
    
    
    # Assigning a Num to a Name (line 75):
    int_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
    # Assigning a type to the variable 'c' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'c', int_86)
    
    
    int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 10), 'int')
    # Getting the type of 'c' (line 76)
    c_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'c')
    # Applying the binary operator '**' (line 76)
    result_pow_89 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 10), '**', int_87, c_88)
    
    # Getting the type of 'n' (line 76)
    n_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'n')
    # Applying the binary operator '<' (line 76)
    result_lt_91 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 10), '<', result_pow_89, n_90)
    
    # Testing if the while is going to be iterated (line 76)
    # Testing the type of an if condition (line 76)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_lt_91)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 76, 4), result_lt_91):
        # SSA begins for while statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'c' (line 77)
        c_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'c')
        int_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'int')
        # Applying the binary operator '+=' (line 77)
        result_iadd_94 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 8), '+=', c_92, int_93)
        # Assigning a type to the variable 'c' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'c', result_iadd_94)
        
        # SSA join for while statement (line 76)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'c' (line 78)
    c_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', c_95)
    
    # ################# End of 'ceillog(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ceillog' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_96)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ceillog'
    return stypy_return_type_96

# Assigning a type to the variable 'ceillog' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'ceillog', ceillog)

@norecursion
def status(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'status'
    module_type_store = module_type_store.open_function_context('status', 81, 0, False)
    
    # Passed parameters checking function
    status.stypy_localization = localization
    status.stypy_type_of_self = None
    status.stypy_type_store = module_type_store
    status.stypy_function_name = 'status'
    status.stypy_param_names_list = ['fr', 'to', 'oldpointer', 'digits', 'digits2', 'length', 'c', 'maxlength']
    status.stypy_varargs_param_name = None
    status.stypy_kwargs_param_name = None
    status.stypy_call_defaults = defaults
    status.stypy_call_varargs = varargs
    status.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'status', ['fr', 'to', 'oldpointer', 'digits', 'digits2', 'length', 'c', 'maxlength'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'status', localization, ['fr', 'to', 'oldpointer', 'digits', 'digits2', 'length', 'c', 'maxlength'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'status(...)' code ##################

    str_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\n    Print out the current pointer location, the current matching\n    string, and the current locations of "from" and "to".\n    Also report the range of conceivable values for the pointer and\n    length quantities.\n    ')
    pass
    
    # ################# End of 'status(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'status' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_98)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'status'
    return stypy_return_type_98

# Assigning a type to the variable 'status' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'status', status)

@norecursion
def searchstatus(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'searchstatus'
    module_type_store = module_type_store.open_function_context('searchstatus', 100, 0, False)
    
    # Passed parameters checking function
    searchstatus.stypy_localization = localization
    searchstatus.stypy_type_of_self = None
    searchstatus.stypy_type_store = module_type_store
    searchstatus.stypy_function_name = 'searchstatus'
    searchstatus.stypy_param_names_list = ['fr', 'to', 'L', 'c']
    searchstatus.stypy_varargs_param_name = None
    searchstatus.stypy_kwargs_param_name = None
    searchstatus.stypy_call_defaults = defaults
    searchstatus.stypy_call_varargs = varargs
    searchstatus.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'searchstatus', ['fr', 'to', 'L', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'searchstatus', localization, ['fr', 'to', 'L', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'searchstatus(...)' code ##################

    str_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', '\n    Show the current string (fr,to) that is being searched for.\n    ')
    pass
    
    # ################# End of 'searchstatus(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'searchstatus' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'searchstatus'
    return stypy_return_type_100

# Assigning a type to the variable 'searchstatus' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'searchstatus', searchstatus)

@norecursion
def encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 21), 'int')
    int_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'int')
    defaults = [int_101, int_102]
    # Create a new context for function 'encode'
    module_type_store = module_type_store.open_function_context('encode', 115, 0, False)
    
    # Passed parameters checking function
    encode.stypy_localization = localization
    encode.stypy_type_of_self = None
    encode.stypy_type_store = module_type_store
    encode.stypy_function_name = 'encode'
    encode.stypy_param_names_list = ['c', 'pretty', 'verbose']
    encode.stypy_varargs_param_name = None
    encode.stypy_kwargs_param_name = None
    encode.stypy_call_defaults = defaults
    encode.stypy_call_varargs = varargs
    encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode', ['c', 'pretty', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode', localization, ['c', 'pretty', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode(...)' code ##################

    str_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, (-1)), 'str', '\n    Encode using Lempel-Ziv (2), which sends pointers and lengths\n    Pretty printing\n    >>> print encode("000000000000100000000000",1)\n    0(0,1)(00,10)(000,100)(0000,0100)(1101,0)(0000,1011)\n\n    Normal printing\n    >>> print encode("000000000000100000000000",0)\n    0010010000100000001001101000001011\n\n    To understand the encoding procedure it might be\n    best to read the decoder.\n\n    Termination rule:\n    We have a special reserved "impossible pointer" string\n    and always have space for one extra one in the pointer area, to allow us to\n    send termination information, including whether the last bit needed removing or not.\n    If the last thing that happened was a perfect match without mismatch event,\n    which is the most common event, do nothing, send EOF. The decoder will sees the special\n    \'0\' bit after the special pointer and so does not include the final character.\n    If instead there happened to be a mismatch event at the exact moment\n    when the final char arrived, we do a standard decoding action.\n\n    ::-----SEQUENCE SO FAR-------::|--SEQUENCE STILL to be sent--\n         ^pointer                   ^fr         ^to\n         ------------               ------------\n         <- length ->               <- length ->\n    Once we have found the maximum length (to-fr) that matches, send\n         the values of pointer and length.\n\n    ')
    
    # Assigning a List to a Name (line 147):
    
    # Obtaining an instance of the builtin type 'list' (line 147)
    list_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 147)
    
    # Assigning a type to the variable 'output' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'output', list_104)
    
    # Assigning a Call to a Name (line 148):
    
    # Call to len(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'c' (line 148)
    c_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'c', False)
    # Processing the call keyword arguments (line 148)
    kwargs_107 = {}
    # Getting the type of 'len' (line 148)
    len_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'len', False)
    # Calling len(args, kwargs) (line 148)
    len_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), len_105, *[c_106], **kwargs_107)
    
    # Assigning a type to the variable 'L' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'L', len_call_result_108)
    # Evaluating assert statement condition
    
    # Getting the type of 'L' (line 149)
    L_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'L')
    int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 15), 'int')
    # Applying the binary operator '>' (line 149)
    result_gt_111 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '>', L_109, int_110)
    
    
    # Call to append(...): (line 150)
    # Processing the call arguments (line 150)
    
    # Obtaining the type of the subscript
    int_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'int')
    # Getting the type of 'c' (line 150)
    c_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 18), c_115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 150, 18), getitem___116, int_114)
    
    # Processing the call keyword arguments (line 150)
    kwargs_118 = {}
    # Getting the type of 'output' (line 150)
    output_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'output', False)
    # Obtaining the member 'append' of a type (line 150)
    append_113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 4), output_112, 'append')
    # Calling append(args, kwargs) (line 150)
    append_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 150, 4), append_113, *[subscript_call_result_117], **kwargs_118)
    
    
    # Assigning a Num to a Name (line 151):
    int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 9), 'int')
    # Assigning a type to the variable 'fr' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'fr', int_120)
    
    # Assigning a Num to a Name (line 152):
    int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'int')
    # Assigning a type to the variable 'eof_sent' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'eof_sent', int_121)
    
    
    # Getting the type of 'eof_sent' (line 153)
    eof_sent_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'eof_sent')
    int_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 23), 'int')
    # Applying the binary operator '==' (line 153)
    result_eq_124 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), '==', eof_sent_122, int_123)
    
    # Testing if the while is going to be iterated (line 153)
    # Testing the type of an if condition (line 153)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 4), result_eq_124)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 153, 4), result_eq_124):
        # SSA begins for while statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'fr' (line 154)
        fr_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'fr')
        # Assigning a type to the variable 'to' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'to', fr_125)
        
        # Assigning a Num to a Name (line 155):
        int_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'int')
        # Assigning a type to the variable 'oldpointer' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'oldpointer', int_126)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'eof_sent' (line 156)
        eof_sent_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'eof_sent')
        int_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 27), 'int')
        # Applying the binary operator '==' (line 156)
        result_eq_129 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 15), '==', eof_sent_127, int_128)
        
        
        # Getting the type of 'to' (line 156)
        to_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'to')
        # Getting the type of 'L' (line 156)
        L_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 41), 'L')
        # Applying the binary operator '<=' (line 156)
        result_le_132 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 35), '<=', to_130, L_131)
        
        # Applying the binary operator 'and' (line 156)
        result_and_keyword_133 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 14), 'and', result_eq_129, result_le_132)
        
        # Testing if the while is going to be iterated (line 156)
        # Testing the type of an if condition (line 156)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_and_keyword_133)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 156, 8), result_and_keyword_133):
            # SSA begins for while statement (line 156)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 'verbose' (line 157)
            verbose_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'verbose')
            int_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'int')
            # Applying the binary operator '>' (line 157)
            result_gt_136 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 15), '>', verbose_134, int_135)
            
            # Testing if the type of an if condition is none (line 157)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 157, 12), result_gt_136):
                pass
            else:
                
                # Testing the type of an if condition (line 157)
                if_condition_137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 12), result_gt_136)
                # Assigning a type to the variable 'if_condition_137' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'if_condition_137', if_condition_137)
                # SSA begins for if statement (line 157)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to searchstatus(...): (line 157)
                # Processing the call arguments (line 157)
                # Getting the type of 'fr' (line 157)
                fr_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), 'fr', False)
                # Getting the type of 'to' (line 157)
                to_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 46), 'to', False)
                # Getting the type of 'L' (line 157)
                L_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 50), 'L', False)
                # Getting the type of 'c' (line 157)
                c_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 53), 'c', False)
                # Processing the call keyword arguments (line 157)
                kwargs_143 = {}
                # Getting the type of 'searchstatus' (line 157)
                searchstatus_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'searchstatus', False)
                # Calling searchstatus(args, kwargs) (line 157)
                searchstatus_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 157, 29), searchstatus_138, *[fr_139, to_140, L_141, c_142], **kwargs_143)
                
                pass
                # SSA join for if statement (line 157)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 158):
            
            # Call to find(...): (line 158)
            # Processing the call arguments (line 158)
            
            # Obtaining the type of the subscript
            # Getting the type of 'fr' (line 158)
            fr_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 37), 'fr', False)
            # Getting the type of 'to' (line 158)
            to_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 40), 'to', False)
            slice_154 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 35), fr_152, to_153, None)
            # Getting the type of 'c' (line 158)
            c_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 35), 'c', False)
            # Obtaining the member '__getitem__' of a type (line 158)
            getitem___156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 35), c_155, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 158)
            subscript_call_result_157 = invoke(stypy.reporting.localization.Localization(__file__, 158, 35), getitem___156, slice_154)
            
            # Processing the call keyword arguments (line 158)
            kwargs_158 = {}
            
            # Obtaining the type of the subscript
            int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 24), 'int')
            # Getting the type of 'fr' (line 158)
            fr_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'fr', False)
            slice_147 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 22), int_145, fr_146, None)
            # Getting the type of 'c' (line 158)
            c_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'c', False)
            # Obtaining the member '__getitem__' of a type (line 158)
            getitem___149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 22), c_148, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 158)
            subscript_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), getitem___149, slice_147)
            
            # Obtaining the member 'find' of a type (line 158)
            find_151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 22), subscript_call_result_150, 'find')
            # Calling find(args, kwargs) (line 158)
            find_call_result_159 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), find_151, *[subscript_call_result_157], **kwargs_158)
            
            # Assigning a type to the variable 'pointer' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'pointer', find_call_result_159)
            
            # Getting the type of 'verbose' (line 159)
            verbose_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'verbose')
            int_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'int')
            # Applying the binary operator '>' (line 159)
            result_gt_162 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 15), '>', verbose_160, int_161)
            
            # Testing if the type of an if condition is none (line 159)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 12), result_gt_162):
                pass
            else:
                
                # Testing the type of an if condition (line 159)
                if_condition_163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 12), result_gt_162)
                # Assigning a type to the variable 'if_condition_163' (line 159)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'if_condition_163', if_condition_163)
                # SSA begins for if statement (line 159)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA join for if statement (line 159)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Evaluating a boolean operation
            
            # Getting the type of 'pointer' (line 160)
            pointer_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'pointer')
            int_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 27), 'int')
            # Applying the binary operator '==' (line 160)
            result_eq_166 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 16), '==', pointer_164, int_165)
            
            
            # Getting the type of 'to' (line 160)
            to_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'to')
            # Getting the type of 'L' (line 160)
            L_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 41), 'L')
            # Applying the binary operator '>=' (line 160)
            result_ge_169 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 35), '>=', to_167, L_168)
            
            # Applying the binary operator 'or' (line 160)
            result_or_keyword_170 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 15), 'or', result_eq_166, result_ge_169)
            
            # Testing if the type of an if condition is none (line 160)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 12), result_or_keyword_170):
                
                # Getting the type of 'to' (line 192)
                to_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'to')
                int_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 22), 'int')
                # Applying the binary operator '+=' (line 192)
                result_iadd_275 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 16), '+=', to_273, int_274)
                # Assigning a type to the variable 'to' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'to', result_iadd_275)
                
                
                # Assigning a Name to a Name (line 193):
                # Getting the type of 'pointer' (line 193)
                pointer_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'pointer')
                # Assigning a type to the variable 'oldpointer' (line 193)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'oldpointer', pointer_276)
                pass
            else:
                
                # Testing the type of an if condition (line 160)
                if_condition_171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 12), result_or_keyword_170)
                # Assigning a type to the variable 'if_condition_171' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'if_condition_171', if_condition_171)
                # SSA begins for if statement (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'pointer' (line 161)
                pointer_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'pointer')
                int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 31), 'int')
                # Applying the binary operator '!=' (line 161)
                result_ne_174 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 20), '!=', pointer_172, int_173)
                
                # Testing if the type of an if condition is none (line 161)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 16), result_ne_174):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 161)
                    if_condition_175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 16), result_ne_174)
                    # Assigning a type to the variable 'if_condition_175' (line 161)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'if_condition_175', if_condition_175)
                    # SSA begins for if statement (line 161)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 161):
                    # Getting the type of 'pointer' (line 161)
                    pointer_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 49), 'pointer')
                    # Assigning a type to the variable 'oldpointer' (line 161)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'oldpointer', pointer_176)
                    pass
                    # SSA join for if statement (line 161)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 162):
                
                # Call to ceillog(...): (line 162)
                # Processing the call arguments (line 162)
                # Getting the type of 'fr' (line 163)
                fr_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'fr', False)
                int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 25), 'int')
                # Applying the binary operator '+' (line 163)
                result_add_180 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 20), '+', fr_178, int_179)
                
                # Processing the call keyword arguments (line 162)
                kwargs_181 = {}
                # Getting the type of 'ceillog' (line 162)
                ceillog_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'ceillog', False)
                # Calling ceillog(args, kwargs) (line 162)
                ceillog_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 162, 25), ceillog_177, *[result_add_180], **kwargs_181)
                
                # Assigning a type to the variable 'digits' (line 162)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'digits', ceillog_call_result_182)
                
                # Assigning a BinOp to a Name (line 165):
                # Getting the type of 'fr' (line 165)
                fr_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'fr')
                # Getting the type of 'oldpointer' (line 165)
                oldpointer_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'oldpointer')
                # Applying the binary operator '-' (line 165)
                result_sub_185 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 28), '-', fr_183, oldpointer_184)
                
                # Assigning a type to the variable 'maxlength' (line 165)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'maxlength', result_sub_185)
                
                # Assigning a Call to a Name (line 166):
                
                # Call to ceillog(...): (line 166)
                # Processing the call arguments (line 166)
                # Getting the type of 'maxlength' (line 166)
                maxlength_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'maxlength', False)
                int_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 46), 'int')
                # Applying the binary operator '+' (line 166)
                result_add_189 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 34), '+', maxlength_187, int_188)
                
                # Processing the call keyword arguments (line 166)
                kwargs_190 = {}
                # Getting the type of 'ceillog' (line 166)
                ceillog_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 26), 'ceillog', False)
                # Calling ceillog(args, kwargs) (line 166)
                ceillog_call_result_191 = invoke(stypy.reporting.localization.Localization(__file__, 166, 26), ceillog_186, *[result_add_189], **kwargs_190)
                
                # Assigning a type to the variable 'digits2' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'digits2', ceillog_call_result_191)
                
                # Getting the type of 'pointer' (line 167)
                pointer_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'pointer')
                int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 31), 'int')
                # Applying the binary operator '==' (line 167)
                result_eq_194 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 20), '==', pointer_192, int_193)
                
                # Testing if the type of an if condition is none (line 167)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 16), result_eq_194):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 167)
                    if_condition_195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 16), result_eq_194)
                    # Assigning a type to the variable 'if_condition_195' (line 167)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'if_condition_195', if_condition_195)
                    # SSA begins for if statement (line 167)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'to' (line 167)
                    to_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'to')
                    int_197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 45), 'int')
                    # Applying the binary operator '-=' (line 167)
                    result_isub_198 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 39), '-=', to_196, int_197)
                    # Assigning a type to the variable 'to' (line 167)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'to', result_isub_198)
                    
                    pass
                    # SSA join for if statement (line 167)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a BinOp to a Name (line 168):
                # Getting the type of 'to' (line 168)
                to_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'to')
                # Getting the type of 'fr' (line 168)
                fr_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'fr')
                # Applying the binary operator '-' (line 168)
                result_sub_201 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 25), '-', to_199, fr_200)
                
                # Assigning a type to the variable 'length' (line 168)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'length', result_sub_201)
                
                # Getting the type of 'length' (line 169)
                length_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'length')
                # Getting the type of 'maxlength' (line 169)
                maxlength_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'maxlength')
                # Applying the binary operator '<' (line 169)
                result_lt_204 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 19), '<', length_202, maxlength_203)
                
                # Testing if the type of an if condition is none (line 169)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 169, 16), result_lt_204):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 169)
                    if_condition_205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 16), result_lt_204)
                    # Assigning a type to the variable 'if_condition_205' (line 169)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'if_condition_205', if_condition_205)
                    # SSA begins for if statement (line 169)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'to' (line 170)
                    to_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'to')
                    int_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 26), 'int')
                    # Applying the binary operator '+=' (line 170)
                    result_iadd_208 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 20), '+=', to_206, int_207)
                    # Assigning a type to the variable 'to' (line 170)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'to', result_iadd_208)
                    
                    pass
                    # SSA join for if statement (line 169)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'to' (line 172)
                to_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'to')
                # Getting the type of 'L' (line 172)
                L_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 26), 'L')
                # Applying the binary operator '>=' (line 172)
                result_ge_211 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 20), '>=', to_209, L_210)
                
                # Testing if the type of an if condition is none (line 172)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 16), result_ge_211):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 172)
                    if_condition_212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 16), result_ge_211)
                    # Assigning a type to the variable 'if_condition_212' (line 172)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'if_condition_212', if_condition_212)
                    # SSA begins for if statement (line 172)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'pointer' (line 173)
                    pointer_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'pointer')
                    int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'int')
                    # Applying the binary operator '!=' (line 173)
                    result_ne_215 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 24), '!=', pointer_213, int_214)
                    
                    # Testing if the type of an if condition is none (line 173)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 173, 20), result_ne_215):
                        
                        # Assigning a Num to a Name (line 176):
                        int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 37), 'int')
                        # Assigning a type to the variable 'specialbit' (line 176)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'specialbit', int_218)
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 173)
                        if_condition_216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 20), result_ne_215)
                        # Assigning a type to the variable 'if_condition_216' (line 173)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'if_condition_216', if_condition_216)
                        # SSA begins for if statement (line 173)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Num to a Name (line 174):
                        int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 37), 'int')
                        # Assigning a type to the variable 'specialbit' (line 174)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'specialbit', int_217)
                        pass
                        # SSA branch for the else part of an if statement (line 173)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Num to a Name (line 176):
                        int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 37), 'int')
                        # Assigning a type to the variable 'specialbit' (line 176)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'specialbit', int_218)
                        pass
                        # SSA join for if statement (line 173)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Call to append(...): (line 177)
                    # Processing the call arguments (line 177)
                    
                    # Call to printout(...): (line 177)
                    # Processing the call arguments (line 177)
                    
                    # Call to dec_to_bin(...): (line 177)
                    # Processing the call arguments (line 177)
                    # Getting the type of 'fr' (line 177)
                    fr_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 54), 'fr', False)
                    # Getting the type of 'digits' (line 177)
                    digits_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 58), 'digits', False)
                    # Processing the call keyword arguments (line 177)
                    kwargs_225 = {}
                    # Getting the type of 'dec_to_bin' (line 177)
                    dec_to_bin_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), 'dec_to_bin', False)
                    # Calling dec_to_bin(args, kwargs) (line 177)
                    dec_to_bin_call_result_226 = invoke(stypy.reporting.localization.Localization(__file__, 177, 43), dec_to_bin_222, *[fr_223, digits_224], **kwargs_225)
                    
                    
                    # Call to str(...): (line 178)
                    # Processing the call arguments (line 178)
                    # Getting the type of 'specialbit' (line 178)
                    specialbit_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 47), 'specialbit', False)
                    # Processing the call keyword arguments (line 178)
                    kwargs_229 = {}
                    # Getting the type of 'str' (line 178)
                    str_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 43), 'str', False)
                    # Calling str(args, kwargs) (line 178)
                    str_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 178, 43), str_227, *[specialbit_228], **kwargs_229)
                    
                    # Getting the type of 'pretty' (line 178)
                    pretty_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 60), 'pretty', False)
                    # Processing the call keyword arguments (line 177)
                    kwargs_232 = {}
                    # Getting the type of 'printout' (line 177)
                    printout_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 34), 'printout', False)
                    # Calling printout(args, kwargs) (line 177)
                    printout_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 177, 34), printout_221, *[dec_to_bin_call_result_226, str_call_result_230, pretty_231], **kwargs_232)
                    
                    # Processing the call keyword arguments (line 177)
                    kwargs_234 = {}
                    # Getting the type of 'output' (line 177)
                    output_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'output', False)
                    # Obtaining the member 'append' of a type (line 177)
                    append_220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), output_219, 'append')
                    # Calling append(args, kwargs) (line 177)
                    append_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 177, 20), append_220, *[printout_call_result_233], **kwargs_234)
                    
                    
                    # Assigning a Num to a Name (line 179):
                    int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'int')
                    # Assigning a type to the variable 'eof_sent' (line 179)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'eof_sent', int_236)
                    pass
                    # SSA join for if statement (line 172)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # Evaluating assert statement condition
                
                # Getting the type of 'length' (line 181)
                length_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'length')
                # Getting the type of 'maxlength' (line 181)
                maxlength_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'maxlength')
                # Applying the binary operator '<=' (line 181)
                result_le_239 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 23), '<=', length_237, maxlength_238)
                
                
                # Call to append(...): (line 182)
                # Processing the call arguments (line 182)
                
                # Call to printout(...): (line 182)
                # Processing the call arguments (line 182)
                
                # Call to dec_to_bin(...): (line 182)
                # Processing the call arguments (line 182)
                # Getting the type of 'oldpointer' (line 182)
                oldpointer_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 50), 'oldpointer', False)
                # Getting the type of 'digits' (line 182)
                digits_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 62), 'digits', False)
                # Processing the call keyword arguments (line 182)
                kwargs_246 = {}
                # Getting the type of 'dec_to_bin' (line 182)
                dec_to_bin_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'dec_to_bin', False)
                # Calling dec_to_bin(args, kwargs) (line 182)
                dec_to_bin_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 182, 39), dec_to_bin_243, *[oldpointer_244, digits_245], **kwargs_246)
                
                
                # Call to dec_to_bin(...): (line 183)
                # Processing the call arguments (line 183)
                # Getting the type of 'length' (line 183)
                length_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 50), 'length', False)
                # Getting the type of 'digits2' (line 183)
                digits2_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 58), 'digits2', False)
                # Processing the call keyword arguments (line 183)
                kwargs_251 = {}
                # Getting the type of 'dec_to_bin' (line 183)
                dec_to_bin_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'dec_to_bin', False)
                # Calling dec_to_bin(args, kwargs) (line 183)
                dec_to_bin_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 183, 39), dec_to_bin_248, *[length_249, digits2_250], **kwargs_251)
                
                # Getting the type of 'pretty' (line 183)
                pretty_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 68), 'pretty', False)
                # Processing the call keyword arguments (line 182)
                kwargs_254 = {}
                # Getting the type of 'printout' (line 182)
                printout_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'printout', False)
                # Calling printout(args, kwargs) (line 182)
                printout_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), printout_242, *[dec_to_bin_call_result_247, dec_to_bin_call_result_252, pretty_253], **kwargs_254)
                
                # Processing the call keyword arguments (line 182)
                kwargs_256 = {}
                # Getting the type of 'output' (line 182)
                output_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'output', False)
                # Obtaining the member 'append' of a type (line 182)
                append_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 16), output_240, 'append')
                # Calling append(args, kwargs) (line 182)
                append_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), append_241, *[printout_call_result_255], **kwargs_256)
                
                # Getting the type of 'verbose' (line 184)
                verbose_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'verbose')
                # Testing if the type of an if condition is none (line 184)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 16), verbose_258):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 184)
                    if_condition_259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 16), verbose_258)
                    # Assigning a type to the variable 'if_condition_259' (line 184)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'if_condition_259', if_condition_259)
                    # SSA begins for if statement (line 184)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to status(...): (line 185)
                    # Processing the call arguments (line 185)
                    # Getting the type of 'fr' (line 185)
                    fr_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'fr', False)
                    # Getting the type of 'to' (line 185)
                    to_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 31), 'to', False)
                    # Getting the type of 'oldpointer' (line 185)
                    oldpointer_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 35), 'oldpointer', False)
                    # Getting the type of 'digits' (line 185)
                    digits_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 47), 'digits', False)
                    # Getting the type of 'digits2' (line 185)
                    digits2_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 55), 'digits2', False)
                    # Getting the type of 'length' (line 185)
                    length_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 64), 'length', False)
                    # Getting the type of 'c' (line 185)
                    c_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 72), 'c', False)
                    # Getting the type of 'maxlength' (line 185)
                    maxlength_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 75), 'maxlength', False)
                    # Processing the call keyword arguments (line 185)
                    kwargs_269 = {}
                    # Getting the type of 'status' (line 185)
                    status_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'status', False)
                    # Calling status(args, kwargs) (line 185)
                    status_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), status_260, *[fr_261, to_262, oldpointer_263, digits_264, digits2_265, length_266, c_267, maxlength_268], **kwargs_269)
                    
                    pass
                    # SSA join for if statement (line 184)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Num to a Name (line 188):
                int_271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'int')
                # Assigning a type to the variable 'oldpointer' (line 188)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'oldpointer', int_271)
                
                # Assigning a Name to a Name (line 189):
                # Getting the type of 'to' (line 189)
                to_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'to')
                # Assigning a type to the variable 'fr' (line 189)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'fr', to_272)
                # SSA branch for the else part of an if statement (line 160)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'to' (line 192)
                to_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'to')
                int_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 22), 'int')
                # Applying the binary operator '+=' (line 192)
                result_iadd_275 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 16), '+=', to_273, int_274)
                # Assigning a type to the variable 'to' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'to', result_iadd_275)
                
                
                # Assigning a Name to a Name (line 193):
                # Getting the type of 'pointer' (line 193)
                pointer_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'pointer')
                # Assigning a type to the variable 'oldpointer' (line 193)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'oldpointer', pointer_276)
                pass
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()
                

            pass
            # SSA join for while statement (line 156)
            module_type_store = module_type_store.join_ssa_context()

        
        pass
        # SSA join for while statement (line 153)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'verbose' (line 197)
    verbose_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 7), 'verbose')
    # Testing if the type of an if condition is none (line 197)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 4), verbose_277):
        pass
    else:
        
        # Testing the type of an if condition (line 197)
        if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 4), verbose_277)
        # Assigning a type to the variable 'if_condition_278' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'if_condition_278', if_condition_278)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to join(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'output' (line 198)
    output_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'output', False)
    # Processing the call keyword arguments (line 198)
    kwargs_282 = {}
    str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 11), 'str', '')
    # Obtaining the member 'join' of a type (line 198)
    join_280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), str_279, 'join')
    # Calling join(args, kwargs) (line 198)
    join_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), join_280, *[output_281], **kwargs_282)
    
    # Assigning a type to the variable 'stypy_return_type' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type', join_call_result_283)
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 115)
    stypy_return_type_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_284

# Assigning a type to the variable 'encode' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'encode', encode)

@norecursion
def printout(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 49), 'int')
    defaults = [int_285]
    # Create a new context for function 'printout'
    module_type_store = module_type_store.open_function_context('printout', 201, 0, False)
    
    # Passed parameters checking function
    printout.stypy_localization = localization
    printout.stypy_type_of_self = None
    printout.stypy_type_store = module_type_store
    printout.stypy_function_name = 'printout'
    printout.stypy_param_names_list = ['pointerstring', 'lengthstring', 'pretty']
    printout.stypy_varargs_param_name = None
    printout.stypy_kwargs_param_name = None
    printout.stypy_call_defaults = defaults
    printout.stypy_call_varargs = varargs
    printout.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'printout', ['pointerstring', 'lengthstring', 'pretty'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'printout', localization, ['pointerstring', 'lengthstring', 'pretty'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'printout(...)' code ##################

    # Getting the type of 'pretty' (line 202)
    pretty_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 7), 'pretty')
    # Testing if the type of an if condition is none (line 202)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 202, 4), pretty_286):
        # Getting the type of 'pointerstring' (line 205)
        pointerstring_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'pointerstring')
        # Getting the type of 'lengthstring' (line 205)
        lengthstring_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'lengthstring')
        # Applying the binary operator '+' (line 205)
        result_add_299 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 15), '+', pointerstring_297, lengthstring_298)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', result_add_299)
    else:
        
        # Testing the type of an if condition (line 202)
        if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), pretty_286)
        # Assigning a type to the variable 'if_condition_287' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_287', if_condition_287)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 15), 'str', '(')
        # Getting the type of 'pointerstring' (line 203)
        pointerstring_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'pointerstring')
        # Applying the binary operator '+' (line 203)
        result_add_290 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 15), '+', str_288, pointerstring_289)
        
        str_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 37), 'str', ',')
        # Applying the binary operator '+' (line 203)
        result_add_292 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 35), '+', result_add_290, str_291)
        
        # Getting the type of 'lengthstring' (line 203)
        lengthstring_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 43), 'lengthstring')
        # Applying the binary operator '+' (line 203)
        result_add_294 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 41), '+', result_add_292, lengthstring_293)
        
        str_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 58), 'str', ')')
        # Applying the binary operator '+' (line 203)
        result_add_296 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 56), '+', result_add_294, str_295)
        
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', result_add_296)
        # SSA branch for the else part of an if statement (line 202)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'pointerstring' (line 205)
        pointerstring_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'pointerstring')
        # Getting the type of 'lengthstring' (line 205)
        lengthstring_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'lengthstring')
        # Applying the binary operator '+' (line 205)
        result_add_299 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 15), '+', pointerstring_297, lengthstring_298)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', result_add_299)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'printout(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'printout' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_300)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'printout'
    return stypy_return_type_300

# Assigning a type to the variable 'printout' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'printout', printout)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'int')
    defaults = [int_301]
    # Create a new context for function 'decode'
    module_type_store = module_type_store.open_function_context('decode', 208, 0, False)
    
    # Passed parameters checking function
    decode.stypy_localization = localization
    decode.stypy_type_of_self = None
    decode.stypy_type_store = module_type_store
    decode.stypy_function_name = 'decode'
    decode.stypy_param_names_list = ['li', 'verbose']
    decode.stypy_varargs_param_name = None
    decode.stypy_kwargs_param_name = None
    decode.stypy_call_defaults = defaults
    decode.stypy_call_varargs = varargs
    decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode', ['li', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode', localization, ['li', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode(...)' code ##################

    str_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, (-1)), 'str', '\n    >>> print decode(list("0010010000100000001001101000001010"))\n    00000000000010000000000\n    ')
    # Evaluating assert statement condition
    
    
    # Call to len(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'li' (line 213)
    li_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'li', False)
    # Processing the call keyword arguments (line 213)
    kwargs_305 = {}
    # Getting the type of 'len' (line 213)
    len_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'len', False)
    # Calling len(args, kwargs) (line 213)
    len_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), len_303, *[li_304], **kwargs_305)
    
    int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 22), 'int')
    # Applying the binary operator '>' (line 213)
    result_gt_308 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 12), '>', len_call_result_306, int_307)
    
    
    # Assigning a Call to a Name (line 214):
    
    # Call to pop(...): (line 214)
    # Processing the call arguments (line 214)
    int_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 15), 'int')
    # Processing the call keyword arguments (line 214)
    kwargs_312 = {}
    # Getting the type of 'li' (line 214)
    li_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'li', False)
    # Obtaining the member 'pop' of a type (line 214)
    pop_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), li_309, 'pop')
    # Calling pop(args, kwargs) (line 214)
    pop_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), pop_310, *[int_311], **kwargs_312)
    
    # Assigning a type to the variable 'c' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'c', pop_call_result_313)
    
    # Assigning a Num to a Name (line 215):
    int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 9), 'int')
    # Assigning a type to the variable 'fr' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'fr', int_314)
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'fr' (line 216)
    fr_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 9), 'fr')
    # Assigning a type to the variable 'to' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'to', fr_315)
    
    # Assigning a Num to a Name (line 218):
    int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 14), 'int')
    # Assigning a type to the variable 'not_eof' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'not_eof', int_316)
    
    # Assigning a Num to a Name (line 219):
    int_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 17), 'int')
    # Assigning a type to the variable 'specialbit' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'specialbit', int_317)
    
    # Getting the type of 'not_eof' (line 220)
    not_eof_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 10), 'not_eof')
    # Testing if the while is going to be iterated (line 220)
    # Testing the type of an if condition (line 220)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 4), not_eof_318)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 220, 4), not_eof_318):
        # SSA begins for while statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'li' (line 221)
        li_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'li', False)
        # Processing the call keyword arguments (line 221)
        kwargs_321 = {}
        # Getting the type of 'len' (line 221)
        len_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'len', False)
        # Calling len(args, kwargs) (line 221)
        len_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), len_319, *[li_320], **kwargs_321)
        
        int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 26), 'int')
        # Applying the binary operator '>' (line 221)
        result_gt_324 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 16), '>', len_call_result_322, int_323)
        
        
        # Assigning a Call to a Name (line 222):
        
        # Call to ceillog(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'fr' (line 222)
        fr_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'fr', False)
        int_327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 30), 'int')
        # Applying the binary operator '+' (line 222)
        result_add_328 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 25), '+', fr_326, int_327)
        
        # Processing the call keyword arguments (line 222)
        kwargs_329 = {}
        # Getting the type of 'ceillog' (line 222)
        ceillog_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 17), 'ceillog', False)
        # Calling ceillog(args, kwargs) (line 222)
        ceillog_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 222, 17), ceillog_325, *[result_add_328], **kwargs_329)
        
        # Assigning a type to the variable 'digits' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'digits', ceillog_call_result_330)
        
        # Assigning a Call to a Name (line 223):
        
        # Call to bin_to_dec(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'li' (line 223)
        li_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'li', False)
        # Getting the type of 'digits' (line 223)
        digits_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'digits', False)
        # Processing the call keyword arguments (line 223)
        kwargs_334 = {}
        # Getting the type of 'bin_to_dec' (line 223)
        bin_to_dec_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 18), 'bin_to_dec', False)
        # Calling bin_to_dec(args, kwargs) (line 223)
        bin_to_dec_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 223, 18), bin_to_dec_331, *[li_332, digits_333], **kwargs_334)
        
        # Assigning a type to the variable 'pointer' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'pointer', bin_to_dec_call_result_335)
        
        # Assigning a BinOp to a Name (line 224):
        # Getting the type of 'fr' (line 224)
        fr_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'fr')
        # Getting the type of 'pointer' (line 224)
        pointer_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 25), 'pointer')
        # Applying the binary operator '-' (line 224)
        result_sub_338 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 20), '-', fr_336, pointer_337)
        
        # Assigning a type to the variable 'maxlength' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'maxlength', result_sub_338)
        
        # Getting the type of 'pointer' (line 225)
        pointer_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'pointer')
        # Getting the type of 'fr' (line 225)
        fr_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 'fr')
        # Applying the binary operator '==' (line 225)
        result_eq_341 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 11), '==', pointer_339, fr_340)
        
        # Testing if the type of an if condition is none (line 225)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 225, 8), result_eq_341):
            pass
        else:
            
            # Testing the type of an if condition (line 225)
            if_condition_342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), result_eq_341)
            # Assigning a type to the variable 'if_condition_342' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_342', if_condition_342)
            # SSA begins for if statement (line 225)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 226):
            
            # Call to int(...): (line 226)
            # Processing the call arguments (line 226)
            
            # Call to pop(...): (line 226)
            # Processing the call arguments (line 226)
            int_346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'int')
            # Processing the call keyword arguments (line 226)
            kwargs_347 = {}
            # Getting the type of 'li' (line 226)
            li_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'li', False)
            # Obtaining the member 'pop' of a type (line 226)
            pop_345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 29), li_344, 'pop')
            # Calling pop(args, kwargs) (line 226)
            pop_call_result_348 = invoke(stypy.reporting.localization.Localization(__file__, 226, 29), pop_345, *[int_346], **kwargs_347)
            
            # Processing the call keyword arguments (line 226)
            kwargs_349 = {}
            # Getting the type of 'int' (line 226)
            int_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 25), 'int', False)
            # Calling int(args, kwargs) (line 226)
            int_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 226, 25), int_343, *[pop_call_result_348], **kwargs_349)
            
            # Assigning a type to the variable 'specialbit' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'specialbit', int_call_result_350)
            
            # Assigning a Call to a Name (line 227):
            
            # Call to bin_to_dec(...): (line 227)
            # Processing the call arguments (line 227)
            # Getting the type of 'li' (line 227)
            li_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'li', False)
            # Getting the type of 'digits' (line 227)
            digits_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'digits', False)
            # Processing the call keyword arguments (line 227)
            kwargs_354 = {}
            # Getting the type of 'bin_to_dec' (line 227)
            bin_to_dec_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'bin_to_dec', False)
            # Calling bin_to_dec(args, kwargs) (line 227)
            bin_to_dec_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 227, 22), bin_to_dec_351, *[li_352, digits_353], **kwargs_354)
            
            # Assigning a type to the variable 'pointer' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'pointer', bin_to_dec_call_result_355)
            
            # Assigning a BinOp to a Name (line 228):
            # Getting the type of 'fr' (line 228)
            fr_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'fr')
            # Getting the type of 'pointer' (line 228)
            pointer_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 29), 'pointer')
            # Applying the binary operator '-' (line 228)
            result_sub_358 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 24), '-', fr_356, pointer_357)
            
            # Assigning a type to the variable 'maxlength' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'maxlength', result_sub_358)
            
            # Assigning a Num to a Name (line 229):
            int_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 22), 'int')
            # Assigning a type to the variable 'not_eof' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'not_eof', int_359)
            pass
            # SSA join for if statement (line 225)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 231):
        
        # Call to ceillog(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'maxlength' (line 231)
        maxlength_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'maxlength', False)
        int_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 38), 'int')
        # Applying the binary operator '+' (line 231)
        result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 26), '+', maxlength_361, int_362)
        
        # Processing the call keyword arguments (line 231)
        kwargs_364 = {}
        # Getting the type of 'ceillog' (line 231)
        ceillog_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'ceillog', False)
        # Calling ceillog(args, kwargs) (line 231)
        ceillog_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 231, 18), ceillog_360, *[result_add_363], **kwargs_364)
        
        # Assigning a type to the variable 'digits2' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'digits2', ceillog_call_result_365)
        
        # Assigning a Call to a Name (line 232):
        
        # Call to bin_to_dec(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'li' (line 232)
        li_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'li', False)
        # Getting the type of 'digits2' (line 232)
        digits2_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'digits2', False)
        # Processing the call keyword arguments (line 232)
        kwargs_369 = {}
        # Getting the type of 'bin_to_dec' (line 232)
        bin_to_dec_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'bin_to_dec', False)
        # Calling bin_to_dec(args, kwargs) (line 232)
        bin_to_dec_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), bin_to_dec_366, *[li_367, digits2_368], **kwargs_369)
        
        # Assigning a type to the variable 'length' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'length', bin_to_dec_call_result_370)
        
        # Assigning a Subscript to a Name (line 233):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pointer' (line 233)
        pointer_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'pointer')
        # Getting the type of 'pointer' (line 233)
        pointer_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 29), 'pointer')
        # Getting the type of 'length' (line 233)
        length_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 39), 'length')
        # Applying the binary operator '+' (line 233)
        result_add_374 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 29), '+', pointer_372, length_373)
        
        slice_375 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 233, 19), pointer_371, result_add_374, None)
        # Getting the type of 'c' (line 233)
        c_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'c')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 19), c_376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_378 = invoke(stypy.reporting.localization.Localization(__file__, 233, 19), getitem___377, slice_375)
        
        # Assigning a type to the variable 'addition' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'addition', subscript_call_result_378)
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'addition' (line 234)
        addition_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'addition', False)
        # Processing the call keyword arguments (line 234)
        kwargs_381 = {}
        # Getting the type of 'len' (line 234)
        len_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'len', False)
        # Calling len(args, kwargs) (line 234)
        len_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 234, 15), len_379, *[addition_380], **kwargs_381)
        
        # Getting the type of 'length' (line 234)
        length_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 32), 'length')
        # Applying the binary operator '==' (line 234)
        result_eq_384 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 15), '==', len_call_result_382, length_383)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'not_eof' (line 235)
        not_eof_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'not_eof')
        int_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'int')
        # Applying the binary operator '==' (line 235)
        result_eq_387 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 13), '==', not_eof_385, int_386)
        
        
        # Getting the type of 'specialbit' (line 235)
        specialbit_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'specialbit')
        int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 46), 'int')
        # Applying the binary operator '==' (line 235)
        result_eq_390 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 32), '==', specialbit_388, int_389)
        
        # Applying the binary operator 'and' (line 235)
        result_and_keyword_391 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 12), 'and', result_eq_387, result_eq_390)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'not_eof' (line 235)
        not_eof_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 54), 'not_eof')
        
        # Getting the type of 'length' (line 235)
        length_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 67), 'length')
        # Getting the type of 'maxlength' (line 235)
        maxlength_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 76), 'maxlength')
        # Applying the binary operator '<' (line 235)
        result_lt_395 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 67), '<', length_393, maxlength_394)
        
        # Applying the binary operator 'and' (line 235)
        result_and_keyword_396 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 54), 'and', not_eof_392, result_lt_395)
        
        # Applying the binary operator 'or' (line 235)
        result_or_keyword_397 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), 'or', result_and_keyword_391, result_and_keyword_396)
        
        # Testing if the type of an if condition is none (line 235)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 235, 8), result_or_keyword_397):
            
            # Assigning a Str to a Name (line 238):
            str_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 23), 'str', '')
            # Assigning a type to the variable 'opposite' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'opposite', str_413)
        else:
            
            # Testing the type of an if condition (line 235)
            if_condition_398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_or_keyword_397)
            # Assigning a type to the variable 'if_condition_398' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_398', if_condition_398)
            # SSA begins for if statement (line 235)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 236):
            
            # Call to str(...): (line 236)
            # Processing the call arguments (line 236)
            int_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 27), 'int')
            
            # Call to int(...): (line 236)
            # Processing the call arguments (line 236)
            
            # Obtaining the type of the subscript
            # Getting the type of 'pointer' (line 236)
            pointer_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 37), 'pointer', False)
            # Getting the type of 'length' (line 236)
            length_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 47), 'length', False)
            # Applying the binary operator '+' (line 236)
            result_add_404 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 37), '+', pointer_402, length_403)
            
            # Getting the type of 'c' (line 236)
            c_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 35), 'c', False)
            # Obtaining the member '__getitem__' of a type (line 236)
            getitem___406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 35), c_405, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 236)
            subscript_call_result_407 = invoke(stypy.reporting.localization.Localization(__file__, 236, 35), getitem___406, result_add_404)
            
            # Processing the call keyword arguments (line 236)
            kwargs_408 = {}
            # Getting the type of 'int' (line 236)
            int_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 31), 'int', False)
            # Calling int(args, kwargs) (line 236)
            int_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 236, 31), int_401, *[subscript_call_result_407], **kwargs_408)
            
            # Applying the binary operator '-' (line 236)
            result_sub_410 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 27), '-', int_400, int_call_result_409)
            
            # Processing the call keyword arguments (line 236)
            kwargs_411 = {}
            # Getting the type of 'str' (line 236)
            str_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'str', False)
            # Calling str(args, kwargs) (line 236)
            str_call_result_412 = invoke(stypy.reporting.localization.Localization(__file__, 236, 23), str_399, *[result_sub_410], **kwargs_411)
            
            # Assigning a type to the variable 'opposite' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'opposite', str_call_result_412)
            # SSA branch for the else part of an if statement (line 235)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 238):
            str_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 23), 'str', '')
            # Assigning a type to the variable 'opposite' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'opposite', str_413)
            # SSA join for if statement (line 235)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 239):
        # Getting the type of 'c' (line 239)
        c_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'c')
        # Getting the type of 'addition' (line 239)
        addition_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'addition')
        # Applying the binary operator '+' (line 239)
        result_add_416 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 12), '+', c_414, addition_415)
        
        # Getting the type of 'opposite' (line 239)
        opposite_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'opposite')
        # Applying the binary operator '+' (line 239)
        result_add_418 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 25), '+', result_add_416, opposite_417)
        
        # Assigning a type to the variable 'c' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'c', result_add_418)
        # Getting the type of 'verbose' (line 240)
        verbose_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'verbose')
        # Testing if the type of an if condition is none (line 240)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 240, 8), verbose_419):
            pass
        else:
            
            # Testing the type of an if condition (line 240)
            if_condition_420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), verbose_419)
            # Assigning a type to the variable 'if_condition_420' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_420', if_condition_420)
            # SSA begins for if statement (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 241):
            # Getting the type of 'length' (line 241)
            length_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'length')
            # Getting the type of 'fr' (line 241)
            fr_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'fr')
            # Applying the binary operator '+' (line 241)
            result_add_423 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 17), '+', length_421, fr_422)
            
            int_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 31), 'int')
            # Applying the binary operator '+' (line 241)
            result_add_425 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 29), '+', result_add_423, int_424)
            
            # Assigning a type to the variable 'to' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'to', result_add_425)
            
            # Call to status(...): (line 242)
            # Processing the call arguments (line 242)
            # Getting the type of 'fr' (line 242)
            fr_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'fr', False)
            # Getting the type of 'to' (line 242)
            to_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'to', False)
            # Getting the type of 'pointer' (line 242)
            pointer_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 27), 'pointer', False)
            # Getting the type of 'digits' (line 242)
            digits_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 36), 'digits', False)
            # Getting the type of 'digits2' (line 242)
            digits2_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 44), 'digits2', False)
            # Getting the type of 'length' (line 242)
            length_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 53), 'length', False)
            # Getting the type of 'c' (line 242)
            c_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 61), 'c', False)
            # Getting the type of 'maxlength' (line 242)
            maxlength_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 64), 'maxlength', False)
            # Processing the call keyword arguments (line 242)
            kwargs_435 = {}
            # Getting the type of 'status' (line 242)
            status_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'status', False)
            # Calling status(args, kwargs) (line 242)
            status_call_result_436 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), status_426, *[fr_427, to_428, pointer_429, digits_430, digits2_431, length_432, c_433, maxlength_434], **kwargs_435)
            
            pass
            # SSA join for if statement (line 240)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 244):
        
        # Call to len(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'c' (line 244)
        c_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'c', False)
        # Processing the call keyword arguments (line 244)
        kwargs_439 = {}
        # Getting the type of 'len' (line 244)
        len_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 13), 'len', False)
        # Calling len(args, kwargs) (line 244)
        len_call_result_440 = invoke(stypy.reporting.localization.Localization(__file__, 244, 13), len_437, *[c_438], **kwargs_439)
        
        # Assigning a type to the variable 'fr' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'fr', len_call_result_440)
        # SSA join for while statement (line 220)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'c' (line 245)
    c_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type', c_441)
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_442)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_442

# Assigning a type to the variable 'decode' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'decode', decode)

@norecursion
def test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test'
    module_type_store = module_type_store.open_function_context('test', 248, 0, False)
    
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

    
    # Assigning a List to a Name (line 250):
    
    # Obtaining an instance of the builtin type 'list' (line 250)
    list_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 250)
    # Adding element type (line 250)
    str_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 16), 'str', '0010000000001000000000001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 15), list_443, str_444)
    # Adding element type (line 250)
    str_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 45), 'str', '00000000000010000000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 15), list_443, str_445)
    
    # Assigning a type to the variable 'examples' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'examples', list_443)
    
    # Assigning a List to a Name (line 251):
    
    # Obtaining an instance of the builtin type 'list' (line 251)
    list_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 251)
    # Adding element type (line 251)
    str_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'str', '1010101010101010101010101010101010101010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_447)
    # Adding element type (line 251)
    str_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 17), 'str', '011')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_448)
    # Adding element type (line 251)
    str_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 17), 'str', '01')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_449)
    # Adding element type (line 251)
    str_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 23), 'str', '10')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_450)
    # Adding element type (line 251)
    str_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 29), 'str', '11')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_451)
    # Adding element type (line 251)
    str_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 35), 'str', '00')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_452)
    # Adding element type (line 251)
    str_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 41), 'str', '000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_453)
    # Adding element type (line 251)
    str_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 48), 'str', '001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_454)
    # Adding element type (line 251)
    str_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 55), 'str', '010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_455)
    # Adding element type (line 251)
    str_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 62), 'str', '011')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_456)
    # Adding element type (line 251)
    str_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 69), 'str', '100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_457)
    # Adding element type (line 251)
    str_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 76), 'str', '101')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_458)
    # Adding element type (line 251)
    str_459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 83), 'str', '110')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_459)
    # Adding element type (line 251)
    str_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 17), 'str', '1010100000000000000000000101010101010101000000000000101010101010101010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_460)
    # Adding element type (line 251)
    str_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 17), 'str', '10101010101010101010101010101010101010101')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_461)
    # Adding element type (line 251)
    str_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 17), 'str', '00000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_462)
    # Adding element type (line 251)
    str_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 26), 'str', '000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_463)
    # Adding element type (line 251)
    str_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 36), 'str', '0000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_464)
    # Adding element type (line 251)
    str_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 47), 'str', '00000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_465)
    # Adding element type (line 251)
    str_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 59), 'str', '000000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_466)
    # Adding element type (line 251)
    str_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 72), 'str', '0000000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_467)
    # Adding element type (line 251)
    str_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 17), 'str', '00001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_468)
    # Adding element type (line 251)
    str_469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 26), 'str', '000001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_469)
    # Adding element type (line 251)
    str_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 36), 'str', '0000001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_470)
    # Adding element type (line 251)
    str_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 47), 'str', '00000001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_471)
    # Adding element type (line 251)
    str_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 59), 'str', '000000001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_472)
    # Adding element type (line 251)
    str_473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 72), 'str', '0000000001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_473)
    # Adding element type (line 251)
    str_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 17), 'str', '0000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_474)
    # Adding element type (line 251)
    str_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 25), 'str', '0001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_475)
    # Adding element type (line 251)
    str_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 33), 'str', '0010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_476)
    # Adding element type (line 251)
    str_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 41), 'str', '0011')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_477)
    # Adding element type (line 251)
    str_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 49), 'str', '0100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_478)
    # Adding element type (line 251)
    str_479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 57), 'str', '0101')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_479)
    # Adding element type (line 251)
    str_480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 65), 'str', '0110')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_480)
    # Adding element type (line 251)
    str_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 17), 'str', '0111')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_481)
    # Adding element type (line 251)
    str_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 25), 'str', '1000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_482)
    # Adding element type (line 251)
    str_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 33), 'str', '1001')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_483)
    # Adding element type (line 251)
    str_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 41), 'str', '1010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_484)
    # Adding element type (line 251)
    str_485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 49), 'str', '1011')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_485)
    # Adding element type (line 251)
    str_486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 57), 'str', '1100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_486)
    # Adding element type (line 251)
    str_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 65), 'str', '1101')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_487)
    # Adding element type (line 251)
    str_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 73), 'str', '1110')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_488)
    # Adding element type (line 251)
    str_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 81), 'str', '1111')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_489)
    # Adding element type (line 251)
    str_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 17), 'str', '111')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_490)
    # Adding element type (line 251)
    str_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'str', '110010010101000000000001110100100100000000000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_491)
    # Adding element type (line 251)
    str_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 17), 'str', '00000000000010000000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_492)
    # Adding element type (line 251)
    str_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 44), 'str', '1100100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_493)
    # Adding element type (line 251)
    str_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 55), 'str', '100100')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), list_446, str_494)
    
    # Assigning a type to the variable 'examples2' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'examples2', list_446)
    
    # Assigning a Num to a Name (line 262):
    int_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 13), 'int')
    # Assigning a type to the variable 'pretty' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'pretty', int_495)
    
    # Assigning a Num to a Name (line 263):
    int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 14), 'int')
    # Assigning a type to the variable 'verbose' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'verbose', int_496)
    
    # Getting the type of 'examples' (line 264)
    examples_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 14), 'examples')
    # Testing if the for loop is going to be iterated (line 264)
    # Testing the type of a for loop iterable (line 264)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 264, 4), examples_497)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 264, 4), examples_497):
        # Getting the type of the for loop variable (line 264)
        for_loop_var_498 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 264, 4), examples_497)
        # Assigning a type to the variable 'ex' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'ex', for_loop_var_498)
        # SSA begins for a for statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 267):
        
        # Call to encode(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'ex' (line 267)
        ex_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'ex', False)
        # Getting the type of 'pretty' (line 267)
        pretty_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'pretty', False)
        # Getting the type of 'verbose' (line 267)
        verbose_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'verbose', False)
        # Processing the call keyword arguments (line 267)
        kwargs_503 = {}
        # Getting the type of 'encode' (line 267)
        encode_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 14), 'encode', False)
        # Calling encode(args, kwargs) (line 267)
        encode_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 267, 14), encode_499, *[ex_500, pretty_501, verbose_502], **kwargs_503)
        
        # Assigning a type to the variable 'zip' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'zip', encode_call_result_504)
        
        # Getting the type of 'verbose' (line 268)
        verbose_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'verbose')
        int_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 21), 'int')
        # Applying the binary operator '>' (line 268)
        result_gt_507 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 11), '>', verbose_505, int_506)
        
        # Testing if the type of an if condition is none (line 268)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 268, 8), result_gt_507):
            pass
        else:
            
            # Testing the type of an if condition (line 268)
            if_condition_508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), result_gt_507)
            # Assigning a type to the variable 'if_condition_508' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_508', if_condition_508)
            # SSA begins for if statement (line 268)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'zip' (line 268)
            zip_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'zip')
            # SSA join for if statement (line 268)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 269):
        
        # Call to encode(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'ex' (line 269)
        ex_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'ex', False)
        int_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 26), 'int')
        int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 29), 'int')
        # Processing the call keyword arguments (line 269)
        kwargs_514 = {}
        # Getting the type of 'encode' (line 269)
        encode_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'encode', False)
        # Calling encode(args, kwargs) (line 269)
        encode_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 269, 15), encode_510, *[ex_511, int_512, int_513], **kwargs_514)
        
        # Assigning a type to the variable 'zip2' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'zip2', encode_call_result_515)
        
        # Assigning a Call to a Name (line 271):
        
        # Call to decode(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Call to list(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'zip2' (line 271)
        zip2_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'zip2', False)
        # Processing the call keyword arguments (line 271)
        kwargs_519 = {}
        # Getting the type of 'list' (line 271)
        list_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'list', False)
        # Calling list(args, kwargs) (line 271)
        list_call_result_520 = invoke(stypy.reporting.localization.Localization(__file__, 271, 21), list_517, *[zip2_518], **kwargs_519)
        
        # Getting the type of 'verbose' (line 271)
        verbose_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 33), 'verbose', False)
        # Processing the call keyword arguments (line 271)
        kwargs_522 = {}
        # Getting the type of 'decode' (line 271)
        decode_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'decode', False)
        # Calling decode(args, kwargs) (line 271)
        decode_call_result_523 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), decode_516, *[list_call_result_520, verbose_521], **kwargs_522)
        
        # Assigning a type to the variable 'unc' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'unc', decode_call_result_523)
        
        # Getting the type of 'unc' (line 273)
        unc_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'unc')
        # Getting the type of 'ex' (line 273)
        ex_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'ex')
        # Applying the binary operator '==' (line 273)
        result_eq_526 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), '==', unc_524, ex_525)
        
        # Testing if the type of an if condition is none (line 273)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_526):
            # Evaluating assert statement condition
            # Getting the type of 'False' (line 278)
            False_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'False')
        else:
            
            # Testing the type of an if condition (line 273)
            if_condition_527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_526)
            # Assigning a type to the variable 'if_condition_527' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_527', if_condition_527)
            # SSA begins for if statement (line 273)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 273)
            module_type_store.open_ssa_branch('else')
            # Evaluating assert statement condition
            # Getting the type of 'False' (line 278)
            False_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'False')
            # SSA join for if statement (line 273)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    int_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'int')
    # Testing if the type of an if condition is none (line 280)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 280, 4), int_529):
        pass
    else:
        
        # Testing the type of an if condition (line 280)
        if_condition_530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 4), int_529)
        # Assigning a type to the variable 'if_condition_530' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'if_condition_530', if_condition_530)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 281):
        int_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 17), 'int')
        # Assigning a type to the variable 'pretty' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'pretty', int_531)
        
        # Assigning a Num to a Name (line 282):
        int_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'int')
        # Assigning a type to the variable 'verbose' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'verbose', int_532)
        
        # Getting the type of 'examples2' (line 283)
        examples2_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 18), 'examples2')
        # Testing if the for loop is going to be iterated (line 283)
        # Testing the type of a for loop iterable (line 283)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 8), examples2_533)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 283, 8), examples2_533):
            # Getting the type of the for loop variable (line 283)
            for_loop_var_534 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 8), examples2_533)
            # Assigning a type to the variable 'ex' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'ex', for_loop_var_534)
            # SSA begins for a for statement (line 283)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 284):
            
            # Call to encode(...): (line 284)
            # Processing the call arguments (line 284)
            # Getting the type of 'ex' (line 284)
            ex_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'ex', False)
            # Getting the type of 'pretty' (line 284)
            pretty_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 'pretty', False)
            # Getting the type of 'verbose' (line 284)
            verbose_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 37), 'verbose', False)
            # Processing the call keyword arguments (line 284)
            kwargs_539 = {}
            # Getting the type of 'encode' (line 284)
            encode_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'encode', False)
            # Calling encode(args, kwargs) (line 284)
            encode_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 284, 18), encode_535, *[ex_536, pretty_537, verbose_538], **kwargs_539)
            
            # Assigning a type to the variable 'zip' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'zip', encode_call_result_540)
            
            # Assigning a Call to a Name (line 286):
            
            # Call to encode(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 'ex' (line 286)
            ex_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'ex', False)
            int_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 30), 'int')
            int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 33), 'int')
            # Processing the call keyword arguments (line 286)
            kwargs_545 = {}
            # Getting the type of 'encode' (line 286)
            encode_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'encode', False)
            # Calling encode(args, kwargs) (line 286)
            encode_call_result_546 = invoke(stypy.reporting.localization.Localization(__file__, 286, 19), encode_541, *[ex_542, int_543, int_544], **kwargs_545)
            
            # Assigning a type to the variable 'zip2' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'zip2', encode_call_result_546)
            
            # Assigning a Call to a Name (line 288):
            
            # Call to decode(...): (line 288)
            # Processing the call arguments (line 288)
            
            # Call to list(...): (line 288)
            # Processing the call arguments (line 288)
            # Getting the type of 'zip2' (line 288)
            zip2_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'zip2', False)
            # Processing the call keyword arguments (line 288)
            kwargs_550 = {}
            # Getting the type of 'list' (line 288)
            list_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'list', False)
            # Calling list(args, kwargs) (line 288)
            list_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 288, 25), list_548, *[zip2_549], **kwargs_550)
            
            # Getting the type of 'verbose' (line 288)
            verbose_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'verbose', False)
            # Processing the call keyword arguments (line 288)
            kwargs_553 = {}
            # Getting the type of 'decode' (line 288)
            decode_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'decode', False)
            # Calling decode(args, kwargs) (line 288)
            decode_call_result_554 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), decode_547, *[list_call_result_551, verbose_552], **kwargs_553)
            
            # Assigning a type to the variable 'unc' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'unc', decode_call_result_554)
            
            # Getting the type of 'unc' (line 290)
            unc_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'unc')
            # Getting the type of 'ex' (line 290)
            ex_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'ex')
            # Applying the binary operator '==' (line 290)
            result_eq_557 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 15), '==', unc_555, ex_556)
            
            # Testing if the type of an if condition is none (line 290)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 290, 12), result_eq_557):
                # Evaluating assert statement condition
                # Getting the type of 'False' (line 294)
                False_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'False')
            else:
                
                # Testing the type of an if condition (line 290)
                if_condition_558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 12), result_eq_557)
                # Assigning a type to the variable 'if_condition_558' (line 290)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'if_condition_558', if_condition_558)
                # SSA begins for if statement (line 290)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 290)
                module_type_store.open_ssa_branch('else')
                # Evaluating assert statement condition
                # Getting the type of 'False' (line 294)
                False_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'False')
                # SSA join for if statement (line 290)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a List to a Name (line 296):
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        str_561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 20), 'str', '0010010000100000001001101000001001')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 19), list_560, str_561)
        
        # Assigning a type to the variable 'examples' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'examples', list_560)
        
        # Getting the type of 'examples' (line 297)
        examples_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 18), 'examples')
        # Testing if the for loop is going to be iterated (line 297)
        # Testing the type of a for loop iterable (line 297)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 297, 8), examples_562)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 297, 8), examples_562):
            # Getting the type of the for loop variable (line 297)
            for_loop_var_563 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 297, 8), examples_562)
            # Assigning a type to the variable 'ex' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'ex', for_loop_var_563)
            # SSA begins for a for statement (line 297)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to decode(...): (line 299)
            # Processing the call arguments (line 299)
            
            # Call to list(...): (line 299)
            # Processing the call arguments (line 299)
            # Getting the type of 'ex' (line 299)
            ex_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'ex', False)
            # Processing the call keyword arguments (line 299)
            kwargs_567 = {}
            # Getting the type of 'list' (line 299)
            list_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'list', False)
            # Calling list(args, kwargs) (line 299)
            list_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 299, 19), list_565, *[ex_566], **kwargs_567)
            
            # Getting the type of 'verbose' (line 299)
            verbose_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'verbose', False)
            # Processing the call keyword arguments (line 299)
            kwargs_570 = {}
            # Getting the type of 'decode' (line 299)
            decode_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'decode', False)
            # Calling decode(args, kwargs) (line 299)
            decode_call_result_571 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), decode_564, *[list_call_result_568, verbose_569], **kwargs_570)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test' in the type store
    # Getting the type of 'stypy_return_type' (line 248)
    stypy_return_type_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_572)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test'
    return stypy_return_type_572

# Assigning a type to the variable 'test' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'test', test)

@norecursion
def hardertest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hardertest'
    module_type_store = module_type_store.open_function_context('hardertest', 302, 0, False)
    
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

    
    # Assigning a Call to a Name (line 304):
    
    # Call to open(...): (line 304)
    # Processing the call arguments (line 304)
    
    # Call to Relative(...): (line 304)
    # Processing the call arguments (line 304)
    str_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 30), 'str', 'testdata/BentCoinFile')
    # Processing the call keyword arguments (line 304)
    kwargs_576 = {}
    # Getting the type of 'Relative' (line 304)
    Relative_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 21), 'Relative', False)
    # Calling Relative(args, kwargs) (line 304)
    Relative_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 304, 21), Relative_574, *[str_575], **kwargs_576)
    
    str_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 56), 'str', 'r')
    # Processing the call keyword arguments (line 304)
    kwargs_579 = {}
    # Getting the type of 'open' (line 304)
    open_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'open', False)
    # Calling open(args, kwargs) (line 304)
    open_call_result_580 = invoke(stypy.reporting.localization.Localization(__file__, 304, 16), open_573, *[Relative_call_result_577, str_578], **kwargs_579)
    
    # Assigning a type to the variable 'inputfile' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'inputfile', open_call_result_580)
    
    # Assigning a Call to a Name (line 305):
    
    # Call to open(...): (line 305)
    # Processing the call arguments (line 305)
    
    # Call to Relative(...): (line 305)
    # Processing the call arguments (line 305)
    str_583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 31), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 305)
    kwargs_584 = {}
    # Getting the type of 'Relative' (line 305)
    Relative_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 305)
    Relative_call_result_585 = invoke(stypy.reporting.localization.Localization(__file__, 305, 22), Relative_582, *[str_583], **kwargs_584)
    
    str_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 43), 'str', 'w')
    # Processing the call keyword arguments (line 305)
    kwargs_587 = {}
    # Getting the type of 'open' (line 305)
    open_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 17), 'open', False)
    # Calling open(args, kwargs) (line 305)
    open_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 305, 17), open_581, *[Relative_call_result_585, str_586], **kwargs_587)
    
    # Assigning a type to the variable 'outputfile' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'outputfile', open_call_result_588)
    
    # Assigning a Call to a Name (line 308):
    
    # Call to encode(...): (line 308)
    # Processing the call arguments (line 308)
    
    # Call to read(...): (line 308)
    # Processing the call keyword arguments (line 308)
    kwargs_592 = {}
    # Getting the type of 'inputfile' (line 308)
    inputfile_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 308)
    read_591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 17), inputfile_590, 'read')
    # Calling read(args, kwargs) (line 308)
    read_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 308, 17), read_591, *[], **kwargs_592)
    
    int_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 35), 'int')
    int_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 38), 'int')
    # Processing the call keyword arguments (line 308)
    kwargs_596 = {}
    # Getting the type of 'encode' (line 308)
    encode_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 10), 'encode', False)
    # Calling encode(args, kwargs) (line 308)
    encode_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 308, 10), encode_589, *[read_call_result_593, int_594, int_595], **kwargs_596)
    
    # Assigning a type to the variable 'zip' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'zip', encode_call_result_597)
    
    # Call to write(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'zip' (line 309)
    zip_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 21), 'zip', False)
    # Processing the call keyword arguments (line 309)
    kwargs_601 = {}
    # Getting the type of 'outputfile' (line 309)
    outputfile_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 309)
    write_599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 4), outputfile_598, 'write')
    # Calling write(args, kwargs) (line 309)
    write_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 309, 4), write_599, *[zip_600], **kwargs_601)
    
    
    # Call to close(...): (line 310)
    # Processing the call keyword arguments (line 310)
    kwargs_605 = {}
    # Getting the type of 'outputfile' (line 310)
    outputfile_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 310)
    close_604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 4), outputfile_603, 'close')
    # Calling close(args, kwargs) (line 310)
    close_call_result_606 = invoke(stypy.reporting.localization.Localization(__file__, 310, 4), close_604, *[], **kwargs_605)
    
    
    # Call to close(...): (line 311)
    # Processing the call keyword arguments (line 311)
    kwargs_609 = {}
    # Getting the type of 'inputfile' (line 311)
    inputfile_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 311)
    close_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 4), inputfile_607, 'close')
    # Calling close(args, kwargs) (line 311)
    close_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 311, 4), close_608, *[], **kwargs_609)
    
    
    # Assigning a Call to a Name (line 314):
    
    # Call to open(...): (line 314)
    # Processing the call arguments (line 314)
    
    # Call to Relative(...): (line 314)
    # Processing the call arguments (line 314)
    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 30), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 314)
    kwargs_614 = {}
    # Getting the type of 'Relative' (line 314)
    Relative_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 21), 'Relative', False)
    # Calling Relative(args, kwargs) (line 314)
    Relative_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 314, 21), Relative_612, *[str_613], **kwargs_614)
    
    str_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 42), 'str', 'r')
    # Processing the call keyword arguments (line 314)
    kwargs_617 = {}
    # Getting the type of 'open' (line 314)
    open_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'open', False)
    # Calling open(args, kwargs) (line 314)
    open_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), open_611, *[Relative_call_result_615, str_616], **kwargs_617)
    
    # Assigning a type to the variable 'inputfile' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'inputfile', open_call_result_618)
    
    # Assigning a Call to a Name (line 315):
    
    # Call to open(...): (line 315)
    # Processing the call arguments (line 315)
    
    # Call to Relative(...): (line 315)
    # Processing the call arguments (line 315)
    str_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 31), 'str', 'tmp2')
    # Processing the call keyword arguments (line 315)
    kwargs_622 = {}
    # Getting the type of 'Relative' (line 315)
    Relative_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 315)
    Relative_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 315, 22), Relative_620, *[str_621], **kwargs_622)
    
    str_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 40), 'str', 'w')
    # Processing the call keyword arguments (line 315)
    kwargs_625 = {}
    # Getting the type of 'open' (line 315)
    open_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), 'open', False)
    # Calling open(args, kwargs) (line 315)
    open_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 315, 17), open_619, *[Relative_call_result_623, str_624], **kwargs_625)
    
    # Assigning a type to the variable 'outputfile' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'outputfile', open_call_result_626)
    
    # Assigning a Call to a Name (line 317):
    
    # Call to decode(...): (line 317)
    # Processing the call arguments (line 317)
    
    # Call to list(...): (line 317)
    # Processing the call arguments (line 317)
    
    # Call to read(...): (line 317)
    # Processing the call keyword arguments (line 317)
    kwargs_631 = {}
    # Getting the type of 'inputfile' (line 317)
    inputfile_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 22), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 317)
    read_630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 22), inputfile_629, 'read')
    # Calling read(args, kwargs) (line 317)
    read_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 317, 22), read_630, *[], **kwargs_631)
    
    # Processing the call keyword arguments (line 317)
    kwargs_633 = {}
    # Getting the type of 'list' (line 317)
    list_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'list', False)
    # Calling list(args, kwargs) (line 317)
    list_call_result_634 = invoke(stypy.reporting.localization.Localization(__file__, 317, 17), list_628, *[read_call_result_632], **kwargs_633)
    
    int_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 41), 'int')
    # Processing the call keyword arguments (line 317)
    kwargs_636 = {}
    # Getting the type of 'decode' (line 317)
    decode_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 10), 'decode', False)
    # Calling decode(args, kwargs) (line 317)
    decode_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 317, 10), decode_627, *[list_call_result_634, int_635], **kwargs_636)
    
    # Assigning a type to the variable 'unc' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'unc', decode_call_result_637)
    
    # Call to write(...): (line 318)
    # Processing the call arguments (line 318)
    # Getting the type of 'unc' (line 318)
    unc_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'unc', False)
    # Processing the call keyword arguments (line 318)
    kwargs_641 = {}
    # Getting the type of 'outputfile' (line 318)
    outputfile_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 318)
    write_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 4), outputfile_638, 'write')
    # Calling write(args, kwargs) (line 318)
    write_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 318, 4), write_639, *[unc_640], **kwargs_641)
    
    
    # Call to close(...): (line 319)
    # Processing the call keyword arguments (line 319)
    kwargs_645 = {}
    # Getting the type of 'outputfile' (line 319)
    outputfile_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 319)
    close_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 4), outputfile_643, 'close')
    # Calling close(args, kwargs) (line 319)
    close_call_result_646 = invoke(stypy.reporting.localization.Localization(__file__, 319, 4), close_644, *[], **kwargs_645)
    
    
    # Call to close(...): (line 320)
    # Processing the call keyword arguments (line 320)
    kwargs_649 = {}
    # Getting the type of 'inputfile' (line 320)
    inputfile_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 320)
    close_648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 4), inputfile_647, 'close')
    # Calling close(args, kwargs) (line 320)
    close_call_result_650 = invoke(stypy.reporting.localization.Localization(__file__, 320, 4), close_648, *[], **kwargs_649)
    
    
    # ################# End of 'hardertest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hardertest' in the type store
    # Getting the type of 'stypy_return_type' (line 302)
    stypy_return_type_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_651)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hardertest'
    return stypy_return_type_651

# Assigning a type to the variable 'hardertest' (line 302)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'hardertest', hardertest)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 329, 0, False)
    
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

    
    # Call to test(...): (line 330)
    # Processing the call keyword arguments (line 330)
    kwargs_653 = {}
    # Getting the type of 'test' (line 330)
    test_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'test', False)
    # Calling test(args, kwargs) (line 330)
    test_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 330, 4), test_652, *[], **kwargs_653)
    
    
    # Call to hardertest(...): (line 331)
    # Processing the call keyword arguments (line 331)
    kwargs_656 = {}
    # Getting the type of 'hardertest' (line 331)
    hardertest_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'hardertest', False)
    # Calling hardertest(args, kwargs) (line 331)
    hardertest_call_result_657 = invoke(stypy.reporting.localization.Localization(__file__, 331, 4), hardertest_655, *[], **kwargs_656)
    
    # Getting the type of 'True' (line 332)
    True_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type', True_658)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 329)
    stypy_return_type_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_659

# Assigning a type to the variable 'run' (line 329)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'run', run)

# Call to run(...): (line 335)
# Processing the call keyword arguments (line 335)
kwargs_661 = {}
# Getting the type of 'run' (line 335)
run_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'run', False)
# Calling run(args, kwargs) (line 335)
run_call_result_662 = invoke(stypy.reporting.localization.Localization(__file__, 335, 0), run_660, *[], **kwargs_661)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

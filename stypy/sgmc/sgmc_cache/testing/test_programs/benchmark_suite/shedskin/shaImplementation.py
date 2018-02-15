
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: # -*- coding: iso-8859-1
3: 
4: # Note that PyPy contains also a built-in module 'sha' which will hide
5: # this one if compiled in.
6: 
7: '''
8:    A sample implementation of SHA-1 in pure Python.
9:    
10:    Framework adapted from Dinu Gherman's MD5 implementation by
11:    J. Hallén and L. Creighton. SHA-1 implementation based directly on
12:    the text of the NIST standard FIPS PUB 180-1.
13: 
14:    ## IMPORTANT NOTE: compile with shedskin -l option (for long integers) 
15: '''
16: 
17: __date__ = '2004-11-17'
18: __version__ = 0.91  # Modernised by J. Hallén and L. Creighton for Pypy
19: __modified__ = []
20: __modified__.append('2011-05 adjusted by Emanuel Rumpf in order to compile with shedskin  ')
21: 
22: ##
23: ## IMPORTANT NOTE: compile with shedskin -l option (for long integers) 
24: ##
25: 
26: 
27: import struct
28: import copy
29: import sys
30: 
31: 
32: # ======================================================================
33: # Bit-Manipulation helpers
34: #
35: #   _long2bytes() was contributed by Barry Warsaw
36: #   and is reused here with tiny modifications.
37: # ======================================================================
38: 
39: def _long2bytesBigEndian(n, blocksize=0):
40:     '''Convert a long integer to a byte string.
41: 
42:     If optional blocksize is given and greater than zero, pad the front
43:     of the byte string with binary zeros so that the length is a multiple
44:     of blocksize.
45:     '''
46: 
47:     # After much testing, this algorithm was deemed to be the fastest.
48:     s = ''
49:     while n > 0:
50:         s = struct.pack('>I', n & 0xffffffffL) + s  # fmt: big endian, uint
51:         n = n >> 32
52: 
53:     # Strip off leading zeros.
54:     for i in range(len(s)):
55:         if s[i] <> '\000':
56:             break
57:     else:
58:         # Only happens when n == 0.
59:         s = '\000'
60:         i = 0
61: 
62:     s = s[i:]
63: 
64:     # Add back some pad bytes. This could be done more efficiently
65:     # w.r.t. the de-padding being done above, but sigh...
66:     if blocksize > 0 and len(s) % blocksize:
67:         s = (blocksize - len(s) % blocksize) * '\000' + s
68: 
69:     return s
70: 
71: 
72: def _bytelist2longBigEndian(list):
73:     "Transform a list of characters into a list of longs."
74: 
75:     imax = len(list) / 4
76:     hl = [0L] * imax
77: 
78:     j = 0
79:     i = 0
80:     while i < imax:
81:         b0 = ord(list[j]) << 24
82:         b1 = ord(list[j + 1]) << 16
83:         b2 = ord(list[j + 2]) << 8
84:         b3 = ord(list[j + 3])
85:         hl[i] = b0 | b1 | b2 | b3
86:         i = i + 1
87:         j = j + 4
88: 
89:     return hl
90: 
91: 
92: def _rotateLeft(x, n):
93:     "Rotate x (32 bit) left n bits circularly."
94: 
95:     return (x << n) | (x >> (32 - n))
96: 
97: 
98: # ======================================================================
99: # The SHA transformation functions
100: #
101: # ======================================================================
102: 
103: def f0_19(B, C, D):
104:     return (B & C) | ((~ B) & D)
105: 
106: 
107: def f20_39(B, C, D):
108:     return B ^ C ^ D
109: 
110: 
111: def f40_59(B, C, D):
112:     return (B & C) | (B & D) | (C & D)
113: 
114: 
115: def f60_79(B, C, D):
116:     return B ^ C ^ D
117: 
118: 
119: # fnc = [f0_19, f20_39, f40_59, f60_79]
120: 
121: # Constants to be used
122: K = [
123:     0x5A827999L,  # ( 0 <= t <= 19)
124:     0x6ED9EBA1L,  # (20 <= t <= 39)
125:     0x8F1BBCDCL,  # (40 <= t <= 59)
126:     0xCA62C1D6L  # (60 <= t <= 79)
127: ]
128: 
129: 
130: # ======================================================================
131: #
132: # start  sha class
133: #
134: # ======================================================================
135: 
136: class sha:
137:     "An implementation of the MD5 hash function in pure Python."
138: 
139:     digest_size = 20
140:     digestsize = 20
141: 
142:     def __init__(self):
143:         "Initialisation."
144: 
145:         # Initial message length in bits(!).
146:         self.length = 0L
147:         self.count = [0, 0]
148: 
149:         # Initial empty message as a sequence of bytes (8 bit characters).
150:         self.input = []
151: 
152:         # Call a separate init function, that can be used repeatedly
153:         # to start from scratch on the same object.
154:         self.init()
155: 
156:     def init(self):
157:         "Initialize the message-digest and set all fields to zero."
158: 
159:         self.length = 0L
160:         self.input = []
161: 
162:         # Initial 160 bit message digest (5 times 32 bit).
163:         self.H0 = 0x67452301L
164:         self.H1 = 0xEFCDAB89L
165:         self.H2 = 0x98BADCFEL
166:         self.H3 = 0x10325476L
167:         self.H4 = 0xC3D2E1F0L
168: 
169:     def _transform(self, W):
170: 
171:         for t in range(16, 80):
172:             W.append(_rotateLeft(
173:                 W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16], 1) & 0xffffffffL)
174: 
175:         A = self.H0
176:         B = self.H1
177:         C = self.H2
178:         D = self.H3
179:         E = self.H4
180: 
181:         '''
182:         This loop was unrolled to gain about 10% in speed
183:         for t in range(0, 80):
184:             TEMP = _rotateLeft(A, 5) + f[t/20] + E + W[t] + K[t/20]
185:             E = D
186:             D = C
187:             C = _rotateLeft(B, 30) & 0xffffffffL
188:             B = A
189:             A = TEMP & 0xffffffffL
190:         '''
191: 
192:         for t in range(0, 20):
193:             TEMP = _rotateLeft(A, 5) + ((B & C) | ((~ B) & D)) + E + W[t] + K[0]
194:             E = D
195:             D = C
196:             C = _rotateLeft(B, 30) & 0xffffffffL
197:             B = A
198:             A = TEMP & 0xffffffffL
199: 
200:         for t in range(20, 40):
201:             TEMP = _rotateLeft(A, 5) + (B ^ C ^ D) + E + W[t] + K[1]
202:             E = D
203:             D = C
204:             C = _rotateLeft(B, 30) & 0xffffffffL
205:             B = A
206:             A = TEMP & 0xffffffffL
207: 
208:         for t in range(40, 60):
209:             TEMP = _rotateLeft(A, 5) + ((B & C) | (B & D) | (C & D)) + E + W[t] + K[2]
210:             E = D
211:             D = C
212:             C = _rotateLeft(B, 30) & 0xffffffffL
213:             B = A
214:             A = TEMP & 0xffffffffL
215: 
216:         for t in range(60, 80):
217:             TEMP = _rotateLeft(A, 5) + (B ^ C ^ D) + E + W[t] + K[3]
218:             E = D
219:             D = C
220:             C = _rotateLeft(B, 30) & 0xffffffffL
221:             B = A
222:             A = TEMP & 0xffffffffL
223: 
224:         self.H0 = (self.H0 + A) & 0xffffffffL
225:         self.H1 = (self.H1 + B) & 0xffffffffL
226:         self.H2 = (self.H2 + C) & 0xffffffffL
227:         self.H3 = (self.H3 + D) & 0xffffffffL
228:         self.H4 = (self.H4 + E) & 0xffffffffL
229: 
230:     # Down from here all methods follow the Python Standard Library
231:     # API of the sha module.
232: 
233:     def update(self, inBufseq):
234:         '''Add to the current message.
235: 
236:         Update the md5 object with the string arg. Repeated calls
237:         are equivalent to a single call with the concatenation of all
238:         the arguments, i.e. m.update(a); m.update(b) is equivalent
239:         to m.update(a+b).
240: 
241:         The hash is immediately calculated for all full blocks. The final
242:         calculation is made in digest(). It will calculate 1-2 blocks,
243:         depending on how much padding we have to add. This allows us to
244:         keep an intermediate value for the hash, so that we only need to
245:         make minimal recalculation if we call update() to add more data
246:         to the hashed string.
247:         '''
248: 
249:         inBuf = list(inBufseq)  # make it a list
250: 
251:         leninBuf = len(inBuf)
252: 
253:         # Compute number of bytes mod 64.
254:         index = (self.count[1] >> 3) & 0x3FL
255: 
256:         # Update number of bits.
257:         self.count[1] = self.count[1] + (leninBuf << 3)
258:         if self.count[1] < (leninBuf << 3):
259:             self.count[0] = self.count[0] + 1
260:         self.count[0] = self.count[0] + (leninBuf >> 29)
261: 
262:         partLen = 64 - index
263: 
264:         if leninBuf >= partLen:
265:             self.input[index:] = list(inBuf[:partLen])
266: 
267:             self._transform(_bytelist2longBigEndian(self.input))
268:             i = partLen
269:             while i + 63 < leninBuf:
270:                 self._transform(_bytelist2longBigEndian(list(inBuf[i:i + 64])))
271: 
272:                 i = i + 64
273:             else:
274:                 self.input = list(inBuf[i:leninBuf])
275:         else:
276:             i = 0
277:             self.input = self.input + list(inBuf)
278: 
279:     def digest(self):
280:         '''Terminate the message-digest computation and return digest.
281: 
282:         Return the digest of the strings passed to the update()
283:         method so far. This is a 16-byte string which may contain
284:         non-ASCII characters, including null bytes.
285:         '''
286: 
287:         H0 = self.H0
288:         H1 = self.H1
289:         H2 = self.H2
290:         H3 = self.H3
291:         H4 = self.H4
292:         input = [] + self.input
293:         count = [] + self.count
294: 
295:         index = (self.count[1] >> 3) & 0x3fL
296: 
297:         if index < 56:
298:             padLen = 56 - index
299:         else:
300:             padLen = 120 - index
301: 
302:         padding = ['\200'] + ['\000'] * 63
303:         self.update(padding[:padLen])
304: 
305:         # Append length (before padding).
306:         bits = _bytelist2longBigEndian(self.input[:56]) + count
307: 
308:         self._transform(bits)
309: 
310:         # Store state in digest.
311:         digest = _long2bytesBigEndian(self.H0, 4) + \
312:                  _long2bytesBigEndian(self.H1, 4) + \
313:                  _long2bytesBigEndian(self.H2, 4) + \
314:                  _long2bytesBigEndian(self.H3, 4) + \
315:                  _long2bytesBigEndian(self.H4, 4)
316: 
317:         self.H0 = H0
318:         self.H1 = H1
319:         self.H2 = H2
320:         self.H3 = H3
321:         self.H4 = H4
322:         self.input = input
323:         self.count = count
324: 
325:         return digest
326: 
327:     def hexdigest(self):
328:         '''Terminate and return digest in HEX form.
329: 
330:         Like digest() except the digest is returned as a string of
331:         length 32, containing only hexadecimal digits. This may be
332:         used to exchange the value safely in email or other non-
333:         binary environments.
334:         '''
335:         return ''.join(['%02x' % ord(c) for c in self.digest()])
336: 
337:     def copy(self):
338:         '''Return a clone object.
339: 
340:         Return a copy ('clone') of the md5 object. This can be used
341:         to efficiently compute the digests of strings that share
342:         a common initial substring.
343:         '''
344: 
345:         return copy.deepcopy(self)
346: 
347: 
348: # ======================================================================
349: # Mimic Python top-level functions from standard library API
350: # for consistency with the md5 module of the standard library.
351: # ======================================================================
352: 
353: # These are mandatory variables in the module. They have constant values
354: # in the SHA standard.
355: 
356: digest_size = 20
357: digestsize = 20
358: blocksize = 1
359: 
360: 
361: def new(arg=None):
362:     '''Return a new sha crypto object.
363: 
364:     If arg is present, the method call update(arg) is made.
365:     '''
366: 
367:     crypto = sha()
368:     if arg:
369:         crypto.update(arg)
370: 
371:     return crypto
372: 
373: 
374: # ======================================================================
375: # MAIN
376: # ======================================================================
377: 
378: def main():
379:     ##    if len( sys.argv ) <= 1:
380:     ##        print ''
381:     ##        print 'No string found. Add predefined text. '
382:     text = '''Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'''
383:     ##    else:
384:     ##        text = sys.argv[1]
385: 
386:     shah = new(text)
387:     ##
388:     ##    print ''
389:     ##    print text, shah.hexdigest()
390:     ##    print ''
391:     shah.hexdigest()
392: 
393:     if 1:
394:         # allows shedskin type inference, don't remove
395:         shah.copy()
396:         B = 0x67452301L
397:         C = 0x67452301L
398:         D = 0x67452301L
399: 
400:         f0_19(B, C, D)
401:         f20_39(B, C, D)
402:         f40_59(B, C, D)
403:         f60_79(B, C, D)
404: 
405: 
406: def run():
407:     for i in range(250):
408:         main()
409:     return True
410: 
411: 
412: run()
413: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', "\n   A sample implementation of SHA-1 in pure Python.\n   \n   Framework adapted from Dinu Gherman's MD5 implementation by\n   J. Hall\xc3\xa9n and L. Creighton. SHA-1 implementation based directly on\n   the text of the NIST standard FIPS PUB 180-1.\n\n   ## IMPORTANT NOTE: compile with shedskin -l option (for long integers) \n")

# Assigning a Str to a Name (line 17):
str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', '2004-11-17')
# Assigning a type to the variable '__date__' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__date__', str_2)

# Assigning a Num to a Name (line 18):
float_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'float')
# Assigning a type to the variable '__version__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__version__', float_3)

# Assigning a List to a Name (line 19):

# Obtaining an instance of the builtin type 'list' (line 19)
list_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)

# Assigning a type to the variable '__modified__' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__modified__', list_4)

# Call to append(...): (line 20)
# Processing the call arguments (line 20)
str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'str', '2011-05 adjusted by Emanuel Rumpf in order to compile with shedskin  ')
# Processing the call keyword arguments (line 20)
kwargs_8 = {}
# Getting the type of '__modified__' (line 20)
modified___5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '__modified__', False)
# Obtaining the member 'append' of a type (line 20)
append_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), modified___5, 'append')
# Calling append(args, kwargs) (line 20)
append_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 20, 0), append_6, *[str_7], **kwargs_8)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import struct' statement (line 27)
import struct

import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'struct', struct, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import copy' statement (line 28)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import sys' statement (line 29)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'sys', sys, module_type_store)


@norecursion
def _long2bytesBigEndian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 38), 'int')
    defaults = [int_10]
    # Create a new context for function '_long2bytesBigEndian'
    module_type_store = module_type_store.open_function_context('_long2bytesBigEndian', 39, 0, False)
    
    # Passed parameters checking function
    _long2bytesBigEndian.stypy_localization = localization
    _long2bytesBigEndian.stypy_type_of_self = None
    _long2bytesBigEndian.stypy_type_store = module_type_store
    _long2bytesBigEndian.stypy_function_name = '_long2bytesBigEndian'
    _long2bytesBigEndian.stypy_param_names_list = ['n', 'blocksize']
    _long2bytesBigEndian.stypy_varargs_param_name = None
    _long2bytesBigEndian.stypy_kwargs_param_name = None
    _long2bytesBigEndian.stypy_call_defaults = defaults
    _long2bytesBigEndian.stypy_call_varargs = varargs
    _long2bytesBigEndian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_long2bytesBigEndian', ['n', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_long2bytesBigEndian', localization, ['n', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_long2bytesBigEndian(...)' code ##################

    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', 'Convert a long integer to a byte string.\n\n    If optional blocksize is given and greater than zero, pad the front\n    of the byte string with binary zeros so that the length is a multiple\n    of blocksize.\n    ')
    
    # Assigning a Str to a Name (line 48):
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'str', '')
    # Assigning a type to the variable 's' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 's', str_12)
    
    
    # Getting the type of 'n' (line 49)
    n_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'n')
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 14), 'int')
    # Applying the binary operator '>' (line 49)
    result_gt_15 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 10), '>', n_13, int_14)
    
    # Assigning a type to the variable 'result_gt_15' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'result_gt_15', result_gt_15)
    # Testing if the while is going to be iterated (line 49)
    # Testing the type of an if condition (line 49)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), result_gt_15)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 49, 4), result_gt_15):
        # SSA begins for while statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 50):
        
        # Call to pack(...): (line 50)
        # Processing the call arguments (line 50)
        str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 24), 'str', '>I')
        # Getting the type of 'n' (line 50)
        n_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'n', False)
        long_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'long')
        # Applying the binary operator '&' (line 50)
        result_and__21 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 30), '&', n_19, long_20)
        
        # Processing the call keyword arguments (line 50)
        kwargs_22 = {}
        # Getting the type of 'struct' (line 50)
        struct_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'struct', False)
        # Obtaining the member 'pack' of a type (line 50)
        pack_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), struct_16, 'pack')
        # Calling pack(args, kwargs) (line 50)
        pack_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), pack_17, *[str_18, result_and__21], **kwargs_22)
        
        # Getting the type of 's' (line 50)
        s_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 49), 's')
        # Applying the binary operator '+' (line 50)
        result_add_25 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), '+', pack_call_result_23, s_24)
        
        # Assigning a type to the variable 's' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 's', result_add_25)
        
        # Assigning a BinOp to a Name (line 51):
        # Getting the type of 'n' (line 51)
        n_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'n')
        int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'int')
        # Applying the binary operator '>>' (line 51)
        result_rshift_28 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 12), '>>', n_26, int_27)
        
        # Assigning a type to the variable 'n' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'n', result_rshift_28)
        # SSA join for while statement (line 49)
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to range(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Call to len(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 's' (line 54)
    s_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 's', False)
    # Processing the call keyword arguments (line 54)
    kwargs_32 = {}
    # Getting the type of 'len' (line 54)
    len_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'len', False)
    # Calling len(args, kwargs) (line 54)
    len_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), len_30, *[s_31], **kwargs_32)
    
    # Processing the call keyword arguments (line 54)
    kwargs_34 = {}
    # Getting the type of 'range' (line 54)
    range_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'range', False)
    # Calling range(args, kwargs) (line 54)
    range_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), range_29, *[len_call_result_33], **kwargs_34)
    
    # Assigning a type to the variable 'range_call_result_35' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'range_call_result_35', range_call_result_35)
    # Testing if the for loop is going to be iterated (line 54)
    # Testing the type of a for loop iterable (line 54)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 4), range_call_result_35)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 4), range_call_result_35):
        # Getting the type of the for loop variable (line 54)
        for_loop_var_36 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 4), range_call_result_35)
        # Assigning a type to the variable 'i' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'i', for_loop_var_36)
        # SSA begins for a for statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 55)
        i_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'i')
        # Getting the type of 's' (line 55)
        s_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 's')
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), s_38, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), getitem___39, i_37)
        
        str_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'str', '\x00')
        # Applying the binary operator '!=' (line 55)
        result_ne_42 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 11), '!=', subscript_call_result_40, str_41)
        
        # Testing if the type of an if condition is none (line 55)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 8), result_ne_42):
            pass
        else:
            
            # Testing the type of an if condition (line 55)
            if_condition_43 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_ne_42)
            # Assigning a type to the variable 'if_condition_43' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_43', if_condition_43)
            # SSA begins for if statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of a for statement (line 54)
        module_type_store.open_ssa_branch('for loop else')
        
        # Assigning a Str to a Name (line 59):
        str_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'str', '\x00')
        # Assigning a type to the variable 's' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 's', str_44)
        
        # Assigning a Num to a Name (line 60):
        int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'int')
        # Assigning a type to the variable 'i' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'i', int_45)
    else:
        
        # Assigning a Str to a Name (line 59):
        str_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'str', '\x00')
        # Assigning a type to the variable 's' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 's', str_44)
        
        # Assigning a Num to a Name (line 60):
        int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'int')
        # Assigning a type to the variable 'i' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'i', int_45)

    
    
    # Assigning a Subscript to a Name (line 62):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 62)
    i_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 10), 'i')
    slice_47 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 62, 8), i_46, None, None)
    # Getting the type of 's' (line 62)
    s_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 's')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), s_48, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___49, slice_47)
    
    # Assigning a type to the variable 's' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 's', subscript_call_result_50)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'blocksize' (line 66)
    blocksize_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'blocksize')
    int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'int')
    # Applying the binary operator '>' (line 66)
    result_gt_53 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), '>', blocksize_51, int_52)
    
    
    # Call to len(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 's' (line 66)
    s_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 's', False)
    # Processing the call keyword arguments (line 66)
    kwargs_56 = {}
    # Getting the type of 'len' (line 66)
    len_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'len', False)
    # Calling len(args, kwargs) (line 66)
    len_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), len_54, *[s_55], **kwargs_56)
    
    # Getting the type of 'blocksize' (line 66)
    blocksize_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'blocksize')
    # Applying the binary operator '%' (line 66)
    result_mod_59 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 25), '%', len_call_result_57, blocksize_58)
    
    # Applying the binary operator 'and' (line 66)
    result_and_keyword_60 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), 'and', result_gt_53, result_mod_59)
    
    # Testing if the type of an if condition is none (line 66)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 4), result_and_keyword_60):
        pass
    else:
        
        # Testing the type of an if condition (line 66)
        if_condition_61 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_and_keyword_60)
        # Assigning a type to the variable 'if_condition_61' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_61', if_condition_61)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 67):
        # Getting the type of 'blocksize' (line 67)
        blocksize_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'blocksize')
        
        # Call to len(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 's' (line 67)
        s_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 's', False)
        # Processing the call keyword arguments (line 67)
        kwargs_65 = {}
        # Getting the type of 'len' (line 67)
        len_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'len', False)
        # Calling len(args, kwargs) (line 67)
        len_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 67, 25), len_63, *[s_64], **kwargs_65)
        
        # Getting the type of 'blocksize' (line 67)
        blocksize_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'blocksize')
        # Applying the binary operator '%' (line 67)
        result_mod_68 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '%', len_call_result_66, blocksize_67)
        
        # Applying the binary operator '-' (line 67)
        result_sub_69 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 13), '-', blocksize_62, result_mod_68)
        
        str_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 47), 'str', '\x00')
        # Applying the binary operator '*' (line 67)
        result_mul_71 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 12), '*', result_sub_69, str_70)
        
        # Getting the type of 's' (line 67)
        s_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 56), 's')
        # Applying the binary operator '+' (line 67)
        result_add_73 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 12), '+', result_mul_71, s_72)
        
        # Assigning a type to the variable 's' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 's', result_add_73)
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 's' (line 69)
    s_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type', s_74)
    
    # ################# End of '_long2bytesBigEndian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_long2bytesBigEndian' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_75)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_long2bytesBigEndian'
    return stypy_return_type_75

# Assigning a type to the variable '_long2bytesBigEndian' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_long2bytesBigEndian', _long2bytesBigEndian)

@norecursion
def _bytelist2longBigEndian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_bytelist2longBigEndian'
    module_type_store = module_type_store.open_function_context('_bytelist2longBigEndian', 72, 0, False)
    
    # Passed parameters checking function
    _bytelist2longBigEndian.stypy_localization = localization
    _bytelist2longBigEndian.stypy_type_of_self = None
    _bytelist2longBigEndian.stypy_type_store = module_type_store
    _bytelist2longBigEndian.stypy_function_name = '_bytelist2longBigEndian'
    _bytelist2longBigEndian.stypy_param_names_list = ['list']
    _bytelist2longBigEndian.stypy_varargs_param_name = None
    _bytelist2longBigEndian.stypy_kwargs_param_name = None
    _bytelist2longBigEndian.stypy_call_defaults = defaults
    _bytelist2longBigEndian.stypy_call_varargs = varargs
    _bytelist2longBigEndian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_bytelist2longBigEndian', ['list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_bytelist2longBigEndian', localization, ['list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_bytelist2longBigEndian(...)' code ##################

    str_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'Transform a list of characters into a list of longs.')
    
    # Assigning a BinOp to a Name (line 75):
    
    # Call to len(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'list' (line 75)
    list_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'list', False)
    # Processing the call keyword arguments (line 75)
    kwargs_79 = {}
    # Getting the type of 'len' (line 75)
    len_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'len', False)
    # Calling len(args, kwargs) (line 75)
    len_call_result_80 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), len_77, *[list_78], **kwargs_79)
    
    int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'int')
    # Applying the binary operator 'div' (line 75)
    result_div_82 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), 'div', len_call_result_80, int_81)
    
    # Assigning a type to the variable 'imax' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'imax', result_div_82)
    
    # Assigning a BinOp to a Name (line 76):
    
    # Obtaining an instance of the builtin type 'list' (line 76)
    list_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 76)
    # Adding element type (line 76)
    long_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 10), 'long')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), list_83, long_84)
    
    # Getting the type of 'imax' (line 76)
    imax_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'imax')
    # Applying the binary operator '*' (line 76)
    result_mul_86 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 9), '*', list_83, imax_85)
    
    # Assigning a type to the variable 'hl' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'hl', result_mul_86)
    
    # Assigning a Num to a Name (line 78):
    int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'int')
    # Assigning a type to the variable 'j' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'j', int_87)
    
    # Assigning a Num to a Name (line 79):
    int_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'int')
    # Assigning a type to the variable 'i' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'i', int_88)
    
    
    # Getting the type of 'i' (line 80)
    i_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 10), 'i')
    # Getting the type of 'imax' (line 80)
    imax_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'imax')
    # Applying the binary operator '<' (line 80)
    result_lt_91 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 10), '<', i_89, imax_90)
    
    # Assigning a type to the variable 'result_lt_91' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'result_lt_91', result_lt_91)
    # Testing if the while is going to be iterated (line 80)
    # Testing the type of an if condition (line 80)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_lt_91)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 80, 4), result_lt_91):
        # SSA begins for while statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 81):
        
        # Call to ord(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 81)
        j_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'j', False)
        # Getting the type of 'list' (line 81)
        list_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'list', False)
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___95 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), list_94, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_96 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), getitem___95, j_93)
        
        # Processing the call keyword arguments (line 81)
        kwargs_97 = {}
        # Getting the type of 'ord' (line 81)
        ord_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'ord', False)
        # Calling ord(args, kwargs) (line 81)
        ord_call_result_98 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), ord_92, *[subscript_call_result_96], **kwargs_97)
        
        int_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 29), 'int')
        # Applying the binary operator '<<' (line 81)
        result_lshift_100 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 13), '<<', ord_call_result_98, int_99)
        
        # Assigning a type to the variable 'b0' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'b0', result_lshift_100)
        
        # Assigning a BinOp to a Name (line 82):
        
        # Call to ord(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 82)
        j_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'j', False)
        int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'int')
        # Applying the binary operator '+' (line 82)
        result_add_104 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 22), '+', j_102, int_103)
        
        # Getting the type of 'list' (line 82)
        list_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'list', False)
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 17), list_105, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), getitem___106, result_add_104)
        
        # Processing the call keyword arguments (line 82)
        kwargs_108 = {}
        # Getting the type of 'ord' (line 82)
        ord_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'ord', False)
        # Calling ord(args, kwargs) (line 82)
        ord_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), ord_101, *[subscript_call_result_107], **kwargs_108)
        
        int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 33), 'int')
        # Applying the binary operator '<<' (line 82)
        result_lshift_111 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 13), '<<', ord_call_result_109, int_110)
        
        # Assigning a type to the variable 'b1' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'b1', result_lshift_111)
        
        # Assigning a BinOp to a Name (line 83):
        
        # Call to ord(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 83)
        j_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'j', False)
        int_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'int')
        # Applying the binary operator '+' (line 83)
        result_add_115 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 22), '+', j_113, int_114)
        
        # Getting the type of 'list' (line 83)
        list_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'list', False)
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), list_116, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), getitem___117, result_add_115)
        
        # Processing the call keyword arguments (line 83)
        kwargs_119 = {}
        # Getting the type of 'ord' (line 83)
        ord_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'ord', False)
        # Calling ord(args, kwargs) (line 83)
        ord_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), ord_112, *[subscript_call_result_118], **kwargs_119)
        
        int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'int')
        # Applying the binary operator '<<' (line 83)
        result_lshift_122 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 13), '<<', ord_call_result_120, int_121)
        
        # Assigning a type to the variable 'b2' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'b2', result_lshift_122)
        
        # Assigning a Call to a Name (line 84):
        
        # Call to ord(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 84)
        j_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'j', False)
        int_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 26), 'int')
        # Applying the binary operator '+' (line 84)
        result_add_126 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 22), '+', j_124, int_125)
        
        # Getting the type of 'list' (line 84)
        list_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'list', False)
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 17), list_127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), getitem___128, result_add_126)
        
        # Processing the call keyword arguments (line 84)
        kwargs_130 = {}
        # Getting the type of 'ord' (line 84)
        ord_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'ord', False)
        # Calling ord(args, kwargs) (line 84)
        ord_call_result_131 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), ord_123, *[subscript_call_result_129], **kwargs_130)
        
        # Assigning a type to the variable 'b3' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'b3', ord_call_result_131)
        
        # Assigning a BinOp to a Subscript (line 85):
        # Getting the type of 'b0' (line 85)
        b0_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'b0')
        # Getting the type of 'b1' (line 85)
        b1_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'b1')
        # Applying the binary operator '|' (line 85)
        result_or__134 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 16), '|', b0_132, b1_133)
        
        # Getting the type of 'b2' (line 85)
        b2_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'b2')
        # Applying the binary operator '|' (line 85)
        result_or__136 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 24), '|', result_or__134, b2_135)
        
        # Getting the type of 'b3' (line 85)
        b3_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'b3')
        # Applying the binary operator '|' (line 85)
        result_or__138 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 29), '|', result_or__136, b3_137)
        
        # Getting the type of 'hl' (line 85)
        hl_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'hl')
        # Getting the type of 'i' (line 85)
        i_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'i')
        # Storing an element on a container (line 85)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), hl_139, (i_140, result_or__138))
        
        # Assigning a BinOp to a Name (line 86):
        # Getting the type of 'i' (line 86)
        i_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'i')
        int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'int')
        # Applying the binary operator '+' (line 86)
        result_add_143 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 12), '+', i_141, int_142)
        
        # Assigning a type to the variable 'i' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'i', result_add_143)
        
        # Assigning a BinOp to a Name (line 87):
        # Getting the type of 'j' (line 87)
        j_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'j')
        int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'int')
        # Applying the binary operator '+' (line 87)
        result_add_146 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 12), '+', j_144, int_145)
        
        # Assigning a type to the variable 'j' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'j', result_add_146)
        # SSA join for while statement (line 80)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'hl' (line 89)
    hl_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'hl')
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', hl_147)
    
    # ################# End of '_bytelist2longBigEndian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_bytelist2longBigEndian' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_148)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_bytelist2longBigEndian'
    return stypy_return_type_148

# Assigning a type to the variable '_bytelist2longBigEndian' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), '_bytelist2longBigEndian', _bytelist2longBigEndian)

@norecursion
def _rotateLeft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_rotateLeft'
    module_type_store = module_type_store.open_function_context('_rotateLeft', 92, 0, False)
    
    # Passed parameters checking function
    _rotateLeft.stypy_localization = localization
    _rotateLeft.stypy_type_of_self = None
    _rotateLeft.stypy_type_store = module_type_store
    _rotateLeft.stypy_function_name = '_rotateLeft'
    _rotateLeft.stypy_param_names_list = ['x', 'n']
    _rotateLeft.stypy_varargs_param_name = None
    _rotateLeft.stypy_kwargs_param_name = None
    _rotateLeft.stypy_call_defaults = defaults
    _rotateLeft.stypy_call_varargs = varargs
    _rotateLeft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_rotateLeft', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_rotateLeft', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_rotateLeft(...)' code ##################

    str_149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'str', 'Rotate x (32 bit) left n bits circularly.')
    # Getting the type of 'x' (line 95)
    x_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'x')
    # Getting the type of 'n' (line 95)
    n_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'n')
    # Applying the binary operator '<<' (line 95)
    result_lshift_152 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), '<<', x_150, n_151)
    
    # Getting the type of 'x' (line 95)
    x_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'x')
    int_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'int')
    # Getting the type of 'n' (line 95)
    n_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'n')
    # Applying the binary operator '-' (line 95)
    result_sub_156 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 29), '-', int_154, n_155)
    
    # Applying the binary operator '>>' (line 95)
    result_rshift_157 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 23), '>>', x_153, result_sub_156)
    
    # Applying the binary operator '|' (line 95)
    result_or__158 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), '|', result_lshift_152, result_rshift_157)
    
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', result_or__158)
    
    # ################# End of '_rotateLeft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_rotateLeft' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_159)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_rotateLeft'
    return stypy_return_type_159

# Assigning a type to the variable '_rotateLeft' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), '_rotateLeft', _rotateLeft)

@norecursion
def f0_19(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f0_19'
    module_type_store = module_type_store.open_function_context('f0_19', 103, 0, False)
    
    # Passed parameters checking function
    f0_19.stypy_localization = localization
    f0_19.stypy_type_of_self = None
    f0_19.stypy_type_store = module_type_store
    f0_19.stypy_function_name = 'f0_19'
    f0_19.stypy_param_names_list = ['B', 'C', 'D']
    f0_19.stypy_varargs_param_name = None
    f0_19.stypy_kwargs_param_name = None
    f0_19.stypy_call_defaults = defaults
    f0_19.stypy_call_varargs = varargs
    f0_19.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f0_19', ['B', 'C', 'D'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f0_19', localization, ['B', 'C', 'D'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f0_19(...)' code ##################

    # Getting the type of 'B' (line 104)
    B_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'B')
    # Getting the type of 'C' (line 104)
    C_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'C')
    # Applying the binary operator '&' (line 104)
    result_and__162 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 12), '&', B_160, C_161)
    
    
    # Getting the type of 'B' (line 104)
    B_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'B')
    # Applying the '~' unary operator (line 104)
    result_inv_164 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 23), '~', B_163)
    
    # Getting the type of 'D' (line 104)
    D_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'D')
    # Applying the binary operator '&' (line 104)
    result_and__166 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), '&', result_inv_164, D_165)
    
    # Applying the binary operator '|' (line 104)
    result_or__167 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), '|', result_and__162, result_and__166)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', result_or__167)
    
    # ################# End of 'f0_19(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f0_19' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f0_19'
    return stypy_return_type_168

# Assigning a type to the variable 'f0_19' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'f0_19', f0_19)

@norecursion
def f20_39(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f20_39'
    module_type_store = module_type_store.open_function_context('f20_39', 107, 0, False)
    
    # Passed parameters checking function
    f20_39.stypy_localization = localization
    f20_39.stypy_type_of_self = None
    f20_39.stypy_type_store = module_type_store
    f20_39.stypy_function_name = 'f20_39'
    f20_39.stypy_param_names_list = ['B', 'C', 'D']
    f20_39.stypy_varargs_param_name = None
    f20_39.stypy_kwargs_param_name = None
    f20_39.stypy_call_defaults = defaults
    f20_39.stypy_call_varargs = varargs
    f20_39.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f20_39', ['B', 'C', 'D'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f20_39', localization, ['B', 'C', 'D'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f20_39(...)' code ##################

    # Getting the type of 'B' (line 108)
    B_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'B')
    # Getting the type of 'C' (line 108)
    C_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'C')
    # Applying the binary operator '^' (line 108)
    result_xor_171 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), '^', B_169, C_170)
    
    # Getting the type of 'D' (line 108)
    D_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'D')
    # Applying the binary operator '^' (line 108)
    result_xor_173 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 17), '^', result_xor_171, D_172)
    
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', result_xor_173)
    
    # ################# End of 'f20_39(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f20_39' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f20_39'
    return stypy_return_type_174

# Assigning a type to the variable 'f20_39' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'f20_39', f20_39)

@norecursion
def f40_59(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f40_59'
    module_type_store = module_type_store.open_function_context('f40_59', 111, 0, False)
    
    # Passed parameters checking function
    f40_59.stypy_localization = localization
    f40_59.stypy_type_of_self = None
    f40_59.stypy_type_store = module_type_store
    f40_59.stypy_function_name = 'f40_59'
    f40_59.stypy_param_names_list = ['B', 'C', 'D']
    f40_59.stypy_varargs_param_name = None
    f40_59.stypy_kwargs_param_name = None
    f40_59.stypy_call_defaults = defaults
    f40_59.stypy_call_varargs = varargs
    f40_59.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f40_59', ['B', 'C', 'D'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f40_59', localization, ['B', 'C', 'D'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f40_59(...)' code ##################

    # Getting the type of 'B' (line 112)
    B_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'B')
    # Getting the type of 'C' (line 112)
    C_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'C')
    # Applying the binary operator '&' (line 112)
    result_and__177 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 12), '&', B_175, C_176)
    
    # Getting the type of 'B' (line 112)
    B_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'B')
    # Getting the type of 'D' (line 112)
    D_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'D')
    # Applying the binary operator '&' (line 112)
    result_and__180 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 22), '&', B_178, D_179)
    
    # Applying the binary operator '|' (line 112)
    result_or__181 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), '|', result_and__177, result_and__180)
    
    # Getting the type of 'C' (line 112)
    C_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'C')
    # Getting the type of 'D' (line 112)
    D_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'D')
    # Applying the binary operator '&' (line 112)
    result_and__184 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 32), '&', C_182, D_183)
    
    # Applying the binary operator '|' (line 112)
    result_or__185 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 29), '|', result_or__181, result_and__184)
    
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', result_or__185)
    
    # ################# End of 'f40_59(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f40_59' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f40_59'
    return stypy_return_type_186

# Assigning a type to the variable 'f40_59' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'f40_59', f40_59)

@norecursion
def f60_79(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f60_79'
    module_type_store = module_type_store.open_function_context('f60_79', 115, 0, False)
    
    # Passed parameters checking function
    f60_79.stypy_localization = localization
    f60_79.stypy_type_of_self = None
    f60_79.stypy_type_store = module_type_store
    f60_79.stypy_function_name = 'f60_79'
    f60_79.stypy_param_names_list = ['B', 'C', 'D']
    f60_79.stypy_varargs_param_name = None
    f60_79.stypy_kwargs_param_name = None
    f60_79.stypy_call_defaults = defaults
    f60_79.stypy_call_varargs = varargs
    f60_79.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f60_79', ['B', 'C', 'D'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f60_79', localization, ['B', 'C', 'D'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f60_79(...)' code ##################

    # Getting the type of 'B' (line 116)
    B_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'B')
    # Getting the type of 'C' (line 116)
    C_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'C')
    # Applying the binary operator '^' (line 116)
    result_xor_189 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), '^', B_187, C_188)
    
    # Getting the type of 'D' (line 116)
    D_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'D')
    # Applying the binary operator '^' (line 116)
    result_xor_191 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '^', result_xor_189, D_190)
    
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', result_xor_191)
    
    # ################# End of 'f60_79(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f60_79' in the type store
    # Getting the type of 'stypy_return_type' (line 115)
    stypy_return_type_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f60_79'
    return stypy_return_type_192

# Assigning a type to the variable 'f60_79' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'f60_79', f60_79)

# Assigning a List to a Name (line 122):

# Obtaining an instance of the builtin type 'list' (line 122)
list_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 122)
# Adding element type (line 122)
long_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 4), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 4), list_193, long_194)
# Adding element type (line 122)
long_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 4), list_193, long_195)
# Adding element type (line 122)
long_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 4), list_193, long_196)
# Adding element type (line 122)
long_197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 4), list_193, long_197)

# Assigning a type to the variable 'K' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'K', list_193)
# Declaration of the 'sha' class

class sha:
    str_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 4), 'str', 'An implementation of the MD5 hash function in pure Python.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sha.__init__', [], None, None, defaults, varargs, kwargs)

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

        str_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 8), 'str', 'Initialisation.')
        
        # Assigning a Num to a Attribute (line 146):
        long_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 22), 'long')
        # Getting the type of 'self' (line 146)
        self_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'length' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_201, 'length', long_200)
        
        # Assigning a List to a Attribute (line 147):
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        int_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_202, int_203)
        # Adding element type (line 147)
        int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_202, int_204)
        
        # Getting the type of 'self' (line 147)
        self_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'count' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_205, 'count', list_202)
        
        # Assigning a List to a Attribute (line 150):
        
        # Obtaining an instance of the builtin type 'list' (line 150)
        list_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 150)
        
        # Getting the type of 'self' (line 150)
        self_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'input' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_207, 'input', list_206)
        
        # Call to init(...): (line 154)
        # Processing the call keyword arguments (line 154)
        kwargs_210 = {}
        # Getting the type of 'self' (line 154)
        self_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self', False)
        # Obtaining the member 'init' of a type (line 154)
        init_209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_208, 'init')
        # Calling init(args, kwargs) (line 154)
        init_call_result_211 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), init_209, *[], **kwargs_210)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'init'
        module_type_store = module_type_store.open_function_context('init', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sha.init.__dict__.__setitem__('stypy_localization', localization)
        sha.init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sha.init.__dict__.__setitem__('stypy_type_store', module_type_store)
        sha.init.__dict__.__setitem__('stypy_function_name', 'sha.init')
        sha.init.__dict__.__setitem__('stypy_param_names_list', [])
        sha.init.__dict__.__setitem__('stypy_varargs_param_name', None)
        sha.init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sha.init.__dict__.__setitem__('stypy_call_defaults', defaults)
        sha.init.__dict__.__setitem__('stypy_call_varargs', varargs)
        sha.init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sha.init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sha.init', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'init', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'init(...)' code ##################

        str_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'str', 'Initialize the message-digest and set all fields to zero.')
        
        # Assigning a Num to a Attribute (line 159):
        long_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'long')
        # Getting the type of 'self' (line 159)
        self_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self')
        # Setting the type of the member 'length' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_214, 'length', long_213)
        
        # Assigning a List to a Attribute (line 160):
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        
        # Getting the type of 'self' (line 160)
        self_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member 'input' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_216, 'input', list_215)
        
        # Assigning a Num to a Attribute (line 163):
        long_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 18), 'long')
        # Getting the type of 'self' (line 163)
        self_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Setting the type of the member 'H0' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_218, 'H0', long_217)
        
        # Assigning a Num to a Attribute (line 164):
        long_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'long')
        # Getting the type of 'self' (line 164)
        self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self')
        # Setting the type of the member 'H1' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_220, 'H1', long_219)
        
        # Assigning a Num to a Attribute (line 165):
        long_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'long')
        # Getting the type of 'self' (line 165)
        self_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member 'H2' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_222, 'H2', long_221)
        
        # Assigning a Num to a Attribute (line 166):
        long_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'long')
        # Getting the type of 'self' (line 166)
        self_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member 'H3' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_224, 'H3', long_223)
        
        # Assigning a Num to a Attribute (line 167):
        long_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 18), 'long')
        # Getting the type of 'self' (line 167)
        self_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'H4' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_226, 'H4', long_225)
        
        # ################# End of 'init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'init' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'init'
        return stypy_return_type_227


    @norecursion
    def _transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_transform'
        module_type_store = module_type_store.open_function_context('_transform', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sha._transform.__dict__.__setitem__('stypy_localization', localization)
        sha._transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sha._transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        sha._transform.__dict__.__setitem__('stypy_function_name', 'sha._transform')
        sha._transform.__dict__.__setitem__('stypy_param_names_list', ['W'])
        sha._transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        sha._transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sha._transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        sha._transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        sha._transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sha._transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sha._transform', ['W'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_transform', localization, ['W'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_transform(...)' code ##################

        
        
        # Call to range(...): (line 171)
        # Processing the call arguments (line 171)
        int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'int')
        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'int')
        # Processing the call keyword arguments (line 171)
        kwargs_231 = {}
        # Getting the type of 'range' (line 171)
        range_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'range', False)
        # Calling range(args, kwargs) (line 171)
        range_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), range_228, *[int_229, int_230], **kwargs_231)
        
        # Assigning a type to the variable 'range_call_result_232' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'range_call_result_232', range_call_result_232)
        # Testing if the for loop is going to be iterated (line 171)
        # Testing the type of a for loop iterable (line 171)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 8), range_call_result_232)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 171, 8), range_call_result_232):
            # Getting the type of the for loop variable (line 171)
            for_loop_var_233 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 8), range_call_result_232)
            # Assigning a type to the variable 't' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 't', for_loop_var_233)
            # SSA begins for a for statement (line 171)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 172)
            # Processing the call arguments (line 172)
            
            # Call to _rotateLeft(...): (line 172)
            # Processing the call arguments (line 172)
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 173)
            t_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 't', False)
            int_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 22), 'int')
            # Applying the binary operator '-' (line 173)
            result_sub_239 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 18), '-', t_237, int_238)
            
            # Getting the type of 'W' (line 173)
            W_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'W', False)
            # Obtaining the member '__getitem__' of a type (line 173)
            getitem___241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), W_240, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 173)
            subscript_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), getitem___241, result_sub_239)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 173)
            t_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 't', False)
            int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 33), 'int')
            # Applying the binary operator '-' (line 173)
            result_sub_245 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 29), '-', t_243, int_244)
            
            # Getting the type of 'W' (line 173)
            W_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'W', False)
            # Obtaining the member '__getitem__' of a type (line 173)
            getitem___247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 27), W_246, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 173)
            subscript_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 173, 27), getitem___247, result_sub_245)
            
            # Applying the binary operator '^' (line 173)
            result_xor_249 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '^', subscript_call_result_242, subscript_call_result_248)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 173)
            t_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 40), 't', False)
            int_251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 44), 'int')
            # Applying the binary operator '-' (line 173)
            result_sub_252 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 40), '-', t_250, int_251)
            
            # Getting the type of 'W' (line 173)
            W_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'W', False)
            # Obtaining the member '__getitem__' of a type (line 173)
            getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 38), W_253, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 173)
            subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 173, 38), getitem___254, result_sub_252)
            
            # Applying the binary operator '^' (line 173)
            result_xor_256 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 36), '^', result_xor_249, subscript_call_result_255)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 173)
            t_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 52), 't', False)
            int_258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 56), 'int')
            # Applying the binary operator '-' (line 173)
            result_sub_259 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 52), '-', t_257, int_258)
            
            # Getting the type of 'W' (line 173)
            W_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'W', False)
            # Obtaining the member '__getitem__' of a type (line 173)
            getitem___261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 50), W_260, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 173)
            subscript_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 173, 50), getitem___261, result_sub_259)
            
            # Applying the binary operator '^' (line 173)
            result_xor_263 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 48), '^', result_xor_256, subscript_call_result_262)
            
            int_264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 61), 'int')
            # Processing the call keyword arguments (line 172)
            kwargs_265 = {}
            # Getting the type of '_rotateLeft' (line 172)
            _rotateLeft_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 172)
            _rotateLeft_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 172, 21), _rotateLeft_236, *[result_xor_263, int_264], **kwargs_265)
            
            long_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 66), 'long')
            # Applying the binary operator '&' (line 172)
            result_and__268 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 21), '&', _rotateLeft_call_result_266, long_267)
            
            # Processing the call keyword arguments (line 172)
            kwargs_269 = {}
            # Getting the type of 'W' (line 172)
            W_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'W', False)
            # Obtaining the member 'append' of a type (line 172)
            append_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), W_234, 'append')
            # Calling append(args, kwargs) (line 172)
            append_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), append_235, *[result_and__268], **kwargs_269)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Attribute to a Name (line 175):
        # Getting the type of 'self' (line 175)
        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'self')
        # Obtaining the member 'H0' of a type (line 175)
        H0_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), self_271, 'H0')
        # Assigning a type to the variable 'A' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'A', H0_272)
        
        # Assigning a Attribute to a Name (line 176):
        # Getting the type of 'self' (line 176)
        self_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self')
        # Obtaining the member 'H1' of a type (line 176)
        H1_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_273, 'H1')
        # Assigning a type to the variable 'B' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'B', H1_274)
        
        # Assigning a Attribute to a Name (line 177):
        # Getting the type of 'self' (line 177)
        self_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'self')
        # Obtaining the member 'H2' of a type (line 177)
        H2_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), self_275, 'H2')
        # Assigning a type to the variable 'C' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'C', H2_276)
        
        # Assigning a Attribute to a Name (line 178):
        # Getting the type of 'self' (line 178)
        self_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self')
        # Obtaining the member 'H3' of a type (line 178)
        H3_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_277, 'H3')
        # Assigning a type to the variable 'D' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'D', H3_278)
        
        # Assigning a Attribute to a Name (line 179):
        # Getting the type of 'self' (line 179)
        self_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'self')
        # Obtaining the member 'H4' of a type (line 179)
        H4_280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), self_279, 'H4')
        # Assigning a type to the variable 'E' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'E', H4_280)
        str_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'str', '\n        This loop was unrolled to gain about 10% in speed\n        for t in range(0, 80):\n            TEMP = _rotateLeft(A, 5) + f[t/20] + E + W[t] + K[t/20]\n            E = D\n            D = C\n            C = _rotateLeft(B, 30) & 0xffffffffL\n            B = A\n            A = TEMP & 0xffffffffL\n        ')
        
        
        # Call to range(...): (line 192)
        # Processing the call arguments (line 192)
        int_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'int')
        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 26), 'int')
        # Processing the call keyword arguments (line 192)
        kwargs_285 = {}
        # Getting the type of 'range' (line 192)
        range_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'range', False)
        # Calling range(args, kwargs) (line 192)
        range_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 192, 17), range_282, *[int_283, int_284], **kwargs_285)
        
        # Assigning a type to the variable 'range_call_result_286' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'range_call_result_286', range_call_result_286)
        # Testing if the for loop is going to be iterated (line 192)
        # Testing the type of a for loop iterable (line 192)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 192, 8), range_call_result_286)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 192, 8), range_call_result_286):
            # Getting the type of the for loop variable (line 192)
            for_loop_var_287 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 192, 8), range_call_result_286)
            # Assigning a type to the variable 't' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 't', for_loop_var_287)
            # SSA begins for a for statement (line 192)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 193):
            
            # Call to _rotateLeft(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'A' (line 193)
            A_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'A', False)
            int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 34), 'int')
            # Processing the call keyword arguments (line 193)
            kwargs_291 = {}
            # Getting the type of '_rotateLeft' (line 193)
            _rotateLeft_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 193)
            _rotateLeft_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 193, 19), _rotateLeft_288, *[A_289, int_290], **kwargs_291)
            
            # Getting the type of 'B' (line 193)
            B_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 41), 'B')
            # Getting the type of 'C' (line 193)
            C_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 45), 'C')
            # Applying the binary operator '&' (line 193)
            result_and__295 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 41), '&', B_293, C_294)
            
            
            # Getting the type of 'B' (line 193)
            B_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 54), 'B')
            # Applying the '~' unary operator (line 193)
            result_inv_297 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 52), '~', B_296)
            
            # Getting the type of 'D' (line 193)
            D_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 59), 'D')
            # Applying the binary operator '&' (line 193)
            result_and__299 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 51), '&', result_inv_297, D_298)
            
            # Applying the binary operator '|' (line 193)
            result_or__300 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 40), '|', result_and__295, result_and__299)
            
            # Applying the binary operator '+' (line 193)
            result_add_301 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 19), '+', _rotateLeft_call_result_292, result_or__300)
            
            # Getting the type of 'E' (line 193)
            E_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 65), 'E')
            # Applying the binary operator '+' (line 193)
            result_add_303 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 63), '+', result_add_301, E_302)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 193)
            t_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 71), 't')
            # Getting the type of 'W' (line 193)
            W_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 69), 'W')
            # Obtaining the member '__getitem__' of a type (line 193)
            getitem___306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 69), W_305, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 193)
            subscript_call_result_307 = invoke(stypy.reporting.localization.Localization(__file__, 193, 69), getitem___306, t_304)
            
            # Applying the binary operator '+' (line 193)
            result_add_308 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 67), '+', result_add_303, subscript_call_result_307)
            
            
            # Obtaining the type of the subscript
            int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 78), 'int')
            # Getting the type of 'K' (line 193)
            K_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 76), 'K')
            # Obtaining the member '__getitem__' of a type (line 193)
            getitem___311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 76), K_310, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 193)
            subscript_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 193, 76), getitem___311, int_309)
            
            # Applying the binary operator '+' (line 193)
            result_add_313 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 74), '+', result_add_308, subscript_call_result_312)
            
            # Assigning a type to the variable 'TEMP' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'TEMP', result_add_313)
            
            # Assigning a Name to a Name (line 194):
            # Getting the type of 'D' (line 194)
            D_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'D')
            # Assigning a type to the variable 'E' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'E', D_314)
            
            # Assigning a Name to a Name (line 195):
            # Getting the type of 'C' (line 195)
            C_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'C')
            # Assigning a type to the variable 'D' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'D', C_315)
            
            # Assigning a BinOp to a Name (line 196):
            
            # Call to _rotateLeft(...): (line 196)
            # Processing the call arguments (line 196)
            # Getting the type of 'B' (line 196)
            B_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'B', False)
            int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'int')
            # Processing the call keyword arguments (line 196)
            kwargs_319 = {}
            # Getting the type of '_rotateLeft' (line 196)
            _rotateLeft_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 196)
            _rotateLeft_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 196, 16), _rotateLeft_316, *[B_317, int_318], **kwargs_319)
            
            long_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 37), 'long')
            # Applying the binary operator '&' (line 196)
            result_and__322 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 16), '&', _rotateLeft_call_result_320, long_321)
            
            # Assigning a type to the variable 'C' (line 196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'C', result_and__322)
            
            # Assigning a Name to a Name (line 197):
            # Getting the type of 'A' (line 197)
            A_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'A')
            # Assigning a type to the variable 'B' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'B', A_323)
            
            # Assigning a BinOp to a Name (line 198):
            # Getting the type of 'TEMP' (line 198)
            TEMP_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'TEMP')
            long_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'long')
            # Applying the binary operator '&' (line 198)
            result_and__326 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 16), '&', TEMP_324, long_325)
            
            # Assigning a type to the variable 'A' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'A', result_and__326)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 200)
        # Processing the call arguments (line 200)
        int_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 23), 'int')
        int_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'int')
        # Processing the call keyword arguments (line 200)
        kwargs_330 = {}
        # Getting the type of 'range' (line 200)
        range_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'range', False)
        # Calling range(args, kwargs) (line 200)
        range_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 200, 17), range_327, *[int_328, int_329], **kwargs_330)
        
        # Assigning a type to the variable 'range_call_result_331' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'range_call_result_331', range_call_result_331)
        # Testing if the for loop is going to be iterated (line 200)
        # Testing the type of a for loop iterable (line 200)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 200, 8), range_call_result_331)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 200, 8), range_call_result_331):
            # Getting the type of the for loop variable (line 200)
            for_loop_var_332 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 200, 8), range_call_result_331)
            # Assigning a type to the variable 't' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 't', for_loop_var_332)
            # SSA begins for a for statement (line 200)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 201):
            
            # Call to _rotateLeft(...): (line 201)
            # Processing the call arguments (line 201)
            # Getting the type of 'A' (line 201)
            A_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 31), 'A', False)
            int_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 34), 'int')
            # Processing the call keyword arguments (line 201)
            kwargs_336 = {}
            # Getting the type of '_rotateLeft' (line 201)
            _rotateLeft_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 201)
            _rotateLeft_call_result_337 = invoke(stypy.reporting.localization.Localization(__file__, 201, 19), _rotateLeft_333, *[A_334, int_335], **kwargs_336)
            
            # Getting the type of 'B' (line 201)
            B_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 40), 'B')
            # Getting the type of 'C' (line 201)
            C_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 44), 'C')
            # Applying the binary operator '^' (line 201)
            result_xor_340 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 40), '^', B_338, C_339)
            
            # Getting the type of 'D' (line 201)
            D_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 48), 'D')
            # Applying the binary operator '^' (line 201)
            result_xor_342 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 46), '^', result_xor_340, D_341)
            
            # Applying the binary operator '+' (line 201)
            result_add_343 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 19), '+', _rotateLeft_call_result_337, result_xor_342)
            
            # Getting the type of 'E' (line 201)
            E_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 53), 'E')
            # Applying the binary operator '+' (line 201)
            result_add_345 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 51), '+', result_add_343, E_344)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 201)
            t_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 59), 't')
            # Getting the type of 'W' (line 201)
            W_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'W')
            # Obtaining the member '__getitem__' of a type (line 201)
            getitem___348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 57), W_347, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 201)
            subscript_call_result_349 = invoke(stypy.reporting.localization.Localization(__file__, 201, 57), getitem___348, t_346)
            
            # Applying the binary operator '+' (line 201)
            result_add_350 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 55), '+', result_add_345, subscript_call_result_349)
            
            
            # Obtaining the type of the subscript
            int_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 66), 'int')
            # Getting the type of 'K' (line 201)
            K_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 64), 'K')
            # Obtaining the member '__getitem__' of a type (line 201)
            getitem___353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 64), K_352, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 201)
            subscript_call_result_354 = invoke(stypy.reporting.localization.Localization(__file__, 201, 64), getitem___353, int_351)
            
            # Applying the binary operator '+' (line 201)
            result_add_355 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 62), '+', result_add_350, subscript_call_result_354)
            
            # Assigning a type to the variable 'TEMP' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'TEMP', result_add_355)
            
            # Assigning a Name to a Name (line 202):
            # Getting the type of 'D' (line 202)
            D_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'D')
            # Assigning a type to the variable 'E' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'E', D_356)
            
            # Assigning a Name to a Name (line 203):
            # Getting the type of 'C' (line 203)
            C_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'C')
            # Assigning a type to the variable 'D' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'D', C_357)
            
            # Assigning a BinOp to a Name (line 204):
            
            # Call to _rotateLeft(...): (line 204)
            # Processing the call arguments (line 204)
            # Getting the type of 'B' (line 204)
            B_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 28), 'B', False)
            int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 31), 'int')
            # Processing the call keyword arguments (line 204)
            kwargs_361 = {}
            # Getting the type of '_rotateLeft' (line 204)
            _rotateLeft_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 204)
            _rotateLeft_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), _rotateLeft_358, *[B_359, int_360], **kwargs_361)
            
            long_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 37), 'long')
            # Applying the binary operator '&' (line 204)
            result_and__364 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 16), '&', _rotateLeft_call_result_362, long_363)
            
            # Assigning a type to the variable 'C' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'C', result_and__364)
            
            # Assigning a Name to a Name (line 205):
            # Getting the type of 'A' (line 205)
            A_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'A')
            # Assigning a type to the variable 'B' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'B', A_365)
            
            # Assigning a BinOp to a Name (line 206):
            # Getting the type of 'TEMP' (line 206)
            TEMP_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'TEMP')
            long_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 23), 'long')
            # Applying the binary operator '&' (line 206)
            result_and__368 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 16), '&', TEMP_366, long_367)
            
            # Assigning a type to the variable 'A' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'A', result_and__368)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 208)
        # Processing the call arguments (line 208)
        int_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'int')
        int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 27), 'int')
        # Processing the call keyword arguments (line 208)
        kwargs_372 = {}
        # Getting the type of 'range' (line 208)
        range_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 'range', False)
        # Calling range(args, kwargs) (line 208)
        range_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 208, 17), range_369, *[int_370, int_371], **kwargs_372)
        
        # Assigning a type to the variable 'range_call_result_373' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'range_call_result_373', range_call_result_373)
        # Testing if the for loop is going to be iterated (line 208)
        # Testing the type of a for loop iterable (line 208)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 208, 8), range_call_result_373)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 208, 8), range_call_result_373):
            # Getting the type of the for loop variable (line 208)
            for_loop_var_374 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 208, 8), range_call_result_373)
            # Assigning a type to the variable 't' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 't', for_loop_var_374)
            # SSA begins for a for statement (line 208)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 209):
            
            # Call to _rotateLeft(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'A' (line 209)
            A_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 31), 'A', False)
            int_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 34), 'int')
            # Processing the call keyword arguments (line 209)
            kwargs_378 = {}
            # Getting the type of '_rotateLeft' (line 209)
            _rotateLeft_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 209)
            _rotateLeft_call_result_379 = invoke(stypy.reporting.localization.Localization(__file__, 209, 19), _rotateLeft_375, *[A_376, int_377], **kwargs_378)
            
            # Getting the type of 'B' (line 209)
            B_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 41), 'B')
            # Getting the type of 'C' (line 209)
            C_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 45), 'C')
            # Applying the binary operator '&' (line 209)
            result_and__382 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 41), '&', B_380, C_381)
            
            # Getting the type of 'B' (line 209)
            B_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'B')
            # Getting the type of 'D' (line 209)
            D_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 55), 'D')
            # Applying the binary operator '&' (line 209)
            result_and__385 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 51), '&', B_383, D_384)
            
            # Applying the binary operator '|' (line 209)
            result_or__386 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 40), '|', result_and__382, result_and__385)
            
            # Getting the type of 'C' (line 209)
            C_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 61), 'C')
            # Getting the type of 'D' (line 209)
            D_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 65), 'D')
            # Applying the binary operator '&' (line 209)
            result_and__389 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 61), '&', C_387, D_388)
            
            # Applying the binary operator '|' (line 209)
            result_or__390 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 58), '|', result_or__386, result_and__389)
            
            # Applying the binary operator '+' (line 209)
            result_add_391 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 19), '+', _rotateLeft_call_result_379, result_or__390)
            
            # Getting the type of 'E' (line 209)
            E_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 71), 'E')
            # Applying the binary operator '+' (line 209)
            result_add_393 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 69), '+', result_add_391, E_392)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 209)
            t_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 77), 't')
            # Getting the type of 'W' (line 209)
            W_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 75), 'W')
            # Obtaining the member '__getitem__' of a type (line 209)
            getitem___396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 75), W_395, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 209)
            subscript_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 209, 75), getitem___396, t_394)
            
            # Applying the binary operator '+' (line 209)
            result_add_398 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 73), '+', result_add_393, subscript_call_result_397)
            
            
            # Obtaining the type of the subscript
            int_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 84), 'int')
            # Getting the type of 'K' (line 209)
            K_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 82), 'K')
            # Obtaining the member '__getitem__' of a type (line 209)
            getitem___401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 82), K_400, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 209)
            subscript_call_result_402 = invoke(stypy.reporting.localization.Localization(__file__, 209, 82), getitem___401, int_399)
            
            # Applying the binary operator '+' (line 209)
            result_add_403 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 80), '+', result_add_398, subscript_call_result_402)
            
            # Assigning a type to the variable 'TEMP' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'TEMP', result_add_403)
            
            # Assigning a Name to a Name (line 210):
            # Getting the type of 'D' (line 210)
            D_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'D')
            # Assigning a type to the variable 'E' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'E', D_404)
            
            # Assigning a Name to a Name (line 211):
            # Getting the type of 'C' (line 211)
            C_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'C')
            # Assigning a type to the variable 'D' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'D', C_405)
            
            # Assigning a BinOp to a Name (line 212):
            
            # Call to _rotateLeft(...): (line 212)
            # Processing the call arguments (line 212)
            # Getting the type of 'B' (line 212)
            B_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'B', False)
            int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 31), 'int')
            # Processing the call keyword arguments (line 212)
            kwargs_409 = {}
            # Getting the type of '_rotateLeft' (line 212)
            _rotateLeft_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 212)
            _rotateLeft_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), _rotateLeft_406, *[B_407, int_408], **kwargs_409)
            
            long_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 37), 'long')
            # Applying the binary operator '&' (line 212)
            result_and__412 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 16), '&', _rotateLeft_call_result_410, long_411)
            
            # Assigning a type to the variable 'C' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'C', result_and__412)
            
            # Assigning a Name to a Name (line 213):
            # Getting the type of 'A' (line 213)
            A_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'A')
            # Assigning a type to the variable 'B' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'B', A_413)
            
            # Assigning a BinOp to a Name (line 214):
            # Getting the type of 'TEMP' (line 214)
            TEMP_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'TEMP')
            long_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 23), 'long')
            # Applying the binary operator '&' (line 214)
            result_and__416 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 16), '&', TEMP_414, long_415)
            
            # Assigning a type to the variable 'A' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'A', result_and__416)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 216)
        # Processing the call arguments (line 216)
        int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'int')
        int_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 27), 'int')
        # Processing the call keyword arguments (line 216)
        kwargs_420 = {}
        # Getting the type of 'range' (line 216)
        range_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'range', False)
        # Calling range(args, kwargs) (line 216)
        range_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 216, 17), range_417, *[int_418, int_419], **kwargs_420)
        
        # Assigning a type to the variable 'range_call_result_421' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'range_call_result_421', range_call_result_421)
        # Testing if the for loop is going to be iterated (line 216)
        # Testing the type of a for loop iterable (line 216)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_421)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_421):
            # Getting the type of the for loop variable (line 216)
            for_loop_var_422 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_421)
            # Assigning a type to the variable 't' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 't', for_loop_var_422)
            # SSA begins for a for statement (line 216)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 217):
            
            # Call to _rotateLeft(...): (line 217)
            # Processing the call arguments (line 217)
            # Getting the type of 'A' (line 217)
            A_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'A', False)
            int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'int')
            # Processing the call keyword arguments (line 217)
            kwargs_426 = {}
            # Getting the type of '_rotateLeft' (line 217)
            _rotateLeft_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 217)
            _rotateLeft_call_result_427 = invoke(stypy.reporting.localization.Localization(__file__, 217, 19), _rotateLeft_423, *[A_424, int_425], **kwargs_426)
            
            # Getting the type of 'B' (line 217)
            B_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 40), 'B')
            # Getting the type of 'C' (line 217)
            C_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 44), 'C')
            # Applying the binary operator '^' (line 217)
            result_xor_430 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 40), '^', B_428, C_429)
            
            # Getting the type of 'D' (line 217)
            D_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 48), 'D')
            # Applying the binary operator '^' (line 217)
            result_xor_432 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 46), '^', result_xor_430, D_431)
            
            # Applying the binary operator '+' (line 217)
            result_add_433 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 19), '+', _rotateLeft_call_result_427, result_xor_432)
            
            # Getting the type of 'E' (line 217)
            E_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 53), 'E')
            # Applying the binary operator '+' (line 217)
            result_add_435 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 51), '+', result_add_433, E_434)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 't' (line 217)
            t_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 59), 't')
            # Getting the type of 'W' (line 217)
            W_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 57), 'W')
            # Obtaining the member '__getitem__' of a type (line 217)
            getitem___438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 57), W_437, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 217)
            subscript_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 217, 57), getitem___438, t_436)
            
            # Applying the binary operator '+' (line 217)
            result_add_440 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 55), '+', result_add_435, subscript_call_result_439)
            
            
            # Obtaining the type of the subscript
            int_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 66), 'int')
            # Getting the type of 'K' (line 217)
            K_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 64), 'K')
            # Obtaining the member '__getitem__' of a type (line 217)
            getitem___443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 64), K_442, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 217)
            subscript_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 217, 64), getitem___443, int_441)
            
            # Applying the binary operator '+' (line 217)
            result_add_445 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 62), '+', result_add_440, subscript_call_result_444)
            
            # Assigning a type to the variable 'TEMP' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'TEMP', result_add_445)
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'D' (line 218)
            D_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'D')
            # Assigning a type to the variable 'E' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'E', D_446)
            
            # Assigning a Name to a Name (line 219):
            # Getting the type of 'C' (line 219)
            C_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'C')
            # Assigning a type to the variable 'D' (line 219)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'D', C_447)
            
            # Assigning a BinOp to a Name (line 220):
            
            # Call to _rotateLeft(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'B' (line 220)
            B_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'B', False)
            int_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 31), 'int')
            # Processing the call keyword arguments (line 220)
            kwargs_451 = {}
            # Getting the type of '_rotateLeft' (line 220)
            _rotateLeft_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), '_rotateLeft', False)
            # Calling _rotateLeft(args, kwargs) (line 220)
            _rotateLeft_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), _rotateLeft_448, *[B_449, int_450], **kwargs_451)
            
            long_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 37), 'long')
            # Applying the binary operator '&' (line 220)
            result_and__454 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 16), '&', _rotateLeft_call_result_452, long_453)
            
            # Assigning a type to the variable 'C' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'C', result_and__454)
            
            # Assigning a Name to a Name (line 221):
            # Getting the type of 'A' (line 221)
            A_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'A')
            # Assigning a type to the variable 'B' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'B', A_455)
            
            # Assigning a BinOp to a Name (line 222):
            # Getting the type of 'TEMP' (line 222)
            TEMP_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'TEMP')
            long_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'long')
            # Applying the binary operator '&' (line 222)
            result_and__458 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 16), '&', TEMP_456, long_457)
            
            # Assigning a type to the variable 'A' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'A', result_and__458)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a BinOp to a Attribute (line 224):
        # Getting the type of 'self' (line 224)
        self_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'self')
        # Obtaining the member 'H0' of a type (line 224)
        H0_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 19), self_459, 'H0')
        # Getting the type of 'A' (line 224)
        A_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 29), 'A')
        # Applying the binary operator '+' (line 224)
        result_add_462 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 19), '+', H0_460, A_461)
        
        long_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 34), 'long')
        # Applying the binary operator '&' (line 224)
        result_and__464 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 18), '&', result_add_462, long_463)
        
        # Getting the type of 'self' (line 224)
        self_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self')
        # Setting the type of the member 'H0' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_465, 'H0', result_and__464)
        
        # Assigning a BinOp to a Attribute (line 225):
        # Getting the type of 'self' (line 225)
        self_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'self')
        # Obtaining the member 'H1' of a type (line 225)
        H1_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), self_466, 'H1')
        # Getting the type of 'B' (line 225)
        B_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 29), 'B')
        # Applying the binary operator '+' (line 225)
        result_add_469 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 19), '+', H1_467, B_468)
        
        long_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 34), 'long')
        # Applying the binary operator '&' (line 225)
        result_and__471 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 18), '&', result_add_469, long_470)
        
        # Getting the type of 'self' (line 225)
        self_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self')
        # Setting the type of the member 'H1' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_472, 'H1', result_and__471)
        
        # Assigning a BinOp to a Attribute (line 226):
        # Getting the type of 'self' (line 226)
        self_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'self')
        # Obtaining the member 'H2' of a type (line 226)
        H2_474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 19), self_473, 'H2')
        # Getting the type of 'C' (line 226)
        C_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'C')
        # Applying the binary operator '+' (line 226)
        result_add_476 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 19), '+', H2_474, C_475)
        
        long_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 34), 'long')
        # Applying the binary operator '&' (line 226)
        result_and__478 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 18), '&', result_add_476, long_477)
        
        # Getting the type of 'self' (line 226)
        self_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'self')
        # Setting the type of the member 'H2' of a type (line 226)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), self_479, 'H2', result_and__478)
        
        # Assigning a BinOp to a Attribute (line 227):
        # Getting the type of 'self' (line 227)
        self_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'self')
        # Obtaining the member 'H3' of a type (line 227)
        H3_481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 19), self_480, 'H3')
        # Getting the type of 'D' (line 227)
        D_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'D')
        # Applying the binary operator '+' (line 227)
        result_add_483 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 19), '+', H3_481, D_482)
        
        long_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 34), 'long')
        # Applying the binary operator '&' (line 227)
        result_and__485 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 18), '&', result_add_483, long_484)
        
        # Getting the type of 'self' (line 227)
        self_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member 'H3' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_486, 'H3', result_and__485)
        
        # Assigning a BinOp to a Attribute (line 228):
        # Getting the type of 'self' (line 228)
        self_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'self')
        # Obtaining the member 'H4' of a type (line 228)
        H4_488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), self_487, 'H4')
        # Getting the type of 'E' (line 228)
        E_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 29), 'E')
        # Applying the binary operator '+' (line 228)
        result_add_490 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 19), '+', H4_488, E_489)
        
        long_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 34), 'long')
        # Applying the binary operator '&' (line 228)
        result_and__492 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 18), '&', result_add_490, long_491)
        
        # Getting the type of 'self' (line 228)
        self_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member 'H4' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_493, 'H4', result_and__492)
        
        # ################# End of '_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_transform'
        return stypy_return_type_494


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sha.update.__dict__.__setitem__('stypy_localization', localization)
        sha.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sha.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        sha.update.__dict__.__setitem__('stypy_function_name', 'sha.update')
        sha.update.__dict__.__setitem__('stypy_param_names_list', ['inBufseq'])
        sha.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        sha.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sha.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        sha.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        sha.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sha.update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sha.update', ['inBufseq'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['inBufseq'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        str_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, (-1)), 'str', 'Add to the current message.\n\n        Update the md5 object with the string arg. Repeated calls\n        are equivalent to a single call with the concatenation of all\n        the arguments, i.e. m.update(a); m.update(b) is equivalent\n        to m.update(a+b).\n\n        The hash is immediately calculated for all full blocks. The final\n        calculation is made in digest(). It will calculate 1-2 blocks,\n        depending on how much padding we have to add. This allows us to\n        keep an intermediate value for the hash, so that we only need to\n        make minimal recalculation if we call update() to add more data\n        to the hashed string.\n        ')
        
        # Assigning a Call to a Name (line 249):
        
        # Call to list(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'inBufseq' (line 249)
        inBufseq_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'inBufseq', False)
        # Processing the call keyword arguments (line 249)
        kwargs_498 = {}
        # Getting the type of 'list' (line 249)
        list_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'list', False)
        # Calling list(args, kwargs) (line 249)
        list_call_result_499 = invoke(stypy.reporting.localization.Localization(__file__, 249, 16), list_496, *[inBufseq_497], **kwargs_498)
        
        # Assigning a type to the variable 'inBuf' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'inBuf', list_call_result_499)
        
        # Assigning a Call to a Name (line 251):
        
        # Call to len(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'inBuf' (line 251)
        inBuf_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'inBuf', False)
        # Processing the call keyword arguments (line 251)
        kwargs_502 = {}
        # Getting the type of 'len' (line 251)
        len_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 19), 'len', False)
        # Calling len(args, kwargs) (line 251)
        len_call_result_503 = invoke(stypy.reporting.localization.Localization(__file__, 251, 19), len_500, *[inBuf_501], **kwargs_502)
        
        # Assigning a type to the variable 'leninBuf' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'leninBuf', len_call_result_503)
        
        # Assigning a BinOp to a Name (line 254):
        
        # Obtaining the type of the subscript
        int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 28), 'int')
        # Getting the type of 'self' (line 254)
        self_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'self')
        # Obtaining the member 'count' of a type (line 254)
        count_506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), self_505, 'count')
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), count_506, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), getitem___507, int_504)
        
        int_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 34), 'int')
        # Applying the binary operator '>>' (line 254)
        result_rshift_510 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 17), '>>', subscript_call_result_508, int_509)
        
        long_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 39), 'long')
        # Applying the binary operator '&' (line 254)
        result_and__512 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 16), '&', result_rshift_510, long_511)
        
        # Assigning a type to the variable 'index' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'index', result_and__512)
        
        # Assigning a BinOp to a Subscript (line 257):
        
        # Obtaining the type of the subscript
        int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 35), 'int')
        # Getting the type of 'self' (line 257)
        self_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 24), 'self')
        # Obtaining the member 'count' of a type (line 257)
        count_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 24), self_514, 'count')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 24), count_515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 257, 24), getitem___516, int_513)
        
        # Getting the type of 'leninBuf' (line 257)
        leninBuf_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 41), 'leninBuf')
        int_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 53), 'int')
        # Applying the binary operator '<<' (line 257)
        result_lshift_520 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 41), '<<', leninBuf_518, int_519)
        
        # Applying the binary operator '+' (line 257)
        result_add_521 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 24), '+', subscript_call_result_517, result_lshift_520)
        
        # Getting the type of 'self' (line 257)
        self_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self')
        # Obtaining the member 'count' of a type (line 257)
        count_523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_522, 'count')
        int_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 19), 'int')
        # Storing an element on a container (line 257)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), count_523, (int_524, result_add_521))
        
        
        # Obtaining the type of the subscript
        int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 22), 'int')
        # Getting the type of 'self' (line 258)
        self_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'self')
        # Obtaining the member 'count' of a type (line 258)
        count_527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 11), self_526, 'count')
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 11), count_527, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 258, 11), getitem___528, int_525)
        
        # Getting the type of 'leninBuf' (line 258)
        leninBuf_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 28), 'leninBuf')
        int_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 40), 'int')
        # Applying the binary operator '<<' (line 258)
        result_lshift_532 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 28), '<<', leninBuf_530, int_531)
        
        # Applying the binary operator '<' (line 258)
        result_lt_533 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 11), '<', subscript_call_result_529, result_lshift_532)
        
        # Testing if the type of an if condition is none (line 258)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 258, 8), result_lt_533):
            pass
        else:
            
            # Testing the type of an if condition (line 258)
            if_condition_534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 8), result_lt_533)
            # Assigning a type to the variable 'if_condition_534' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'if_condition_534', if_condition_534)
            # SSA begins for if statement (line 258)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Subscript (line 259):
            
            # Obtaining the type of the subscript
            int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 39), 'int')
            # Getting the type of 'self' (line 259)
            self_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'self')
            # Obtaining the member 'count' of a type (line 259)
            count_537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 28), self_536, 'count')
            # Obtaining the member '__getitem__' of a type (line 259)
            getitem___538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 28), count_537, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 259)
            subscript_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 259, 28), getitem___538, int_535)
            
            int_540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 44), 'int')
            # Applying the binary operator '+' (line 259)
            result_add_541 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 28), '+', subscript_call_result_539, int_540)
            
            # Getting the type of 'self' (line 259)
            self_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'self')
            # Obtaining the member 'count' of a type (line 259)
            count_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), self_542, 'count')
            int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 23), 'int')
            # Storing an element on a container (line 259)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 12), count_543, (int_544, result_add_541))
            # SSA join for if statement (line 258)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Subscript (line 260):
        
        # Obtaining the type of the subscript
        int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 35), 'int')
        # Getting the type of 'self' (line 260)
        self_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'self')
        # Obtaining the member 'count' of a type (line 260)
        count_547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 24), self_546, 'count')
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 24), count_547, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_549 = invoke(stypy.reporting.localization.Localization(__file__, 260, 24), getitem___548, int_545)
        
        # Getting the type of 'leninBuf' (line 260)
        leninBuf_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 41), 'leninBuf')
        int_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 53), 'int')
        # Applying the binary operator '>>' (line 260)
        result_rshift_552 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 41), '>>', leninBuf_550, int_551)
        
        # Applying the binary operator '+' (line 260)
        result_add_553 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 24), '+', subscript_call_result_549, result_rshift_552)
        
        # Getting the type of 'self' (line 260)
        self_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'self')
        # Obtaining the member 'count' of a type (line 260)
        count_555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), self_554, 'count')
        int_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 19), 'int')
        # Storing an element on a container (line 260)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), count_555, (int_556, result_add_553))
        
        # Assigning a BinOp to a Name (line 262):
        int_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'int')
        # Getting the type of 'index' (line 262)
        index_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'index')
        # Applying the binary operator '-' (line 262)
        result_sub_559 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 18), '-', int_557, index_558)
        
        # Assigning a type to the variable 'partLen' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'partLen', result_sub_559)
        
        # Getting the type of 'leninBuf' (line 264)
        leninBuf_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'leninBuf')
        # Getting the type of 'partLen' (line 264)
        partLen_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 23), 'partLen')
        # Applying the binary operator '>=' (line 264)
        result_ge_562 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 11), '>=', leninBuf_560, partLen_561)
        
        # Testing if the type of an if condition is none (line 264)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 264, 8), result_ge_562):
            
            # Assigning a Num to a Name (line 276):
            int_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'int')
            # Assigning a type to the variable 'i' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'i', int_622)
            
            # Assigning a BinOp to a Attribute (line 277):
            # Getting the type of 'self' (line 277)
            self_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'self')
            # Obtaining the member 'input' of a type (line 277)
            input_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 25), self_623, 'input')
            
            # Call to list(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'inBuf' (line 277)
            inBuf_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 43), 'inBuf', False)
            # Processing the call keyword arguments (line 277)
            kwargs_627 = {}
            # Getting the type of 'list' (line 277)
            list_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 38), 'list', False)
            # Calling list(args, kwargs) (line 277)
            list_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 277, 38), list_625, *[inBuf_626], **kwargs_627)
            
            # Applying the binary operator '+' (line 277)
            result_add_629 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 25), '+', input_624, list_call_result_628)
            
            # Getting the type of 'self' (line 277)
            self_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'self')
            # Setting the type of the member 'input' of a type (line 277)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), self_630, 'input', result_add_629)
        else:
            
            # Testing the type of an if condition (line 264)
            if_condition_563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 8), result_ge_562)
            # Assigning a type to the variable 'if_condition_563' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'if_condition_563', if_condition_563)
            # SSA begins for if statement (line 264)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 265):
            
            # Call to list(...): (line 265)
            # Processing the call arguments (line 265)
            
            # Obtaining the type of the subscript
            # Getting the type of 'partLen' (line 265)
            partLen_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 45), 'partLen', False)
            slice_566 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 38), None, partLen_565, None)
            # Getting the type of 'inBuf' (line 265)
            inBuf_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 38), 'inBuf', False)
            # Obtaining the member '__getitem__' of a type (line 265)
            getitem___568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 38), inBuf_567, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 265)
            subscript_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 265, 38), getitem___568, slice_566)
            
            # Processing the call keyword arguments (line 265)
            kwargs_570 = {}
            # Getting the type of 'list' (line 265)
            list_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 33), 'list', False)
            # Calling list(args, kwargs) (line 265)
            list_call_result_571 = invoke(stypy.reporting.localization.Localization(__file__, 265, 33), list_564, *[subscript_call_result_569], **kwargs_570)
            
            # Getting the type of 'self' (line 265)
            self_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'self')
            # Obtaining the member 'input' of a type (line 265)
            input_573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), self_572, 'input')
            # Getting the type of 'index' (line 265)
            index_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'index')
            slice_575 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 12), index_574, None, None)
            # Storing an element on a container (line 265)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 12), input_573, (slice_575, list_call_result_571))
            
            # Call to _transform(...): (line 267)
            # Processing the call arguments (line 267)
            
            # Call to _bytelist2longBigEndian(...): (line 267)
            # Processing the call arguments (line 267)
            # Getting the type of 'self' (line 267)
            self_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 52), 'self', False)
            # Obtaining the member 'input' of a type (line 267)
            input_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 52), self_579, 'input')
            # Processing the call keyword arguments (line 267)
            kwargs_581 = {}
            # Getting the type of '_bytelist2longBigEndian' (line 267)
            _bytelist2longBigEndian_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 28), '_bytelist2longBigEndian', False)
            # Calling _bytelist2longBigEndian(args, kwargs) (line 267)
            _bytelist2longBigEndian_call_result_582 = invoke(stypy.reporting.localization.Localization(__file__, 267, 28), _bytelist2longBigEndian_578, *[input_580], **kwargs_581)
            
            # Processing the call keyword arguments (line 267)
            kwargs_583 = {}
            # Getting the type of 'self' (line 267)
            self_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'self', False)
            # Obtaining the member '_transform' of a type (line 267)
            _transform_577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), self_576, '_transform')
            # Calling _transform(args, kwargs) (line 267)
            _transform_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), _transform_577, *[_bytelist2longBigEndian_call_result_582], **kwargs_583)
            
            
            # Assigning a Name to a Name (line 268):
            # Getting the type of 'partLen' (line 268)
            partLen_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'partLen')
            # Assigning a type to the variable 'i' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'i', partLen_585)
            
            
            # Getting the type of 'i' (line 269)
            i_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), 'i')
            int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 22), 'int')
            # Applying the binary operator '+' (line 269)
            result_add_588 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 18), '+', i_586, int_587)
            
            # Getting the type of 'leninBuf' (line 269)
            leninBuf_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'leninBuf')
            # Applying the binary operator '<' (line 269)
            result_lt_590 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 18), '<', result_add_588, leninBuf_589)
            
            # Assigning a type to the variable 'result_lt_590' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'result_lt_590', result_lt_590)
            # Testing if the while is going to be iterated (line 269)
            # Testing the type of an if condition (line 269)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 12), result_lt_590)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 269, 12), result_lt_590):
                # SSA begins for while statement (line 269)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Call to _transform(...): (line 270)
                # Processing the call arguments (line 270)
                
                # Call to _bytelist2longBigEndian(...): (line 270)
                # Processing the call arguments (line 270)
                
                # Call to list(...): (line 270)
                # Processing the call arguments (line 270)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 270)
                i_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 67), 'i', False)
                # Getting the type of 'i' (line 270)
                i_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 69), 'i', False)
                int_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 73), 'int')
                # Applying the binary operator '+' (line 270)
                result_add_598 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 69), '+', i_596, int_597)
                
                slice_599 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 61), i_595, result_add_598, None)
                # Getting the type of 'inBuf' (line 270)
                inBuf_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 61), 'inBuf', False)
                # Obtaining the member '__getitem__' of a type (line 270)
                getitem___601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 61), inBuf_600, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 270)
                subscript_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 270, 61), getitem___601, slice_599)
                
                # Processing the call keyword arguments (line 270)
                kwargs_603 = {}
                # Getting the type of 'list' (line 270)
                list_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 56), 'list', False)
                # Calling list(args, kwargs) (line 270)
                list_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 270, 56), list_594, *[subscript_call_result_602], **kwargs_603)
                
                # Processing the call keyword arguments (line 270)
                kwargs_605 = {}
                # Getting the type of '_bytelist2longBigEndian' (line 270)
                _bytelist2longBigEndian_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), '_bytelist2longBigEndian', False)
                # Calling _bytelist2longBigEndian(args, kwargs) (line 270)
                _bytelist2longBigEndian_call_result_606 = invoke(stypy.reporting.localization.Localization(__file__, 270, 32), _bytelist2longBigEndian_593, *[list_call_result_604], **kwargs_605)
                
                # Processing the call keyword arguments (line 270)
                kwargs_607 = {}
                # Getting the type of 'self' (line 270)
                self_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'self', False)
                # Obtaining the member '_transform' of a type (line 270)
                _transform_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), self_591, '_transform')
                # Calling _transform(args, kwargs) (line 270)
                _transform_call_result_608 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), _transform_592, *[_bytelist2longBigEndian_call_result_606], **kwargs_607)
                
                
                # Assigning a BinOp to a Name (line 272):
                # Getting the type of 'i' (line 272)
                i_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'i')
                int_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 24), 'int')
                # Applying the binary operator '+' (line 272)
                result_add_611 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 20), '+', i_609, int_610)
                
                # Assigning a type to the variable 'i' (line 272)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'i', result_add_611)
                # SSA branch for the else part of a while statement (line 269)
                module_type_store.open_ssa_branch('while loop else')
                
                # Assigning a Call to a Attribute (line 274):
                
                # Call to list(...): (line 274)
                # Processing the call arguments (line 274)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 274)
                i_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 40), 'i', False)
                # Getting the type of 'leninBuf' (line 274)
                leninBuf_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'leninBuf', False)
                slice_615 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 34), i_613, leninBuf_614, None)
                # Getting the type of 'inBuf' (line 274)
                inBuf_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 34), 'inBuf', False)
                # Obtaining the member '__getitem__' of a type (line 274)
                getitem___617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 34), inBuf_616, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 274)
                subscript_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 274, 34), getitem___617, slice_615)
                
                # Processing the call keyword arguments (line 274)
                kwargs_619 = {}
                # Getting the type of 'list' (line 274)
                list_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'list', False)
                # Calling list(args, kwargs) (line 274)
                list_call_result_620 = invoke(stypy.reporting.localization.Localization(__file__, 274, 29), list_612, *[subscript_call_result_618], **kwargs_619)
                
                # Getting the type of 'self' (line 274)
                self_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'self')
                # Setting the type of the member 'input' of a type (line 274)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), self_621, 'input', list_call_result_620)
                # SSA join for while statement (line 269)
                module_type_store = module_type_store.join_ssa_context()
            else:
                
                # Assigning a Call to a Attribute (line 274):
                
                # Call to list(...): (line 274)
                # Processing the call arguments (line 274)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 274)
                i_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 40), 'i', False)
                # Getting the type of 'leninBuf' (line 274)
                leninBuf_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'leninBuf', False)
                slice_615 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 34), i_613, leninBuf_614, None)
                # Getting the type of 'inBuf' (line 274)
                inBuf_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 34), 'inBuf', False)
                # Obtaining the member '__getitem__' of a type (line 274)
                getitem___617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 34), inBuf_616, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 274)
                subscript_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 274, 34), getitem___617, slice_615)
                
                # Processing the call keyword arguments (line 274)
                kwargs_619 = {}
                # Getting the type of 'list' (line 274)
                list_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'list', False)
                # Calling list(args, kwargs) (line 274)
                list_call_result_620 = invoke(stypy.reporting.localization.Localization(__file__, 274, 29), list_612, *[subscript_call_result_618], **kwargs_619)
                
                # Getting the type of 'self' (line 274)
                self_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'self')
                # Setting the type of the member 'input' of a type (line 274)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), self_621, 'input', list_call_result_620)

            
            # SSA branch for the else part of an if statement (line 264)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 276):
            int_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'int')
            # Assigning a type to the variable 'i' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'i', int_622)
            
            # Assigning a BinOp to a Attribute (line 277):
            # Getting the type of 'self' (line 277)
            self_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'self')
            # Obtaining the member 'input' of a type (line 277)
            input_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 25), self_623, 'input')
            
            # Call to list(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'inBuf' (line 277)
            inBuf_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 43), 'inBuf', False)
            # Processing the call keyword arguments (line 277)
            kwargs_627 = {}
            # Getting the type of 'list' (line 277)
            list_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 38), 'list', False)
            # Calling list(args, kwargs) (line 277)
            list_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 277, 38), list_625, *[inBuf_626], **kwargs_627)
            
            # Applying the binary operator '+' (line 277)
            result_add_629 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 25), '+', input_624, list_call_result_628)
            
            # Getting the type of 'self' (line 277)
            self_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'self')
            # Setting the type of the member 'input' of a type (line 277)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), self_630, 'input', result_add_629)
            # SSA join for if statement (line 264)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_631


    @norecursion
    def digest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'digest'
        module_type_store = module_type_store.open_function_context('digest', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sha.digest.__dict__.__setitem__('stypy_localization', localization)
        sha.digest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sha.digest.__dict__.__setitem__('stypy_type_store', module_type_store)
        sha.digest.__dict__.__setitem__('stypy_function_name', 'sha.digest')
        sha.digest.__dict__.__setitem__('stypy_param_names_list', [])
        sha.digest.__dict__.__setitem__('stypy_varargs_param_name', None)
        sha.digest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sha.digest.__dict__.__setitem__('stypy_call_defaults', defaults)
        sha.digest.__dict__.__setitem__('stypy_call_varargs', varargs)
        sha.digest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sha.digest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sha.digest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'digest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'digest(...)' code ##################

        str_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, (-1)), 'str', 'Terminate the message-digest computation and return digest.\n\n        Return the digest of the strings passed to the update()\n        method so far. This is a 16-byte string which may contain\n        non-ASCII characters, including null bytes.\n        ')
        
        # Assigning a Attribute to a Name (line 287):
        # Getting the type of 'self' (line 287)
        self_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 13), 'self')
        # Obtaining the member 'H0' of a type (line 287)
        H0_634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 13), self_633, 'H0')
        # Assigning a type to the variable 'H0' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'H0', H0_634)
        
        # Assigning a Attribute to a Name (line 288):
        # Getting the type of 'self' (line 288)
        self_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 13), 'self')
        # Obtaining the member 'H1' of a type (line 288)
        H1_636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 13), self_635, 'H1')
        # Assigning a type to the variable 'H1' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'H1', H1_636)
        
        # Assigning a Attribute to a Name (line 289):
        # Getting the type of 'self' (line 289)
        self_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 13), 'self')
        # Obtaining the member 'H2' of a type (line 289)
        H2_638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 13), self_637, 'H2')
        # Assigning a type to the variable 'H2' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'H2', H2_638)
        
        # Assigning a Attribute to a Name (line 290):
        # Getting the type of 'self' (line 290)
        self_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 13), 'self')
        # Obtaining the member 'H3' of a type (line 290)
        H3_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 13), self_639, 'H3')
        # Assigning a type to the variable 'H3' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'H3', H3_640)
        
        # Assigning a Attribute to a Name (line 291):
        # Getting the type of 'self' (line 291)
        self_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 13), 'self')
        # Obtaining the member 'H4' of a type (line 291)
        H4_642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 13), self_641, 'H4')
        # Assigning a type to the variable 'H4' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'H4', H4_642)
        
        # Assigning a BinOp to a Name (line 292):
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        
        # Getting the type of 'self' (line 292)
        self_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'self')
        # Obtaining the member 'input' of a type (line 292)
        input_645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 21), self_644, 'input')
        # Applying the binary operator '+' (line 292)
        result_add_646 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 16), '+', list_643, input_645)
        
        # Assigning a type to the variable 'input' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'input', result_add_646)
        
        # Assigning a BinOp to a Name (line 293):
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        
        # Getting the type of 'self' (line 293)
        self_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'self')
        # Obtaining the member 'count' of a type (line 293)
        count_649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 21), self_648, 'count')
        # Applying the binary operator '+' (line 293)
        result_add_650 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 16), '+', list_647, count_649)
        
        # Assigning a type to the variable 'count' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'count', result_add_650)
        
        # Assigning a BinOp to a Name (line 295):
        
        # Obtaining the type of the subscript
        int_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 28), 'int')
        # Getting the type of 'self' (line 295)
        self_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'self')
        # Obtaining the member 'count' of a type (line 295)
        count_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 17), self_652, 'count')
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 17), count_653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 295, 17), getitem___654, int_651)
        
        int_656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 34), 'int')
        # Applying the binary operator '>>' (line 295)
        result_rshift_657 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 17), '>>', subscript_call_result_655, int_656)
        
        long_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 39), 'long')
        # Applying the binary operator '&' (line 295)
        result_and__659 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 16), '&', result_rshift_657, long_658)
        
        # Assigning a type to the variable 'index' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'index', result_and__659)
        
        # Getting the type of 'index' (line 297)
        index_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'index')
        int_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 19), 'int')
        # Applying the binary operator '<' (line 297)
        result_lt_662 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 11), '<', index_660, int_661)
        
        # Testing if the type of an if condition is none (line 297)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 297, 8), result_lt_662):
            
            # Assigning a BinOp to a Name (line 300):
            int_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 21), 'int')
            # Getting the type of 'index' (line 300)
            index_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'index')
            # Applying the binary operator '-' (line 300)
            result_sub_669 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 21), '-', int_667, index_668)
            
            # Assigning a type to the variable 'padLen' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'padLen', result_sub_669)
        else:
            
            # Testing the type of an if condition (line 297)
            if_condition_663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 8), result_lt_662)
            # Assigning a type to the variable 'if_condition_663' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'if_condition_663', if_condition_663)
            # SSA begins for if statement (line 297)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 298):
            int_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 21), 'int')
            # Getting the type of 'index' (line 298)
            index_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 26), 'index')
            # Applying the binary operator '-' (line 298)
            result_sub_666 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 21), '-', int_664, index_665)
            
            # Assigning a type to the variable 'padLen' (line 298)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'padLen', result_sub_666)
            # SSA branch for the else part of an if statement (line 297)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 300):
            int_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 21), 'int')
            # Getting the type of 'index' (line 300)
            index_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'index')
            # Applying the binary operator '-' (line 300)
            result_sub_669 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 21), '-', int_667, index_668)
            
            # Assigning a type to the variable 'padLen' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'padLen', result_sub_669)
            # SSA join for if statement (line 297)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 302):
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        str_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 19), 'str', '\x80')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 18), list_670, str_671)
        
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        str_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 30), 'str', '\x00')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 29), list_672, str_673)
        
        int_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 40), 'int')
        # Applying the binary operator '*' (line 302)
        result_mul_675 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 29), '*', list_672, int_674)
        
        # Applying the binary operator '+' (line 302)
        result_add_676 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 18), '+', list_670, result_mul_675)
        
        # Assigning a type to the variable 'padding' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'padding', result_add_676)
        
        # Call to update(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Obtaining the type of the subscript
        # Getting the type of 'padLen' (line 303)
        padLen_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 29), 'padLen', False)
        slice_680 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 20), None, padLen_679, None)
        # Getting the type of 'padding' (line 303)
        padding_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'padding', False)
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), padding_681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 303, 20), getitem___682, slice_680)
        
        # Processing the call keyword arguments (line 303)
        kwargs_684 = {}
        # Getting the type of 'self' (line 303)
        self_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'self', False)
        # Obtaining the member 'update' of a type (line 303)
        update_678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), self_677, 'update')
        # Calling update(args, kwargs) (line 303)
        update_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), update_678, *[subscript_call_result_683], **kwargs_684)
        
        
        # Assigning a BinOp to a Name (line 306):
        
        # Call to _bytelist2longBigEndian(...): (line 306)
        # Processing the call arguments (line 306)
        
        # Obtaining the type of the subscript
        int_687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 51), 'int')
        slice_688 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 306, 39), None, int_687, None)
        # Getting the type of 'self' (line 306)
        self_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 39), 'self', False)
        # Obtaining the member 'input' of a type (line 306)
        input_690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 39), self_689, 'input')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 39), input_690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_692 = invoke(stypy.reporting.localization.Localization(__file__, 306, 39), getitem___691, slice_688)
        
        # Processing the call keyword arguments (line 306)
        kwargs_693 = {}
        # Getting the type of '_bytelist2longBigEndian' (line 306)
        _bytelist2longBigEndian_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), '_bytelist2longBigEndian', False)
        # Calling _bytelist2longBigEndian(args, kwargs) (line 306)
        _bytelist2longBigEndian_call_result_694 = invoke(stypy.reporting.localization.Localization(__file__, 306, 15), _bytelist2longBigEndian_686, *[subscript_call_result_692], **kwargs_693)
        
        # Getting the type of 'count' (line 306)
        count_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 58), 'count')
        # Applying the binary operator '+' (line 306)
        result_add_696 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 15), '+', _bytelist2longBigEndian_call_result_694, count_695)
        
        # Assigning a type to the variable 'bits' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'bits', result_add_696)
        
        # Call to _transform(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'bits' (line 308)
        bits_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'bits', False)
        # Processing the call keyword arguments (line 308)
        kwargs_700 = {}
        # Getting the type of 'self' (line 308)
        self_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self', False)
        # Obtaining the member '_transform' of a type (line 308)
        _transform_698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_697, '_transform')
        # Calling _transform(args, kwargs) (line 308)
        _transform_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), _transform_698, *[bits_699], **kwargs_700)
        
        
        # Assigning a BinOp to a Name (line 311):
        
        # Call to _long2bytesBigEndian(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'self' (line 311)
        self_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 38), 'self', False)
        # Obtaining the member 'H0' of a type (line 311)
        H0_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 38), self_703, 'H0')
        int_705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 47), 'int')
        # Processing the call keyword arguments (line 311)
        kwargs_706 = {}
        # Getting the type of '_long2bytesBigEndian' (line 311)
        _long2bytesBigEndian_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 17), '_long2bytesBigEndian', False)
        # Calling _long2bytesBigEndian(args, kwargs) (line 311)
        _long2bytesBigEndian_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 311, 17), _long2bytesBigEndian_702, *[H0_704, int_705], **kwargs_706)
        
        
        # Call to _long2bytesBigEndian(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'self' (line 312)
        self_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 38), 'self', False)
        # Obtaining the member 'H1' of a type (line 312)
        H1_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 38), self_709, 'H1')
        int_711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 47), 'int')
        # Processing the call keyword arguments (line 312)
        kwargs_712 = {}
        # Getting the type of '_long2bytesBigEndian' (line 312)
        _long2bytesBigEndian_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), '_long2bytesBigEndian', False)
        # Calling _long2bytesBigEndian(args, kwargs) (line 312)
        _long2bytesBigEndian_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 312, 17), _long2bytesBigEndian_708, *[H1_710, int_711], **kwargs_712)
        
        # Applying the binary operator '+' (line 311)
        result_add_714 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 17), '+', _long2bytesBigEndian_call_result_707, _long2bytesBigEndian_call_result_713)
        
        
        # Call to _long2bytesBigEndian(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'self' (line 313)
        self_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 38), 'self', False)
        # Obtaining the member 'H2' of a type (line 313)
        H2_717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 38), self_716, 'H2')
        int_718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 47), 'int')
        # Processing the call keyword arguments (line 313)
        kwargs_719 = {}
        # Getting the type of '_long2bytesBigEndian' (line 313)
        _long2bytesBigEndian_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), '_long2bytesBigEndian', False)
        # Calling _long2bytesBigEndian(args, kwargs) (line 313)
        _long2bytesBigEndian_call_result_720 = invoke(stypy.reporting.localization.Localization(__file__, 313, 17), _long2bytesBigEndian_715, *[H2_717, int_718], **kwargs_719)
        
        # Applying the binary operator '+' (line 312)
        result_add_721 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 50), '+', result_add_714, _long2bytesBigEndian_call_result_720)
        
        
        # Call to _long2bytesBigEndian(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'self' (line 314)
        self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 38), 'self', False)
        # Obtaining the member 'H3' of a type (line 314)
        H3_724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 38), self_723, 'H3')
        int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 47), 'int')
        # Processing the call keyword arguments (line 314)
        kwargs_726 = {}
        # Getting the type of '_long2bytesBigEndian' (line 314)
        _long2bytesBigEndian_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 17), '_long2bytesBigEndian', False)
        # Calling _long2bytesBigEndian(args, kwargs) (line 314)
        _long2bytesBigEndian_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 314, 17), _long2bytesBigEndian_722, *[H3_724, int_725], **kwargs_726)
        
        # Applying the binary operator '+' (line 313)
        result_add_728 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 50), '+', result_add_721, _long2bytesBigEndian_call_result_727)
        
        
        # Call to _long2bytesBigEndian(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 38), 'self', False)
        # Obtaining the member 'H4' of a type (line 315)
        H4_731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 38), self_730, 'H4')
        int_732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 47), 'int')
        # Processing the call keyword arguments (line 315)
        kwargs_733 = {}
        # Getting the type of '_long2bytesBigEndian' (line 315)
        _long2bytesBigEndian_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), '_long2bytesBigEndian', False)
        # Calling _long2bytesBigEndian(args, kwargs) (line 315)
        _long2bytesBigEndian_call_result_734 = invoke(stypy.reporting.localization.Localization(__file__, 315, 17), _long2bytesBigEndian_729, *[H4_731, int_732], **kwargs_733)
        
        # Applying the binary operator '+' (line 314)
        result_add_735 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 50), '+', result_add_728, _long2bytesBigEndian_call_result_734)
        
        # Assigning a type to the variable 'digest' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'digest', result_add_735)
        
        # Assigning a Name to a Attribute (line 317):
        # Getting the type of 'H0' (line 317)
        H0_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 18), 'H0')
        # Getting the type of 'self' (line 317)
        self_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self')
        # Setting the type of the member 'H0' of a type (line 317)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_737, 'H0', H0_736)
        
        # Assigning a Name to a Attribute (line 318):
        # Getting the type of 'H1' (line 318)
        H1_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'H1')
        # Getting the type of 'self' (line 318)
        self_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'self')
        # Setting the type of the member 'H1' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), self_739, 'H1', H1_738)
        
        # Assigning a Name to a Attribute (line 319):
        # Getting the type of 'H2' (line 319)
        H2_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 18), 'H2')
        # Getting the type of 'self' (line 319)
        self_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'self')
        # Setting the type of the member 'H2' of a type (line 319)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), self_741, 'H2', H2_740)
        
        # Assigning a Name to a Attribute (line 320):
        # Getting the type of 'H3' (line 320)
        H3_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 18), 'H3')
        # Getting the type of 'self' (line 320)
        self_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self')
        # Setting the type of the member 'H3' of a type (line 320)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), self_743, 'H3', H3_742)
        
        # Assigning a Name to a Attribute (line 321):
        # Getting the type of 'H4' (line 321)
        H4_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'H4')
        # Getting the type of 'self' (line 321)
        self_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self')
        # Setting the type of the member 'H4' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_745, 'H4', H4_744)
        
        # Assigning a Name to a Attribute (line 322):
        # Getting the type of 'input' (line 322)
        input_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 21), 'input')
        # Getting the type of 'self' (line 322)
        self_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self')
        # Setting the type of the member 'input' of a type (line 322)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_747, 'input', input_746)
        
        # Assigning a Name to a Attribute (line 323):
        # Getting the type of 'count' (line 323)
        count_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'count')
        # Getting the type of 'self' (line 323)
        self_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self')
        # Setting the type of the member 'count' of a type (line 323)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_749, 'count', count_748)
        # Getting the type of 'digest' (line 325)
        digest_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'digest')
        # Assigning a type to the variable 'stypy_return_type' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'stypy_return_type', digest_750)
        
        # ################# End of 'digest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'digest' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'digest'
        return stypy_return_type_751


    @norecursion
    def hexdigest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hexdigest'
        module_type_store = module_type_store.open_function_context('hexdigest', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sha.hexdigest.__dict__.__setitem__('stypy_localization', localization)
        sha.hexdigest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sha.hexdigest.__dict__.__setitem__('stypy_type_store', module_type_store)
        sha.hexdigest.__dict__.__setitem__('stypy_function_name', 'sha.hexdigest')
        sha.hexdigest.__dict__.__setitem__('stypy_param_names_list', [])
        sha.hexdigest.__dict__.__setitem__('stypy_varargs_param_name', None)
        sha.hexdigest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sha.hexdigest.__dict__.__setitem__('stypy_call_defaults', defaults)
        sha.hexdigest.__dict__.__setitem__('stypy_call_varargs', varargs)
        sha.hexdigest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sha.hexdigest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sha.hexdigest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hexdigest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hexdigest(...)' code ##################

        str_752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, (-1)), 'str', 'Terminate and return digest in HEX form.\n\n        Like digest() except the digest is returned as a string of\n        length 32, containing only hexadecimal digits. This may be\n        used to exchange the value safely in email or other non-\n        binary environments.\n        ')
        
        # Call to join(...): (line 335)
        # Processing the call arguments (line 335)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to digest(...): (line 335)
        # Processing the call keyword arguments (line 335)
        kwargs_763 = {}
        # Getting the type of 'self' (line 335)
        self_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 49), 'self', False)
        # Obtaining the member 'digest' of a type (line 335)
        digest_762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 49), self_761, 'digest')
        # Calling digest(args, kwargs) (line 335)
        digest_call_result_764 = invoke(stypy.reporting.localization.Localization(__file__, 335, 49), digest_762, *[], **kwargs_763)
        
        comprehension_765 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 24), digest_call_result_764)
        # Assigning a type to the variable 'c' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'c', comprehension_765)
        str_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 24), 'str', '%02x')
        
        # Call to ord(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'c' (line 335)
        c_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 37), 'c', False)
        # Processing the call keyword arguments (line 335)
        kwargs_758 = {}
        # Getting the type of 'ord' (line 335)
        ord_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), 'ord', False)
        # Calling ord(args, kwargs) (line 335)
        ord_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 335, 33), ord_756, *[c_757], **kwargs_758)
        
        # Applying the binary operator '%' (line 335)
        result_mod_760 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 24), '%', str_755, ord_call_result_759)
        
        list_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 24), list_766, result_mod_760)
        # Processing the call keyword arguments (line 335)
        kwargs_767 = {}
        str_753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 15), 'str', '')
        # Obtaining the member 'join' of a type (line 335)
        join_754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 15), str_753, 'join')
        # Calling join(args, kwargs) (line 335)
        join_call_result_768 = invoke(stypy.reporting.localization.Localization(__file__, 335, 15), join_754, *[list_766], **kwargs_767)
        
        # Assigning a type to the variable 'stypy_return_type' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'stypy_return_type', join_call_result_768)
        
        # ################# End of 'hexdigest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hexdigest' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_769)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hexdigest'
        return stypy_return_type_769


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 337, 4, False)
        # Assigning a type to the variable 'self' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sha.copy.__dict__.__setitem__('stypy_localization', localization)
        sha.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sha.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        sha.copy.__dict__.__setitem__('stypy_function_name', 'sha.copy')
        sha.copy.__dict__.__setitem__('stypy_param_names_list', [])
        sha.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        sha.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sha.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        sha.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        sha.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sha.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sha.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        str_770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, (-1)), 'str', "Return a clone object.\n\n        Return a copy ('clone') of the md5 object. This can be used\n        to efficiently compute the digests of strings that share\n        a common initial substring.\n        ")
        
        # Call to deepcopy(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'self' (line 345)
        self_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 29), 'self', False)
        # Processing the call keyword arguments (line 345)
        kwargs_774 = {}
        # Getting the type of 'copy' (line 345)
        copy_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 15), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 345)
        deepcopy_772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 15), copy_771, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 345)
        deepcopy_call_result_775 = invoke(stypy.reporting.localization.Localization(__file__, 345, 15), deepcopy_772, *[self_773], **kwargs_774)
        
        # Assigning a type to the variable 'stypy_return_type' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'stypy_return_type', deepcopy_call_result_775)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_776


# Assigning a type to the variable 'sha' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'sha', sha)

# Assigning a Num to a Name (line 139):
int_777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 18), 'int')
# Getting the type of 'sha'
sha_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sha')
# Setting the type of the member 'digest_size' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sha_778, 'digest_size', int_777)

# Assigning a Num to a Name (line 140):
int_779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 17), 'int')
# Getting the type of 'sha'
sha_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sha')
# Setting the type of the member 'digestsize' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sha_780, 'digestsize', int_779)

# Assigning a Num to a Name (line 356):
int_781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 14), 'int')
# Assigning a type to the variable 'digest_size' (line 356)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 0), 'digest_size', int_781)

# Assigning a Num to a Name (line 357):
int_782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 13), 'int')
# Assigning a type to the variable 'digestsize' (line 357)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'digestsize', int_782)

# Assigning a Num to a Name (line 358):
int_783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
# Assigning a type to the variable 'blocksize' (line 358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'blocksize', int_783)

@norecursion
def new(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 361)
    None_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'None')
    defaults = [None_784]
    # Create a new context for function 'new'
    module_type_store = module_type_store.open_function_context('new', 361, 0, False)
    
    # Passed parameters checking function
    new.stypy_localization = localization
    new.stypy_type_of_self = None
    new.stypy_type_store = module_type_store
    new.stypy_function_name = 'new'
    new.stypy_param_names_list = ['arg']
    new.stypy_varargs_param_name = None
    new.stypy_kwargs_param_name = None
    new.stypy_call_defaults = defaults
    new.stypy_call_varargs = varargs
    new.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new(...)' code ##################

    str_785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, (-1)), 'str', 'Return a new sha crypto object.\n\n    If arg is present, the method call update(arg) is made.\n    ')
    
    # Assigning a Call to a Name (line 367):
    
    # Call to sha(...): (line 367)
    # Processing the call keyword arguments (line 367)
    kwargs_787 = {}
    # Getting the type of 'sha' (line 367)
    sha_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'sha', False)
    # Calling sha(args, kwargs) (line 367)
    sha_call_result_788 = invoke(stypy.reporting.localization.Localization(__file__, 367, 13), sha_786, *[], **kwargs_787)
    
    # Assigning a type to the variable 'crypto' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'crypto', sha_call_result_788)
    # Getting the type of 'arg' (line 368)
    arg_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 7), 'arg')
    # Testing if the type of an if condition is none (line 368)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 368, 4), arg_789):
        pass
    else:
        
        # Testing the type of an if condition (line 368)
        if_condition_790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 4), arg_789)
        # Assigning a type to the variable 'if_condition_790' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'if_condition_790', if_condition_790)
        # SSA begins for if statement (line 368)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'arg' (line 369)
        arg_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 22), 'arg', False)
        # Processing the call keyword arguments (line 369)
        kwargs_794 = {}
        # Getting the type of 'crypto' (line 369)
        crypto_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'crypto', False)
        # Obtaining the member 'update' of a type (line 369)
        update_792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), crypto_791, 'update')
        # Calling update(args, kwargs) (line 369)
        update_call_result_795 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), update_792, *[arg_793], **kwargs_794)
        
        # SSA join for if statement (line 368)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'crypto' (line 371)
    crypto_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'crypto')
    # Assigning a type to the variable 'stypy_return_type' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type', crypto_796)
    
    # ################# End of 'new(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new' in the type store
    # Getting the type of 'stypy_return_type' (line 361)
    stypy_return_type_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_797)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new'
    return stypy_return_type_797

# Assigning a type to the variable 'new' (line 361)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 0), 'new', new)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 378, 0, False)
    
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

    
    # Assigning a Str to a Name (line 382):
    str_798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 11), 'str', 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.')
    # Assigning a type to the variable 'text' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'text', str_798)
    
    # Assigning a Call to a Name (line 386):
    
    # Call to new(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'text' (line 386)
    text_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'text', False)
    # Processing the call keyword arguments (line 386)
    kwargs_801 = {}
    # Getting the type of 'new' (line 386)
    new_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 11), 'new', False)
    # Calling new(args, kwargs) (line 386)
    new_call_result_802 = invoke(stypy.reporting.localization.Localization(__file__, 386, 11), new_799, *[text_800], **kwargs_801)
    
    # Assigning a type to the variable 'shah' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'shah', new_call_result_802)
    
    # Call to hexdigest(...): (line 391)
    # Processing the call keyword arguments (line 391)
    kwargs_805 = {}
    # Getting the type of 'shah' (line 391)
    shah_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'shah', False)
    # Obtaining the member 'hexdigest' of a type (line 391)
    hexdigest_804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 4), shah_803, 'hexdigest')
    # Calling hexdigest(args, kwargs) (line 391)
    hexdigest_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 391, 4), hexdigest_804, *[], **kwargs_805)
    
    int_807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 7), 'int')
    # Testing if the type of an if condition is none (line 393)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 393, 4), int_807):
        pass
    else:
        
        # Testing the type of an if condition (line 393)
        if_condition_808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 4), int_807)
        # Assigning a type to the variable 'if_condition_808' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'if_condition_808', if_condition_808)
        # SSA begins for if statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 395)
        # Processing the call keyword arguments (line 395)
        kwargs_811 = {}
        # Getting the type of 'shah' (line 395)
        shah_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'shah', False)
        # Obtaining the member 'copy' of a type (line 395)
        copy_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), shah_809, 'copy')
        # Calling copy(args, kwargs) (line 395)
        copy_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), copy_810, *[], **kwargs_811)
        
        
        # Assigning a Num to a Name (line 396):
        long_813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 12), 'long')
        # Assigning a type to the variable 'B' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'B', long_813)
        
        # Assigning a Num to a Name (line 397):
        long_814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 12), 'long')
        # Assigning a type to the variable 'C' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'C', long_814)
        
        # Assigning a Num to a Name (line 398):
        long_815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 12), 'long')
        # Assigning a type to the variable 'D' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'D', long_815)
        
        # Call to f0_19(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'B' (line 400)
        B_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 14), 'B', False)
        # Getting the type of 'C' (line 400)
        C_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'C', False)
        # Getting the type of 'D' (line 400)
        D_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 20), 'D', False)
        # Processing the call keyword arguments (line 400)
        kwargs_820 = {}
        # Getting the type of 'f0_19' (line 400)
        f0_19_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'f0_19', False)
        # Calling f0_19(args, kwargs) (line 400)
        f0_19_call_result_821 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), f0_19_816, *[B_817, C_818, D_819], **kwargs_820)
        
        
        # Call to f20_39(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'B' (line 401)
        B_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'B', False)
        # Getting the type of 'C' (line 401)
        C_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 18), 'C', False)
        # Getting the type of 'D' (line 401)
        D_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 21), 'D', False)
        # Processing the call keyword arguments (line 401)
        kwargs_826 = {}
        # Getting the type of 'f20_39' (line 401)
        f20_39_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'f20_39', False)
        # Calling f20_39(args, kwargs) (line 401)
        f20_39_call_result_827 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), f20_39_822, *[B_823, C_824, D_825], **kwargs_826)
        
        
        # Call to f40_59(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'B' (line 402)
        B_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'B', False)
        # Getting the type of 'C' (line 402)
        C_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'C', False)
        # Getting the type of 'D' (line 402)
        D_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 21), 'D', False)
        # Processing the call keyword arguments (line 402)
        kwargs_832 = {}
        # Getting the type of 'f40_59' (line 402)
        f40_59_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'f40_59', False)
        # Calling f40_59(args, kwargs) (line 402)
        f40_59_call_result_833 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), f40_59_828, *[B_829, C_830, D_831], **kwargs_832)
        
        
        # Call to f60_79(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'B' (line 403)
        B_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'B', False)
        # Getting the type of 'C' (line 403)
        C_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 18), 'C', False)
        # Getting the type of 'D' (line 403)
        D_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 21), 'D', False)
        # Processing the call keyword arguments (line 403)
        kwargs_838 = {}
        # Getting the type of 'f60_79' (line 403)
        f60_79_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'f60_79', False)
        # Calling f60_79(args, kwargs) (line 403)
        f60_79_call_result_839 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), f60_79_834, *[B_835, C_836, D_837], **kwargs_838)
        
        # SSA join for if statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 378)
    stypy_return_type_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_840)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_840

# Assigning a type to the variable 'main' (line 378)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 406, 0, False)
    
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

    
    
    # Call to range(...): (line 407)
    # Processing the call arguments (line 407)
    int_842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 19), 'int')
    # Processing the call keyword arguments (line 407)
    kwargs_843 = {}
    # Getting the type of 'range' (line 407)
    range_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 13), 'range', False)
    # Calling range(args, kwargs) (line 407)
    range_call_result_844 = invoke(stypy.reporting.localization.Localization(__file__, 407, 13), range_841, *[int_842], **kwargs_843)
    
    # Assigning a type to the variable 'range_call_result_844' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'range_call_result_844', range_call_result_844)
    # Testing if the for loop is going to be iterated (line 407)
    # Testing the type of a for loop iterable (line 407)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 407, 4), range_call_result_844)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 407, 4), range_call_result_844):
        # Getting the type of the for loop variable (line 407)
        for_loop_var_845 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 407, 4), range_call_result_844)
        # Assigning a type to the variable 'i' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'i', for_loop_var_845)
        # SSA begins for a for statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to main(...): (line 408)
        # Processing the call keyword arguments (line 408)
        kwargs_847 = {}
        # Getting the type of 'main' (line 408)
        main_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'main', False)
        # Calling main(args, kwargs) (line 408)
        main_call_result_848 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), main_846, *[], **kwargs_847)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 409)
    True_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type', True_849)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 406)
    stypy_return_type_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_850)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_850

# Assigning a type to the variable 'run' (line 406)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'run', run)

# Call to run(...): (line 412)
# Processing the call keyword arguments (line 412)
kwargs_852 = {}
# Getting the type of 'run' (line 412)
run_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 0), 'run', False)
# Calling run(args, kwargs) (line 412)
run_call_result_853 = invoke(stypy.reporting.localization.Localization(__file__, 412, 0), run_851, *[], **kwargs_852)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

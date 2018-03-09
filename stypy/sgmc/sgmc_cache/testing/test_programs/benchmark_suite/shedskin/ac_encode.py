
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ## Arithmetic coding compressor and uncompressor for binary source.
2: ## This is a cleaned-up version of AEncode.py
3: 
4: ## (c) David MacKay - Free software. License: GPL
5: 
6: import os
7: 
8: def Relative(path):
9:     return os.path.join(os.path.dirname(__file__), path)
10: 
11: BETA0=1;BETA1=1 ## default prior distribution
12: M = 30 ; ONE = (1<<M) ; HALF = (1<<(M-1))
13: QUARTER = (1<<(M-2)) ; THREEQU = HALF+QUARTER
14: def clear (c,charstack):
15:     ## print out character c, and other queued characters
16:     a = str(c) + str(1-c)*charstack[0]
17:     charstack[0]=0
18:     return a
19:     pass
20: 
21: def encode (string, c0=BETA0, c1=BETA1, adaptive=1,verbose=0):
22:     b=ONE; a=0;  tot0=0;tot1=0;     assert c0>0; assert c1>0
23:     if adaptive==0:
24:         p0 = c0*1.0/(c0+c1)
25:         pass
26:     ans="";
27:     charstack=[0] ## how many undecided characters remain to print
28:     for c in string:
29:         w=b-a
30:         if adaptive :
31:             cT = c0+c1
32:             p0 = c0*1.0/cT
33:             pass
34:         boundary = a + int(p0*w)
35:         if (boundary == a): boundary += 1; print "warningA"; pass # these warnings mean that some of the probabilities
36:         if (boundary == b): boundary -= 1; print "warningB"; pass # requested by the probabilistic model
37:         ## are so small (compared to our integers) that we had to round them up to bigger values
38:         if (c=='1') :
39:             a = boundary
40:             tot1 += 1
41:             if adaptive: c1 += 1.0 ; pass
42:         elif (c=='0'):
43:             b = boundary
44:             tot0 +=1
45:             if adaptive: c0 += 1.0 ; pass
46:             pass ## ignore other characters
47: 
48:         while ( (a>=HALF) or (b<=HALF) ) :   ## output bits
49:             if (a>=HALF) :
50:                 ans = ans + clear(1,charstack)
51:                 a = a-HALF ;
52:                 b = b-HALF ;
53:             else :
54:                 ans = ans + clear(0,charstack)
55:                 pass
56:             a *= 2 ;      b *= 2
57:             pass
58: 
59:         assert a<=HALF; assert b>=HALF; assert a>=0; assert b<=ONE
60:         ## if the gap a-b is getting small, rescale it
61:         while ( (a>QUARTER) and (b<THREEQU) ):
62:             charstack[0] += 1
63:             a = 2*a-HALF
64:             b = 2*b-HALF
65:             pass
66: 
67:         assert a<=HALF; assert b>=HALF; assert a>=0; assert b<=ONE
68:         pass
69: 
70:     # terminate
71:     if ( (HALF-a) > (b-HALF) ) :
72:         w = (HALF-a) ;
73:         ans = ans + clear(0,charstack)
74:         while ( w < HALF ) :
75:             ans = ans + clear(1,charstack)
76:             w *=2
77:             pass
78:         pass
79:     else :
80:         w = (b-HALF) ;
81:         ans = ans + clear(1,charstack)
82:         while ( w < HALF ) :
83:             ans = ans + clear(0,charstack)
84:             w *=2
85:             pass
86:         pass
87:     return ans
88:     pass
89: 
90: 
91: 
92: def decode (string, N=10000, c0=BETA0, c1=BETA1, adaptive=1,verbose=0):
93:     ## must supply N, the number of source characters remaining.
94:     b=ONE ; a=0 ;      tot0=0;tot1=0  ;     assert c0>0 ; assert c1>0
95:     model_needs_updating = 1
96:     if adaptive==0:
97:         p0 = c0*1.0/(c0+c1)
98:         pass
99:     ans=""
100:     u=0 ; v=ONE
101:     for c in string :
102:         if N<=0 :
103:             break ## break out of the string-reading loop
104:         assert N>0
105: ##    // (u,v) is the current "encoded alphabet" binary interval, and halfway is its midpoint.
106: ##    // (a,b) is the current "source alphabet" interval, and boundary is the "midpoint"
107:         assert u>=0 ; assert v<=ONE
108:         halfway = u + (v-u)/2
109:         if( c == '1' ) :
110:             u = halfway
111:         elif ( c=='0' ):
112:             v = halfway
113:         else:
114:             pass
115: ##    // Read bits until we can decide what the source symbol was.
116: ##    // Then emulate the encoder's computations, and tie (u,v) to tag along for the ride.
117:         while (1): ## condition at end
118:             firsttime = 0
119:             if(model_needs_updating):
120:                 w = b-a
121:                 if adaptive :
122:                     cT = c0 + c1 ;   p0 = c0 *1.0/cT
123:                     pass
124:                 boundary = a + int(p0*w)
125:                 if (boundary == a): boundary += 1; print "warningA"; pass
126:                 if (boundary == b): boundary -= 1; print "warningB"; pass
127:                 model_needs_updating = 0
128:                 pass
129:             if  ( boundary <= u ) :
130:                 ans = ans + "1";             tot1 +=1
131:                 if adaptive: c1 += 1.0 ; pass
132:                 a = boundary ;	model_needs_updating = 1 ; 	N-=1
133:             elif ( boundary >= v )  :
134:                 ans = ans + "0";             tot0 +=1
135:                 if adaptive: c0 += 1.0 ; pass
136:                 b = boundary ;	model_needs_updating = 1 ; 	N-=1
137: ##	// every time we discover a source bit, implement exactly the
138: ##	// computations that were done by the encoder (below).
139:             else :
140: ##	// not enough bits have yet been read to know the decision.
141:                 pass
142: 
143: ##      // emulate outputting of bits by the encoder, and tie (u,v) to tag along for the ride.
144:             while ( (a>=HALF) or (b<=HALF) ) :
145:                 if (a>=HALF) :
146:                     a = a-HALF ;  b = b-HALF ;    u = u-HALF ;     v = v-HALF
147:                     pass
148:                 else :
149:                     pass
150:                 a *= 2 ;      b *= 2 ;      u *= 2 ;      v *= 2 ;
151:                 model_needs_updating = 1
152:                 pass
153: 
154:             assert a<=HALF;            assert b>=HALF;            assert a>=0;            assert b<=ONE
155:         ## if the gap a-b is getting small, rescale it
156:             while ( (a>QUARTER) and (b<THREEQU) ):
157:                 a = 2*a-HALF;  b = 2*b-HALF ; u = 2*u-HALF ;  v = 2*v-HALF
158:                 pass
159:             if not (N>0 and model_needs_updating) : ## this is the "while" for this "do" loop
160:                 break
161:             pass
162:         pass
163:     return ans
164:     pass
165: 
166: def hardertest():
167:     #print "Reading the BentCoinFile"
168:     inputfile = open( Relative("testdata/BentCoinFile") , "r" )
169:     outputfile = open( Relative("tmp.zip") , "w" )
170:     #print  "Compressing to tmp.zip"
171: 
172:     s = inputfile.read()
173:     N = len(s)
174:     zip = encode(s, 10, 1)
175:     outputfile.write(zip)
176:     outputfile.close();     inputfile.close()
177:     #print "DONE compressing"
178: 
179:     inputfile = open( Relative("tmp.zip") , "r" )
180:     outputfile = open( Relative("tmp2") , "w" )
181:     #print  "Uncompressing to tmp2"
182:     unc = decode(list(inputfile.read()), N, 10, 1)
183:     outputfile.write(unc)
184:     outputfile.close();     inputfile.close()
185:     #print "DONE uncompressing"
186: 
187:     #print "Checking for differences..."
188:     #os.system( "diff testdata/BentCoinFile tmp2" )
189:     #os.system( "wc tmp.zip testdata/BentCoinFile tmp2" )
190: 
191: def test():
192:     sl=["1010", "111", "00001000000000000000",\
193:         "1", "10" , "01" , "0" ,"0000000", \
194:         "000000000000000100000000000000000000000000000000100000000000000000011000000" ]
195:     for s in sl:
196:         #print "encoding", s
197:         N=len(s)
198:         e = encode(s,10,1)
199:         #print "decoding", e
200:         ds = decode(e,N,10,1)
201:         #print ds
202:         if  (ds != s) :
203:             #print s
204:             #print "ERR@"
205:             pass
206:         else:
207:             pass#print "ok ---------- "
208:         pass
209:     pass
210: 
211: def run():
212:     test()
213:     hardertest()
214:     return True
215: 
216: run()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 8, 0, False)
    
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

    
    # Call to join(...): (line 9)
    # Processing the call arguments (line 9)
    
    # Call to dirname(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of '__file__' (line 9)
    file___7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 40), '__file__', False)
    # Processing the call keyword arguments (line 9)
    kwargs_8 = {}
    # Getting the type of 'os' (line 9)
    os_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 9)
    path_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 24), os_4, 'path')
    # Obtaining the member 'dirname' of a type (line 9)
    dirname_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 24), path_5, 'dirname')
    # Calling dirname(args, kwargs) (line 9)
    dirname_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 9, 24), dirname_6, *[file___7], **kwargs_8)
    
    # Getting the type of 'path' (line 9)
    path_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 51), 'path', False)
    # Processing the call keyword arguments (line 9)
    kwargs_11 = {}
    # Getting the type of 'os' (line 9)
    os_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 9)
    path_2 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), os_1, 'path')
    # Obtaining the member 'join' of a type (line 9)
    join_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), path_2, 'join')
    # Calling join(args, kwargs) (line 9)
    join_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), join_3, *[dirname_call_result_9, path_10], **kwargs_11)
    
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', join_call_result_12)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_13

# Assigning a type to the variable 'Relative' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'Relative', Relative)

# Assigning a Num to a Name (line 11):
int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 6), 'int')
# Assigning a type to the variable 'BETA0' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'BETA0', int_14)

# Assigning a Num to a Name (line 11):
int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
# Assigning a type to the variable 'BETA1' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'BETA1', int_15)

# Assigning a Num to a Name (line 12):
int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')
# Assigning a type to the variable 'M' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'M', int_16)

# Assigning a BinOp to a Name (line 12):
int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
# Getting the type of 'M' (line 12)
M_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'M')
# Applying the binary operator '<<' (line 12)
result_lshift_19 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 16), '<<', int_17, M_18)

# Assigning a type to the variable 'ONE' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'ONE', result_lshift_19)

# Assigning a BinOp to a Name (line 12):
int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 32), 'int')
# Getting the type of 'M' (line 12)
M_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 36), 'M')
int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 38), 'int')
# Applying the binary operator '-' (line 12)
result_sub_23 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 36), '-', M_21, int_22)

# Applying the binary operator '<<' (line 12)
result_lshift_24 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 32), '<<', int_20, result_sub_23)

# Assigning a type to the variable 'HALF' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'HALF', result_lshift_24)

# Assigning a BinOp to a Name (line 13):
int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'int')
# Getting the type of 'M' (line 13)
M_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'M')
int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
# Applying the binary operator '-' (line 13)
result_sub_28 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 15), '-', M_26, int_27)

# Applying the binary operator '<<' (line 13)
result_lshift_29 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 11), '<<', int_25, result_sub_28)

# Assigning a type to the variable 'QUARTER' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'QUARTER', result_lshift_29)

# Assigning a BinOp to a Name (line 13):
# Getting the type of 'HALF' (line 13)
HALF_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'HALF')
# Getting the type of 'QUARTER' (line 13)
QUARTER_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 38), 'QUARTER')
# Applying the binary operator '+' (line 13)
result_add_32 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 33), '+', HALF_30, QUARTER_31)

# Assigning a type to the variable 'THREEQU' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'THREEQU', result_add_32)

@norecursion
def clear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'clear'
    module_type_store = module_type_store.open_function_context('clear', 14, 0, False)
    
    # Passed parameters checking function
    clear.stypy_localization = localization
    clear.stypy_type_of_self = None
    clear.stypy_type_store = module_type_store
    clear.stypy_function_name = 'clear'
    clear.stypy_param_names_list = ['c', 'charstack']
    clear.stypy_varargs_param_name = None
    clear.stypy_kwargs_param_name = None
    clear.stypy_call_defaults = defaults
    clear.stypy_call_varargs = varargs
    clear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clear', ['c', 'charstack'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clear', localization, ['c', 'charstack'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clear(...)' code ##################

    
    # Assigning a BinOp to a Name (line 16):
    
    # Call to str(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'c' (line 16)
    c_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'c', False)
    # Processing the call keyword arguments (line 16)
    kwargs_35 = {}
    # Getting the type of 'str' (line 16)
    str_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'str', False)
    # Calling str(args, kwargs) (line 16)
    str_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), str_33, *[c_34], **kwargs_35)
    
    
    # Call to str(...): (line 16)
    # Processing the call arguments (line 16)
    int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
    # Getting the type of 'c' (line 16)
    c_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'c', False)
    # Applying the binary operator '-' (line 16)
    result_sub_40 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 21), '-', int_38, c_39)
    
    # Processing the call keyword arguments (line 16)
    kwargs_41 = {}
    # Getting the type of 'str' (line 16)
    str_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'str', False)
    # Calling str(args, kwargs) (line 16)
    str_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), str_37, *[result_sub_40], **kwargs_41)
    
    
    # Obtaining the type of the subscript
    int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 36), 'int')
    # Getting the type of 'charstack' (line 16)
    charstack_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 26), 'charstack')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 26), charstack_44, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 16, 26), getitem___45, int_43)
    
    # Applying the binary operator '*' (line 16)
    result_mul_47 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 17), '*', str_call_result_42, subscript_call_result_46)
    
    # Applying the binary operator '+' (line 16)
    result_add_48 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 8), '+', str_call_result_36, result_mul_47)
    
    # Assigning a type to the variable 'a' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a', result_add_48)
    
    # Assigning a Num to a Subscript (line 17):
    int_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
    # Getting the type of 'charstack' (line 17)
    charstack_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'charstack')
    int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    # Storing an element on a container (line 17)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 4), charstack_50, (int_51, int_49))
    # Getting the type of 'a' (line 18)
    a_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', a_52)
    pass
    
    # ################# End of 'clear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clear' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_53)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clear'
    return stypy_return_type_53

# Assigning a type to the variable 'clear' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'clear', clear)

@norecursion
def encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'BETA0' (line 21)
    BETA0_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'BETA0')
    # Getting the type of 'BETA1' (line 21)
    BETA1_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'BETA1')
    int_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 49), 'int')
    int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 59), 'int')
    defaults = [BETA0_54, BETA1_55, int_56, int_57]
    # Create a new context for function 'encode'
    module_type_store = module_type_store.open_function_context('encode', 21, 0, False)
    
    # Passed parameters checking function
    encode.stypy_localization = localization
    encode.stypy_type_of_self = None
    encode.stypy_type_store = module_type_store
    encode.stypy_function_name = 'encode'
    encode.stypy_param_names_list = ['string', 'c0', 'c1', 'adaptive', 'verbose']
    encode.stypy_varargs_param_name = None
    encode.stypy_kwargs_param_name = None
    encode.stypy_call_defaults = defaults
    encode.stypy_call_varargs = varargs
    encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode', ['string', 'c0', 'c1', 'adaptive', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode', localization, ['string', 'c0', 'c1', 'adaptive', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode(...)' code ##################

    
    # Assigning a Name to a Name (line 22):
    # Getting the type of 'ONE' (line 22)
    ONE_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'ONE')
    # Assigning a type to the variable 'b' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'b', ONE_58)
    
    # Assigning a Num to a Name (line 22):
    int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'int')
    # Assigning a type to the variable 'a' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'a', int_59)
    
    # Assigning a Num to a Name (line 22):
    int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'int')
    # Assigning a type to the variable 'tot0' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'tot0', int_60)
    
    # Assigning a Num to a Name (line 22):
    int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 29), 'int')
    # Assigning a type to the variable 'tot1' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'tot1', int_61)
    # Evaluating assert statement condition
    
    # Getting the type of 'c0' (line 22)
    c0_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 43), 'c0')
    int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'int')
    # Applying the binary operator '>' (line 22)
    result_gt_64 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 43), '>', c0_62, int_63)
    
    assert_65 = result_gt_64
    # Assigning a type to the variable 'assert_65' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 36), 'assert_65', result_gt_64)
    # Evaluating assert statement condition
    
    # Getting the type of 'c1' (line 22)
    c1_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 56), 'c1')
    int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 59), 'int')
    # Applying the binary operator '>' (line 22)
    result_gt_68 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 56), '>', c1_66, int_67)
    
    assert_69 = result_gt_68
    # Assigning a type to the variable 'assert_69' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 49), 'assert_69', result_gt_68)
    
    # Getting the type of 'adaptive' (line 23)
    adaptive_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'adaptive')
    int_71 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'int')
    # Applying the binary operator '==' (line 23)
    result_eq_72 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 7), '==', adaptive_70, int_71)
    
    # Testing if the type of an if condition is none (line 23)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 23, 4), result_eq_72):
        pass
    else:
        
        # Testing the type of an if condition (line 23)
        if_condition_73 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), result_eq_72)
        # Assigning a type to the variable 'if_condition_73' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_condition_73', if_condition_73)
        # SSA begins for if statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 24):
        # Getting the type of 'c0' (line 24)
        c0_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'c0')
        float_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'float')
        # Applying the binary operator '*' (line 24)
        result_mul_76 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 13), '*', c0_74, float_75)
        
        # Getting the type of 'c0' (line 24)
        c0_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'c0')
        # Getting the type of 'c1' (line 24)
        c1_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'c1')
        # Applying the binary operator '+' (line 24)
        result_add_79 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 21), '+', c0_77, c1_78)
        
        # Applying the binary operator 'div' (line 24)
        result_div_80 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), 'div', result_mul_76, result_add_79)
        
        # Assigning a type to the variable 'p0' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'p0', result_div_80)
        pass
        # SSA join for if statement (line 23)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Str to a Name (line 26):
    str_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'str', '')
    # Assigning a type to the variable 'ans' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'ans', str_81)
    
    # Assigning a List to a Name (line 27):
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), list_82, int_83)
    
    # Assigning a type to the variable 'charstack' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'charstack', list_82)
    
    # Getting the type of 'string' (line 28)
    string_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'string')
    # Assigning a type to the variable 'string_84' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'string_84', string_84)
    # Testing if the for loop is going to be iterated (line 28)
    # Testing the type of a for loop iterable (line 28)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 4), string_84)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 28, 4), string_84):
        # Getting the type of the for loop variable (line 28)
        for_loop_var_85 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 4), string_84)
        # Assigning a type to the variable 'c' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'c', for_loop_var_85)
        # SSA begins for a for statement (line 28)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 29):
        # Getting the type of 'b' (line 29)
        b_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'b')
        # Getting the type of 'a' (line 29)
        a_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'a')
        # Applying the binary operator '-' (line 29)
        result_sub_88 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 10), '-', b_86, a_87)
        
        # Assigning a type to the variable 'w' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'w', result_sub_88)
        # Getting the type of 'adaptive' (line 30)
        adaptive_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'adaptive')
        # Testing if the type of an if condition is none (line 30)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 30, 8), adaptive_89):
            pass
        else:
            
            # Testing the type of an if condition (line 30)
            if_condition_90 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), adaptive_89)
            # Assigning a type to the variable 'if_condition_90' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_90', if_condition_90)
            # SSA begins for if statement (line 30)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 31):
            # Getting the type of 'c0' (line 31)
            c0_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'c0')
            # Getting the type of 'c1' (line 31)
            c1_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'c1')
            # Applying the binary operator '+' (line 31)
            result_add_93 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 17), '+', c0_91, c1_92)
            
            # Assigning a type to the variable 'cT' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'cT', result_add_93)
            
            # Assigning a BinOp to a Name (line 32):
            # Getting the type of 'c0' (line 32)
            c0_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'c0')
            float_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'float')
            # Applying the binary operator '*' (line 32)
            result_mul_96 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 17), '*', c0_94, float_95)
            
            # Getting the type of 'cT' (line 32)
            cT_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'cT')
            # Applying the binary operator 'div' (line 32)
            result_div_98 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 23), 'div', result_mul_96, cT_97)
            
            # Assigning a type to the variable 'p0' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'p0', result_div_98)
            pass
            # SSA join for if statement (line 30)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'a' (line 34)
        a_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'a')
        
        # Call to int(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'p0' (line 34)
        p0_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'p0', False)
        # Getting the type of 'w' (line 34)
        w_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'w', False)
        # Applying the binary operator '*' (line 34)
        result_mul_103 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 27), '*', p0_101, w_102)
        
        # Processing the call keyword arguments (line 34)
        kwargs_104 = {}
        # Getting the type of 'int' (line 34)
        int_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'int', False)
        # Calling int(args, kwargs) (line 34)
        int_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 34, 23), int_100, *[result_mul_103], **kwargs_104)
        
        # Applying the binary operator '+' (line 34)
        result_add_106 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 19), '+', a_99, int_call_result_105)
        
        # Assigning a type to the variable 'boundary' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'boundary', result_add_106)
        
        # Getting the type of 'boundary' (line 35)
        boundary_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'boundary')
        # Getting the type of 'a' (line 35)
        a_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'a')
        # Applying the binary operator '==' (line 35)
        result_eq_109 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), '==', boundary_107, a_108)
        
        # Testing if the type of an if condition is none (line 35)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 8), result_eq_109):
            pass
        else:
            
            # Testing the type of an if condition (line 35)
            if_condition_110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_eq_109)
            # Assigning a type to the variable 'if_condition_110' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_110', if_condition_110)
            # SSA begins for if statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'boundary' (line 35)
            boundary_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'boundary')
            int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'int')
            # Applying the binary operator '+=' (line 35)
            result_iadd_113 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 28), '+=', boundary_111, int_112)
            # Assigning a type to the variable 'boundary' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'boundary', result_iadd_113)
            
            str_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 49), 'str', 'warningA')
            pass
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'boundary' (line 36)
        boundary_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'boundary')
        # Getting the type of 'b' (line 36)
        b_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'b')
        # Applying the binary operator '==' (line 36)
        result_eq_117 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '==', boundary_115, b_116)
        
        # Testing if the type of an if condition is none (line 36)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 8), result_eq_117):
            pass
        else:
            
            # Testing the type of an if condition (line 36)
            if_condition_118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), result_eq_117)
            # Assigning a type to the variable 'if_condition_118' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_118', if_condition_118)
            # SSA begins for if statement (line 36)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'boundary' (line 36)
            boundary_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'boundary')
            int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 40), 'int')
            # Applying the binary operator '-=' (line 36)
            result_isub_121 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 28), '-=', boundary_119, int_120)
            # Assigning a type to the variable 'boundary' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'boundary', result_isub_121)
            
            str_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 49), 'str', 'warningB')
            pass
            # SSA join for if statement (line 36)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'c' (line 38)
        c_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'c')
        str_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'str', '1')
        # Applying the binary operator '==' (line 38)
        result_eq_125 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 12), '==', c_123, str_124)
        
        # Testing if the type of an if condition is none (line 38)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 8), result_eq_125):
            
            # Getting the type of 'c' (line 42)
            c_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'c')
            str_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'str', '0')
            # Applying the binary operator '==' (line 42)
            result_eq_138 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 14), '==', c_136, str_137)
            
            # Testing if the type of an if condition is none (line 42)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 13), result_eq_138):
                pass
            else:
                
                # Testing the type of an if condition (line 42)
                if_condition_139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 13), result_eq_138)
                # Assigning a type to the variable 'if_condition_139' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'if_condition_139', if_condition_139)
                # SSA begins for if statement (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 43):
                # Getting the type of 'boundary' (line 43)
                boundary_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'boundary')
                # Assigning a type to the variable 'b' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'b', boundary_140)
                
                # Getting the type of 'tot0' (line 44)
                tot0_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'tot0')
                int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
                # Applying the binary operator '+=' (line 44)
                result_iadd_143 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '+=', tot0_141, int_142)
                # Assigning a type to the variable 'tot0' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'tot0', result_iadd_143)
                
                # Getting the type of 'adaptive' (line 45)
                adaptive_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'adaptive')
                # Testing if the type of an if condition is none (line 45)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 12), adaptive_144):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 45)
                    if_condition_145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 12), adaptive_144)
                    # Assigning a type to the variable 'if_condition_145' (line 45)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'if_condition_145', if_condition_145)
                    # SSA begins for if statement (line 45)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'c0' (line 45)
                    c0_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'c0')
                    float_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'float')
                    # Applying the binary operator '+=' (line 45)
                    result_iadd_148 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 25), '+=', c0_146, float_147)
                    # Assigning a type to the variable 'c0' (line 45)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'c0', result_iadd_148)
                    
                    pass
                    # SSA join for if statement (line 45)
                    module_type_store = module_type_store.join_ssa_context()
                    

                pass
                # SSA join for if statement (line 42)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 38)
            if_condition_126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), result_eq_125)
            # Assigning a type to the variable 'if_condition_126' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_126', if_condition_126)
            # SSA begins for if statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 39):
            # Getting the type of 'boundary' (line 39)
            boundary_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'boundary')
            # Assigning a type to the variable 'a' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'a', boundary_127)
            
            # Getting the type of 'tot1' (line 40)
            tot1_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tot1')
            int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'int')
            # Applying the binary operator '+=' (line 40)
            result_iadd_130 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 12), '+=', tot1_128, int_129)
            # Assigning a type to the variable 'tot1' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tot1', result_iadd_130)
            
            # Getting the type of 'adaptive' (line 41)
            adaptive_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'adaptive')
            # Testing if the type of an if condition is none (line 41)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 12), adaptive_131):
                pass
            else:
                
                # Testing the type of an if condition (line 41)
                if_condition_132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), adaptive_131)
                # Assigning a type to the variable 'if_condition_132' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_132', if_condition_132)
                # SSA begins for if statement (line 41)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'c1' (line 41)
                c1_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'c1')
                float_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'float')
                # Applying the binary operator '+=' (line 41)
                result_iadd_135 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '+=', c1_133, float_134)
                # Assigning a type to the variable 'c1' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'c1', result_iadd_135)
                
                pass
                # SSA join for if statement (line 41)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 38)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'c' (line 42)
            c_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'c')
            str_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'str', '0')
            # Applying the binary operator '==' (line 42)
            result_eq_138 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 14), '==', c_136, str_137)
            
            # Testing if the type of an if condition is none (line 42)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 13), result_eq_138):
                pass
            else:
                
                # Testing the type of an if condition (line 42)
                if_condition_139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 13), result_eq_138)
                # Assigning a type to the variable 'if_condition_139' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'if_condition_139', if_condition_139)
                # SSA begins for if statement (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 43):
                # Getting the type of 'boundary' (line 43)
                boundary_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'boundary')
                # Assigning a type to the variable 'b' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'b', boundary_140)
                
                # Getting the type of 'tot0' (line 44)
                tot0_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'tot0')
                int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
                # Applying the binary operator '+=' (line 44)
                result_iadd_143 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '+=', tot0_141, int_142)
                # Assigning a type to the variable 'tot0' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'tot0', result_iadd_143)
                
                # Getting the type of 'adaptive' (line 45)
                adaptive_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'adaptive')
                # Testing if the type of an if condition is none (line 45)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 12), adaptive_144):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 45)
                    if_condition_145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 12), adaptive_144)
                    # Assigning a type to the variable 'if_condition_145' (line 45)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'if_condition_145', if_condition_145)
                    # SSA begins for if statement (line 45)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'c0' (line 45)
                    c0_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'c0')
                    float_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'float')
                    # Applying the binary operator '+=' (line 45)
                    result_iadd_148 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 25), '+=', c0_146, float_147)
                    # Assigning a type to the variable 'c0' (line 45)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'c0', result_iadd_148)
                    
                    pass
                    # SSA join for if statement (line 45)
                    module_type_store = module_type_store.join_ssa_context()
                    

                pass
                # SSA join for if statement (line 42)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'a' (line 48)
        a_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'a')
        # Getting the type of 'HALF' (line 48)
        HALF_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'HALF')
        # Applying the binary operator '>=' (line 48)
        result_ge_151 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 17), '>=', a_149, HALF_150)
        
        
        # Getting the type of 'b' (line 48)
        b_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'b')
        # Getting the type of 'HALF' (line 48)
        HALF_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'HALF')
        # Applying the binary operator '<=' (line 48)
        result_le_154 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 30), '<=', b_152, HALF_153)
        
        # Applying the binary operator 'or' (line 48)
        result_or_keyword_155 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 16), 'or', result_ge_151, result_le_154)
        
        # Assigning a type to the variable 'result_or_keyword_155' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'result_or_keyword_155', result_or_keyword_155)
        # Testing if the while is going to be iterated (line 48)
        # Testing the type of an if condition (line 48)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_or_keyword_155)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 8), result_or_keyword_155):
            # SSA begins for while statement (line 48)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 'a' (line 49)
            a_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'a')
            # Getting the type of 'HALF' (line 49)
            HALF_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'HALF')
            # Applying the binary operator '>=' (line 49)
            result_ge_158 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '>=', a_156, HALF_157)
            
            # Testing if the type of an if condition is none (line 49)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 12), result_ge_158):
                
                # Assigning a BinOp to a Name (line 54):
                # Getting the type of 'ans' (line 54)
                ans_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'ans')
                
                # Call to clear(...): (line 54)
                # Processing the call arguments (line 54)
                int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'int')
                # Getting the type of 'charstack' (line 54)
                charstack_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'charstack', False)
                # Processing the call keyword arguments (line 54)
                kwargs_177 = {}
                # Getting the type of 'clear' (line 54)
                clear_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'clear', False)
                # Calling clear(args, kwargs) (line 54)
                clear_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), clear_174, *[int_175, charstack_176], **kwargs_177)
                
                # Applying the binary operator '+' (line 54)
                result_add_179 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 22), '+', ans_173, clear_call_result_178)
                
                # Assigning a type to the variable 'ans' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'ans', result_add_179)
                pass
            else:
                
                # Testing the type of an if condition (line 49)
                if_condition_159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 12), result_ge_158)
                # Assigning a type to the variable 'if_condition_159' (line 49)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'if_condition_159', if_condition_159)
                # SSA begins for if statement (line 49)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 50):
                # Getting the type of 'ans' (line 50)
                ans_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'ans')
                
                # Call to clear(...): (line 50)
                # Processing the call arguments (line 50)
                int_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'int')
                # Getting the type of 'charstack' (line 50)
                charstack_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'charstack', False)
                # Processing the call keyword arguments (line 50)
                kwargs_164 = {}
                # Getting the type of 'clear' (line 50)
                clear_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'clear', False)
                # Calling clear(args, kwargs) (line 50)
                clear_call_result_165 = invoke(stypy.reporting.localization.Localization(__file__, 50, 28), clear_161, *[int_162, charstack_163], **kwargs_164)
                
                # Applying the binary operator '+' (line 50)
                result_add_166 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), '+', ans_160, clear_call_result_165)
                
                # Assigning a type to the variable 'ans' (line 50)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'ans', result_add_166)
                
                # Assigning a BinOp to a Name (line 51):
                # Getting the type of 'a' (line 51)
                a_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'a')
                # Getting the type of 'HALF' (line 51)
                HALF_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'HALF')
                # Applying the binary operator '-' (line 51)
                result_sub_169 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 20), '-', a_167, HALF_168)
                
                # Assigning a type to the variable 'a' (line 51)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'a', result_sub_169)
                
                # Assigning a BinOp to a Name (line 52):
                # Getting the type of 'b' (line 52)
                b_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'b')
                # Getting the type of 'HALF' (line 52)
                HALF_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'HALF')
                # Applying the binary operator '-' (line 52)
                result_sub_172 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), '-', b_170, HALF_171)
                
                # Assigning a type to the variable 'b' (line 52)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'b', result_sub_172)
                # SSA branch for the else part of an if statement (line 49)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a BinOp to a Name (line 54):
                # Getting the type of 'ans' (line 54)
                ans_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'ans')
                
                # Call to clear(...): (line 54)
                # Processing the call arguments (line 54)
                int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'int')
                # Getting the type of 'charstack' (line 54)
                charstack_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'charstack', False)
                # Processing the call keyword arguments (line 54)
                kwargs_177 = {}
                # Getting the type of 'clear' (line 54)
                clear_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'clear', False)
                # Calling clear(args, kwargs) (line 54)
                clear_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), clear_174, *[int_175, charstack_176], **kwargs_177)
                
                # Applying the binary operator '+' (line 54)
                result_add_179 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 22), '+', ans_173, clear_call_result_178)
                
                # Assigning a type to the variable 'ans' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'ans', result_add_179)
                pass
                # SSA join for if statement (line 49)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'a' (line 56)
            a_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'a')
            int_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'int')
            # Applying the binary operator '*=' (line 56)
            result_imul_182 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '*=', a_180, int_181)
            # Assigning a type to the variable 'a' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'a', result_imul_182)
            
            
            # Getting the type of 'b' (line 56)
            b_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'b')
            int_184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'int')
            # Applying the binary operator '*=' (line 56)
            result_imul_185 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 26), '*=', b_183, int_184)
            # Assigning a type to the variable 'b' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'b', result_imul_185)
            
            pass
            # SSA join for while statement (line 48)
            module_type_store = module_type_store.join_ssa_context()

        
        # Evaluating assert statement condition
        
        # Getting the type of 'a' (line 59)
        a_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'a')
        # Getting the type of 'HALF' (line 59)
        HALF_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'HALF')
        # Applying the binary operator '<=' (line 59)
        result_le_188 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), '<=', a_186, HALF_187)
        
        assert_189 = result_le_188
        # Assigning a type to the variable 'assert_189' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_189', result_le_188)
        # Evaluating assert statement condition
        
        # Getting the type of 'b' (line 59)
        b_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'b')
        # Getting the type of 'HALF' (line 59)
        HALF_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'HALF')
        # Applying the binary operator '>=' (line 59)
        result_ge_192 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 31), '>=', b_190, HALF_191)
        
        assert_193 = result_ge_192
        # Assigning a type to the variable 'assert_193' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'assert_193', result_ge_192)
        # Evaluating assert statement condition
        
        # Getting the type of 'a' (line 59)
        a_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 47), 'a')
        int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 50), 'int')
        # Applying the binary operator '>=' (line 59)
        result_ge_196 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 47), '>=', a_194, int_195)
        
        assert_197 = result_ge_196
        # Assigning a type to the variable 'assert_197' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'assert_197', result_ge_196)
        # Evaluating assert statement condition
        
        # Getting the type of 'b' (line 59)
        b_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 60), 'b')
        # Getting the type of 'ONE' (line 59)
        ONE_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 63), 'ONE')
        # Applying the binary operator '<=' (line 59)
        result_le_200 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 60), '<=', b_198, ONE_199)
        
        assert_201 = result_le_200
        # Assigning a type to the variable 'assert_201' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 53), 'assert_201', result_le_200)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'a' (line 61)
        a_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'a')
        # Getting the type of 'QUARTER' (line 61)
        QUARTER_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'QUARTER')
        # Applying the binary operator '>' (line 61)
        result_gt_204 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 17), '>', a_202, QUARTER_203)
        
        
        # Getting the type of 'b' (line 61)
        b_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'b')
        # Getting the type of 'THREEQU' (line 61)
        THREEQU_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'THREEQU')
        # Applying the binary operator '<' (line 61)
        result_lt_207 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 33), '<', b_205, THREEQU_206)
        
        # Applying the binary operator 'and' (line 61)
        result_and_keyword_208 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 16), 'and', result_gt_204, result_lt_207)
        
        # Assigning a type to the variable 'result_and_keyword_208' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'result_and_keyword_208', result_and_keyword_208)
        # Testing if the while is going to be iterated (line 61)
        # Testing the type of an if condition (line 61)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), result_and_keyword_208)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 8), result_and_keyword_208):
            # SSA begins for while statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 'charstack' (line 62)
            charstack_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'charstack')
            
            # Obtaining the type of the subscript
            int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'int')
            # Getting the type of 'charstack' (line 62)
            charstack_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'charstack')
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), charstack_211, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), getitem___212, int_210)
            
            int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'int')
            # Applying the binary operator '+=' (line 62)
            result_iadd_215 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '+=', subscript_call_result_213, int_214)
            # Getting the type of 'charstack' (line 62)
            charstack_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'charstack')
            int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'int')
            # Storing an element on a container (line 62)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 12), charstack_216, (int_217, result_iadd_215))
            
            
            # Assigning a BinOp to a Name (line 63):
            int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'int')
            # Getting the type of 'a' (line 63)
            a_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'a')
            # Applying the binary operator '*' (line 63)
            result_mul_220 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '*', int_218, a_219)
            
            # Getting the type of 'HALF' (line 63)
            HALF_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'HALF')
            # Applying the binary operator '-' (line 63)
            result_sub_222 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '-', result_mul_220, HALF_221)
            
            # Assigning a type to the variable 'a' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'a', result_sub_222)
            
            # Assigning a BinOp to a Name (line 64):
            int_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'int')
            # Getting the type of 'b' (line 64)
            b_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'b')
            # Applying the binary operator '*' (line 64)
            result_mul_225 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '*', int_223, b_224)
            
            # Getting the type of 'HALF' (line 64)
            HALF_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'HALF')
            # Applying the binary operator '-' (line 64)
            result_sub_227 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '-', result_mul_225, HALF_226)
            
            # Assigning a type to the variable 'b' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'b', result_sub_227)
            pass
            # SSA join for while statement (line 61)
            module_type_store = module_type_store.join_ssa_context()

        
        # Evaluating assert statement condition
        
        # Getting the type of 'a' (line 67)
        a_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'a')
        # Getting the type of 'HALF' (line 67)
        HALF_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'HALF')
        # Applying the binary operator '<=' (line 67)
        result_le_230 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), '<=', a_228, HALF_229)
        
        assert_231 = result_le_230
        # Assigning a type to the variable 'assert_231' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'assert_231', result_le_230)
        # Evaluating assert statement condition
        
        # Getting the type of 'b' (line 67)
        b_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'b')
        # Getting the type of 'HALF' (line 67)
        HALF_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'HALF')
        # Applying the binary operator '>=' (line 67)
        result_ge_234 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 31), '>=', b_232, HALF_233)
        
        assert_235 = result_ge_234
        # Assigning a type to the variable 'assert_235' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'assert_235', result_ge_234)
        # Evaluating assert statement condition
        
        # Getting the type of 'a' (line 67)
        a_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 47), 'a')
        int_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 50), 'int')
        # Applying the binary operator '>=' (line 67)
        result_ge_238 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 47), '>=', a_236, int_237)
        
        assert_239 = result_ge_238
        # Assigning a type to the variable 'assert_239' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'assert_239', result_ge_238)
        # Evaluating assert statement condition
        
        # Getting the type of 'b' (line 67)
        b_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 60), 'b')
        # Getting the type of 'ONE' (line 67)
        ONE_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 63), 'ONE')
        # Applying the binary operator '<=' (line 67)
        result_le_242 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 60), '<=', b_240, ONE_241)
        
        assert_243 = result_le_242
        # Assigning a type to the variable 'assert_243' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 53), 'assert_243', result_le_242)
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'HALF' (line 71)
    HALF_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'HALF')
    # Getting the type of 'a' (line 71)
    a_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'a')
    # Applying the binary operator '-' (line 71)
    result_sub_246 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 10), '-', HALF_244, a_245)
    
    # Getting the type of 'b' (line 71)
    b_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'b')
    # Getting the type of 'HALF' (line 71)
    HALF_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'HALF')
    # Applying the binary operator '-' (line 71)
    result_sub_249 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 21), '-', b_247, HALF_248)
    
    # Applying the binary operator '>' (line 71)
    result_gt_250 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), '>', result_sub_246, result_sub_249)
    
    # Testing if the type of an if condition is none (line 71)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 4), result_gt_250):
        
        # Assigning a BinOp to a Name (line 80):
        # Getting the type of 'b' (line 80)
        b_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'b')
        # Getting the type of 'HALF' (line 80)
        HALF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'HALF')
        # Applying the binary operator '-' (line 80)
        result_sub_277 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '-', b_275, HALF_276)
        
        # Assigning a type to the variable 'w' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'w', result_sub_277)
        
        # Assigning a BinOp to a Name (line 81):
        # Getting the type of 'ans' (line 81)
        ans_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'ans')
        
        # Call to clear(...): (line 81)
        # Processing the call arguments (line 81)
        int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'int')
        # Getting the type of 'charstack' (line 81)
        charstack_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'charstack', False)
        # Processing the call keyword arguments (line 81)
        kwargs_282 = {}
        # Getting the type of 'clear' (line 81)
        clear_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'clear', False)
        # Calling clear(args, kwargs) (line 81)
        clear_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 81, 20), clear_279, *[int_280, charstack_281], **kwargs_282)
        
        # Applying the binary operator '+' (line 81)
        result_add_284 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 14), '+', ans_278, clear_call_result_283)
        
        # Assigning a type to the variable 'ans' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'ans', result_add_284)
        
        
        # Getting the type of 'w' (line 82)
        w_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'w')
        # Getting the type of 'HALF' (line 82)
        HALF_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'HALF')
        # Applying the binary operator '<' (line 82)
        result_lt_287 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 16), '<', w_285, HALF_286)
        
        # Assigning a type to the variable 'result_lt_287' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'result_lt_287', result_lt_287)
        # Testing if the while is going to be iterated (line 82)
        # Testing the type of an if condition (line 82)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_lt_287)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 82, 8), result_lt_287):
            # SSA begins for while statement (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a BinOp to a Name (line 83):
            # Getting the type of 'ans' (line 83)
            ans_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'ans')
            
            # Call to clear(...): (line 83)
            # Processing the call arguments (line 83)
            int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'int')
            # Getting the type of 'charstack' (line 83)
            charstack_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'charstack', False)
            # Processing the call keyword arguments (line 83)
            kwargs_292 = {}
            # Getting the type of 'clear' (line 83)
            clear_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'clear', False)
            # Calling clear(args, kwargs) (line 83)
            clear_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), clear_289, *[int_290, charstack_291], **kwargs_292)
            
            # Applying the binary operator '+' (line 83)
            result_add_294 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 18), '+', ans_288, clear_call_result_293)
            
            # Assigning a type to the variable 'ans' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'ans', result_add_294)
            
            # Getting the type of 'w' (line 84)
            w_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'w')
            int_296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'int')
            # Applying the binary operator '*=' (line 84)
            result_imul_297 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '*=', w_295, int_296)
            # Assigning a type to the variable 'w' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'w', result_imul_297)
            
            pass
            # SSA join for while statement (line 82)
            module_type_store = module_type_store.join_ssa_context()

        
        pass
    else:
        
        # Testing the type of an if condition (line 71)
        if_condition_251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_gt_250)
        # Assigning a type to the variable 'if_condition_251' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_251', if_condition_251)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 72):
        # Getting the type of 'HALF' (line 72)
        HALF_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'HALF')
        # Getting the type of 'a' (line 72)
        a_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'a')
        # Applying the binary operator '-' (line 72)
        result_sub_254 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '-', HALF_252, a_253)
        
        # Assigning a type to the variable 'w' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'w', result_sub_254)
        
        # Assigning a BinOp to a Name (line 73):
        # Getting the type of 'ans' (line 73)
        ans_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'ans')
        
        # Call to clear(...): (line 73)
        # Processing the call arguments (line 73)
        int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'int')
        # Getting the type of 'charstack' (line 73)
        charstack_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'charstack', False)
        # Processing the call keyword arguments (line 73)
        kwargs_259 = {}
        # Getting the type of 'clear' (line 73)
        clear_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'clear', False)
        # Calling clear(args, kwargs) (line 73)
        clear_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), clear_256, *[int_257, charstack_258], **kwargs_259)
        
        # Applying the binary operator '+' (line 73)
        result_add_261 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 14), '+', ans_255, clear_call_result_260)
        
        # Assigning a type to the variable 'ans' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'ans', result_add_261)
        
        
        # Getting the type of 'w' (line 74)
        w_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'w')
        # Getting the type of 'HALF' (line 74)
        HALF_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'HALF')
        # Applying the binary operator '<' (line 74)
        result_lt_264 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 16), '<', w_262, HALF_263)
        
        # Assigning a type to the variable 'result_lt_264' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'result_lt_264', result_lt_264)
        # Testing if the while is going to be iterated (line 74)
        # Testing the type of an if condition (line 74)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), result_lt_264)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 74, 8), result_lt_264):
            # SSA begins for while statement (line 74)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a BinOp to a Name (line 75):
            # Getting the type of 'ans' (line 75)
            ans_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'ans')
            
            # Call to clear(...): (line 75)
            # Processing the call arguments (line 75)
            int_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 30), 'int')
            # Getting the type of 'charstack' (line 75)
            charstack_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'charstack', False)
            # Processing the call keyword arguments (line 75)
            kwargs_269 = {}
            # Getting the type of 'clear' (line 75)
            clear_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'clear', False)
            # Calling clear(args, kwargs) (line 75)
            clear_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 75, 24), clear_266, *[int_267, charstack_268], **kwargs_269)
            
            # Applying the binary operator '+' (line 75)
            result_add_271 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 18), '+', ans_265, clear_call_result_270)
            
            # Assigning a type to the variable 'ans' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'ans', result_add_271)
            
            # Getting the type of 'w' (line 76)
            w_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'w')
            int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 16), 'int')
            # Applying the binary operator '*=' (line 76)
            result_imul_274 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 12), '*=', w_272, int_273)
            # Assigning a type to the variable 'w' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'w', result_imul_274)
            
            pass
            # SSA join for while statement (line 74)
            module_type_store = module_type_store.join_ssa_context()

        
        pass
        # SSA branch for the else part of an if statement (line 71)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 80):
        # Getting the type of 'b' (line 80)
        b_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'b')
        # Getting the type of 'HALF' (line 80)
        HALF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'HALF')
        # Applying the binary operator '-' (line 80)
        result_sub_277 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '-', b_275, HALF_276)
        
        # Assigning a type to the variable 'w' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'w', result_sub_277)
        
        # Assigning a BinOp to a Name (line 81):
        # Getting the type of 'ans' (line 81)
        ans_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'ans')
        
        # Call to clear(...): (line 81)
        # Processing the call arguments (line 81)
        int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'int')
        # Getting the type of 'charstack' (line 81)
        charstack_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'charstack', False)
        # Processing the call keyword arguments (line 81)
        kwargs_282 = {}
        # Getting the type of 'clear' (line 81)
        clear_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'clear', False)
        # Calling clear(args, kwargs) (line 81)
        clear_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 81, 20), clear_279, *[int_280, charstack_281], **kwargs_282)
        
        # Applying the binary operator '+' (line 81)
        result_add_284 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 14), '+', ans_278, clear_call_result_283)
        
        # Assigning a type to the variable 'ans' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'ans', result_add_284)
        
        
        # Getting the type of 'w' (line 82)
        w_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'w')
        # Getting the type of 'HALF' (line 82)
        HALF_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'HALF')
        # Applying the binary operator '<' (line 82)
        result_lt_287 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 16), '<', w_285, HALF_286)
        
        # Assigning a type to the variable 'result_lt_287' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'result_lt_287', result_lt_287)
        # Testing if the while is going to be iterated (line 82)
        # Testing the type of an if condition (line 82)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_lt_287)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 82, 8), result_lt_287):
            # SSA begins for while statement (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a BinOp to a Name (line 83):
            # Getting the type of 'ans' (line 83)
            ans_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'ans')
            
            # Call to clear(...): (line 83)
            # Processing the call arguments (line 83)
            int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'int')
            # Getting the type of 'charstack' (line 83)
            charstack_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'charstack', False)
            # Processing the call keyword arguments (line 83)
            kwargs_292 = {}
            # Getting the type of 'clear' (line 83)
            clear_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'clear', False)
            # Calling clear(args, kwargs) (line 83)
            clear_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), clear_289, *[int_290, charstack_291], **kwargs_292)
            
            # Applying the binary operator '+' (line 83)
            result_add_294 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 18), '+', ans_288, clear_call_result_293)
            
            # Assigning a type to the variable 'ans' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'ans', result_add_294)
            
            # Getting the type of 'w' (line 84)
            w_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'w')
            int_296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'int')
            # Applying the binary operator '*=' (line 84)
            result_imul_297 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '*=', w_295, int_296)
            # Assigning a type to the variable 'w' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'w', result_imul_297)
            
            pass
            # SSA join for while statement (line 82)
            module_type_store = module_type_store.join_ssa_context()

        
        pass
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'ans' (line 87)
    ans_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'ans')
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type', ans_298)
    pass
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_299

# Assigning a type to the variable 'encode' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'encode', encode)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'int')
    # Getting the type of 'BETA0' (line 92)
    BETA0_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'BETA0')
    # Getting the type of 'BETA1' (line 92)
    BETA1_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'BETA1')
    int_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 58), 'int')
    int_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 68), 'int')
    defaults = [int_300, BETA0_301, BETA1_302, int_303, int_304]
    # Create a new context for function 'decode'
    module_type_store = module_type_store.open_function_context('decode', 92, 0, False)
    
    # Passed parameters checking function
    decode.stypy_localization = localization
    decode.stypy_type_of_self = None
    decode.stypy_type_store = module_type_store
    decode.stypy_function_name = 'decode'
    decode.stypy_param_names_list = ['string', 'N', 'c0', 'c1', 'adaptive', 'verbose']
    decode.stypy_varargs_param_name = None
    decode.stypy_kwargs_param_name = None
    decode.stypy_call_defaults = defaults
    decode.stypy_call_varargs = varargs
    decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode', ['string', 'N', 'c0', 'c1', 'adaptive', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode', localization, ['string', 'N', 'c0', 'c1', 'adaptive', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode(...)' code ##################

    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'ONE' (line 94)
    ONE_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 6), 'ONE')
    # Assigning a type to the variable 'b' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'b', ONE_305)
    
    # Assigning a Num to a Name (line 94):
    int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 14), 'int')
    # Assigning a type to the variable 'a' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'a', int_306)
    
    # Assigning a Num to a Name (line 94):
    int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'int')
    # Assigning a type to the variable 'tot0' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'tot0', int_307)
    
    # Assigning a Num to a Name (line 94):
    int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 35), 'int')
    # Assigning a type to the variable 'tot1' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), 'tot1', int_308)
    # Evaluating assert statement condition
    
    # Getting the type of 'c0' (line 94)
    c0_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 51), 'c0')
    int_310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 54), 'int')
    # Applying the binary operator '>' (line 94)
    result_gt_311 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 51), '>', c0_309, int_310)
    
    assert_312 = result_gt_311
    # Assigning a type to the variable 'assert_312' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'assert_312', result_gt_311)
    # Evaluating assert statement condition
    
    # Getting the type of 'c1' (line 94)
    c1_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 65), 'c1')
    int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 68), 'int')
    # Applying the binary operator '>' (line 94)
    result_gt_315 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 65), '>', c1_313, int_314)
    
    assert_316 = result_gt_315
    # Assigning a type to the variable 'assert_316' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 58), 'assert_316', result_gt_315)
    
    # Assigning a Num to a Name (line 95):
    int_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 27), 'int')
    # Assigning a type to the variable 'model_needs_updating' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'model_needs_updating', int_317)
    
    # Getting the type of 'adaptive' (line 96)
    adaptive_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'adaptive')
    int_319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'int')
    # Applying the binary operator '==' (line 96)
    result_eq_320 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '==', adaptive_318, int_319)
    
    # Testing if the type of an if condition is none (line 96)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 4), result_eq_320):
        pass
    else:
        
        # Testing the type of an if condition (line 96)
        if_condition_321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_eq_320)
        # Assigning a type to the variable 'if_condition_321' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_321', if_condition_321)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 97):
        # Getting the type of 'c0' (line 97)
        c0_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'c0')
        float_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'float')
        # Applying the binary operator '*' (line 97)
        result_mul_324 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 13), '*', c0_322, float_323)
        
        # Getting the type of 'c0' (line 97)
        c0_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'c0')
        # Getting the type of 'c1' (line 97)
        c1_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'c1')
        # Applying the binary operator '+' (line 97)
        result_add_327 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 21), '+', c0_325, c1_326)
        
        # Applying the binary operator 'div' (line 97)
        result_div_328 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 19), 'div', result_mul_324, result_add_327)
        
        # Assigning a type to the variable 'p0' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'p0', result_div_328)
        pass
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Str to a Name (line 99):
    str_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'str', '')
    # Assigning a type to the variable 'ans' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'ans', str_329)
    
    # Assigning a Num to a Name (line 100):
    int_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 6), 'int')
    # Assigning a type to the variable 'u' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'u', int_330)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'ONE' (line 100)
    ONE_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ONE')
    # Assigning a type to the variable 'v' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 10), 'v', ONE_331)
    
    # Getting the type of 'string' (line 101)
    string_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'string')
    # Assigning a type to the variable 'string_332' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'string_332', string_332)
    # Testing if the for loop is going to be iterated (line 101)
    # Testing the type of a for loop iterable (line 101)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 4), string_332)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 101, 4), string_332):
        # Getting the type of the for loop variable (line 101)
        for_loop_var_333 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 4), string_332)
        # Assigning a type to the variable 'c' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'c', for_loop_var_333)
        # SSA begins for a for statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'N' (line 102)
        N_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'N')
        int_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 14), 'int')
        # Applying the binary operator '<=' (line 102)
        result_le_336 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '<=', N_334, int_335)
        
        # Testing if the type of an if condition is none (line 102)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 8), result_le_336):
            pass
        else:
            
            # Testing the type of an if condition (line 102)
            if_condition_337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_le_336)
            # Assigning a type to the variable 'if_condition_337' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_337', if_condition_337)
            # SSA begins for if statement (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 102)
            module_type_store = module_type_store.join_ssa_context()
            

        # Evaluating assert statement condition
        
        # Getting the type of 'N' (line 104)
        N_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'N')
        int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'int')
        # Applying the binary operator '>' (line 104)
        result_gt_340 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '>', N_338, int_339)
        
        assert_341 = result_gt_340
        # Assigning a type to the variable 'assert_341' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'assert_341', result_gt_340)
        # Evaluating assert statement condition
        
        # Getting the type of 'u' (line 107)
        u_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'u')
        int_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'int')
        # Applying the binary operator '>=' (line 107)
        result_ge_344 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), '>=', u_342, int_343)
        
        assert_345 = result_ge_344
        # Assigning a type to the variable 'assert_345' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_345', result_ge_344)
        # Evaluating assert statement condition
        
        # Getting the type of 'v' (line 107)
        v_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'v')
        # Getting the type of 'ONE' (line 107)
        ONE_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'ONE')
        # Applying the binary operator '<=' (line 107)
        result_le_348 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 29), '<=', v_346, ONE_347)
        
        assert_349 = result_le_348
        # Assigning a type to the variable 'assert_349' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'assert_349', result_le_348)
        
        # Assigning a BinOp to a Name (line 108):
        # Getting the type of 'u' (line 108)
        u_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'u')
        # Getting the type of 'v' (line 108)
        v_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'v')
        # Getting the type of 'u' (line 108)
        u_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'u')
        # Applying the binary operator '-' (line 108)
        result_sub_353 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 23), '-', v_351, u_352)
        
        int_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'int')
        # Applying the binary operator 'div' (line 108)
        result_div_355 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 22), 'div', result_sub_353, int_354)
        
        # Applying the binary operator '+' (line 108)
        result_add_356 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 18), '+', u_350, result_div_355)
        
        # Assigning a type to the variable 'halfway' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'halfway', result_add_356)
        
        # Getting the type of 'c' (line 109)
        c_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'c')
        str_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'str', '1')
        # Applying the binary operator '==' (line 109)
        result_eq_359 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 12), '==', c_357, str_358)
        
        # Testing if the type of an if condition is none (line 109)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_359):
            
            # Getting the type of 'c' (line 111)
            c_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'c')
            str_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 18), 'str', '0')
            # Applying the binary operator '==' (line 111)
            result_eq_364 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '==', c_362, str_363)
            
            # Testing if the type of an if condition is none (line 111)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 111, 13), result_eq_364):
                pass
            else:
                
                # Testing the type of an if condition (line 111)
                if_condition_365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 13), result_eq_364)
                # Assigning a type to the variable 'if_condition_365' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'if_condition_365', if_condition_365)
                # SSA begins for if statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 112):
                # Getting the type of 'halfway' (line 112)
                halfway_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'halfway')
                # Assigning a type to the variable 'v' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'v', halfway_366)
                # SSA branch for the else part of an if statement (line 111)
                module_type_store.open_ssa_branch('else')
                pass
                # SSA join for if statement (line 111)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 109)
            if_condition_360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_359)
            # Assigning a type to the variable 'if_condition_360' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_360', if_condition_360)
            # SSA begins for if statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 110):
            # Getting the type of 'halfway' (line 110)
            halfway_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'halfway')
            # Assigning a type to the variable 'u' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'u', halfway_361)
            # SSA branch for the else part of an if statement (line 109)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'c' (line 111)
            c_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'c')
            str_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 18), 'str', '0')
            # Applying the binary operator '==' (line 111)
            result_eq_364 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '==', c_362, str_363)
            
            # Testing if the type of an if condition is none (line 111)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 111, 13), result_eq_364):
                pass
            else:
                
                # Testing the type of an if condition (line 111)
                if_condition_365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 13), result_eq_364)
                # Assigning a type to the variable 'if_condition_365' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'if_condition_365', if_condition_365)
                # SSA begins for if statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 112):
                # Getting the type of 'halfway' (line 112)
                halfway_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'halfway')
                # Assigning a type to the variable 'v' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'v', halfway_366)
                # SSA branch for the else part of an if statement (line 111)
                module_type_store.open_ssa_branch('else')
                pass
                # SSA join for if statement (line 111)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()
            

        
        int_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 15), 'int')
        # Assigning a type to the variable 'int_367' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'int_367', int_367)
        # Testing if the while is going to be iterated (line 117)
        # Testing the type of an if condition (line 117)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), int_367)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 117, 8), int_367):
            # SSA begins for while statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Num to a Name (line 118):
            int_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'int')
            # Assigning a type to the variable 'firsttime' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'firsttime', int_368)
            # Getting the type of 'model_needs_updating' (line 119)
            model_needs_updating_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'model_needs_updating')
            # Testing if the type of an if condition is none (line 119)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 119, 12), model_needs_updating_369):
                pass
            else:
                
                # Testing the type of an if condition (line 119)
                if_condition_370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), model_needs_updating_369)
                # Assigning a type to the variable 'if_condition_370' (line 119)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'if_condition_370', if_condition_370)
                # SSA begins for if statement (line 119)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 120):
                # Getting the type of 'b' (line 120)
                b_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'b')
                # Getting the type of 'a' (line 120)
                a_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'a')
                # Applying the binary operator '-' (line 120)
                result_sub_373 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 20), '-', b_371, a_372)
                
                # Assigning a type to the variable 'w' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'w', result_sub_373)
                # Getting the type of 'adaptive' (line 121)
                adaptive_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'adaptive')
                # Testing if the type of an if condition is none (line 121)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 16), adaptive_374):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 121)
                    if_condition_375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), adaptive_374)
                    # Assigning a type to the variable 'if_condition_375' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_375', if_condition_375)
                    # SSA begins for if statement (line 121)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 122):
                    # Getting the type of 'c0' (line 122)
                    c0_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'c0')
                    # Getting the type of 'c1' (line 122)
                    c1_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'c1')
                    # Applying the binary operator '+' (line 122)
                    result_add_378 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 25), '+', c0_376, c1_377)
                    
                    # Assigning a type to the variable 'cT' (line 122)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'cT', result_add_378)
                    
                    # Assigning a BinOp to a Name (line 122):
                    # Getting the type of 'c0' (line 122)
                    c0_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 42), 'c0')
                    float_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 46), 'float')
                    # Applying the binary operator '*' (line 122)
                    result_mul_381 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 42), '*', c0_379, float_380)
                    
                    # Getting the type of 'cT' (line 122)
                    cT_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 50), 'cT')
                    # Applying the binary operator 'div' (line 122)
                    result_div_383 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 49), 'div', result_mul_381, cT_382)
                    
                    # Assigning a type to the variable 'p0' (line 122)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'p0', result_div_383)
                    pass
                    # SSA join for if statement (line 121)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a BinOp to a Name (line 124):
                # Getting the type of 'a' (line 124)
                a_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'a')
                
                # Call to int(...): (line 124)
                # Processing the call arguments (line 124)
                # Getting the type of 'p0' (line 124)
                p0_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'p0', False)
                # Getting the type of 'w' (line 124)
                w_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 38), 'w', False)
                # Applying the binary operator '*' (line 124)
                result_mul_388 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 35), '*', p0_386, w_387)
                
                # Processing the call keyword arguments (line 124)
                kwargs_389 = {}
                # Getting the type of 'int' (line 124)
                int_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'int', False)
                # Calling int(args, kwargs) (line 124)
                int_call_result_390 = invoke(stypy.reporting.localization.Localization(__file__, 124, 31), int_385, *[result_mul_388], **kwargs_389)
                
                # Applying the binary operator '+' (line 124)
                result_add_391 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 27), '+', a_384, int_call_result_390)
                
                # Assigning a type to the variable 'boundary' (line 124)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'boundary', result_add_391)
                
                # Getting the type of 'boundary' (line 125)
                boundary_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'boundary')
                # Getting the type of 'a' (line 125)
                a_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'a')
                # Applying the binary operator '==' (line 125)
                result_eq_394 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 20), '==', boundary_392, a_393)
                
                # Testing if the type of an if condition is none (line 125)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 16), result_eq_394):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 125)
                    if_condition_395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_eq_394)
                    # Assigning a type to the variable 'if_condition_395' (line 125)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'if_condition_395', if_condition_395)
                    # SSA begins for if statement (line 125)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'boundary' (line 125)
                    boundary_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'boundary')
                    int_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 48), 'int')
                    # Applying the binary operator '+=' (line 125)
                    result_iadd_398 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 36), '+=', boundary_396, int_397)
                    # Assigning a type to the variable 'boundary' (line 125)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'boundary', result_iadd_398)
                    
                    str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 57), 'str', 'warningA')
                    pass
                    # SSA join for if statement (line 125)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'boundary' (line 126)
                boundary_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'boundary')
                # Getting the type of 'b' (line 126)
                b_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'b')
                # Applying the binary operator '==' (line 126)
                result_eq_402 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 20), '==', boundary_400, b_401)
                
                # Testing if the type of an if condition is none (line 126)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 16), result_eq_402):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 126)
                    if_condition_403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 16), result_eq_402)
                    # Assigning a type to the variable 'if_condition_403' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'if_condition_403', if_condition_403)
                    # SSA begins for if statement (line 126)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'boundary' (line 126)
                    boundary_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'boundary')
                    int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 48), 'int')
                    # Applying the binary operator '-=' (line 126)
                    result_isub_406 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 36), '-=', boundary_404, int_405)
                    # Assigning a type to the variable 'boundary' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'boundary', result_isub_406)
                    
                    str_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 57), 'str', 'warningB')
                    pass
                    # SSA join for if statement (line 126)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Num to a Name (line 127):
                int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'int')
                # Assigning a type to the variable 'model_needs_updating' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'model_needs_updating', int_408)
                pass
                # SSA join for if statement (line 119)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'boundary' (line 129)
            boundary_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'boundary')
            # Getting the type of 'u' (line 129)
            u_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'u')
            # Applying the binary operator '<=' (line 129)
            result_le_411 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 18), '<=', boundary_409, u_410)
            
            # Testing if the type of an if condition is none (line 129)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 129, 12), result_le_411):
                
                # Getting the type of 'boundary' (line 133)
                boundary_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'boundary')
                # Getting the type of 'v' (line 133)
                v_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'v')
                # Applying the binary operator '>=' (line 133)
                result_ge_431 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '>=', boundary_429, v_430)
                
                # Testing if the type of an if condition is none (line 133)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 133, 17), result_ge_431):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 133)
                    if_condition_432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 17), result_ge_431)
                    # Assigning a type to the variable 'if_condition_432' (line 133)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'if_condition_432', if_condition_432)
                    # SSA begins for if statement (line 133)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 134):
                    # Getting the type of 'ans' (line 134)
                    ans_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'ans')
                    str_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 28), 'str', '0')
                    # Applying the binary operator '+' (line 134)
                    result_add_435 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 22), '+', ans_433, str_434)
                    
                    # Assigning a type to the variable 'ans' (line 134)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'ans', result_add_435)
                    
                    # Getting the type of 'tot0' (line 134)
                    tot0_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'tot0')
                    int_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 52), 'int')
                    # Applying the binary operator '+=' (line 134)
                    result_iadd_438 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 45), '+=', tot0_436, int_437)
                    # Assigning a type to the variable 'tot0' (line 134)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'tot0', result_iadd_438)
                    
                    # Getting the type of 'adaptive' (line 135)
                    adaptive_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'adaptive')
                    # Testing if the type of an if condition is none (line 135)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 16), adaptive_439):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 135)
                        if_condition_440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 16), adaptive_439)
                        # Assigning a type to the variable 'if_condition_440' (line 135)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'if_condition_440', if_condition_440)
                        # SSA begins for if statement (line 135)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'c0' (line 135)
                        c0_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'c0')
                        float_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'float')
                        # Applying the binary operator '+=' (line 135)
                        result_iadd_443 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 29), '+=', c0_441, float_442)
                        # Assigning a type to the variable 'c0' (line 135)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'c0', result_iadd_443)
                        
                        pass
                        # SSA join for if statement (line 135)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Name to a Name (line 136):
                    # Getting the type of 'boundary' (line 136)
                    boundary_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'boundary')
                    # Assigning a type to the variable 'b' (line 136)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'b', boundary_444)
                    
                    # Assigning a Num to a Name (line 136):
                    int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 54), 'int')
                    # Assigning a type to the variable 'model_needs_updating' (line 136)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'model_needs_updating', int_445)
                    
                    # Getting the type of 'N' (line 136)
                    N_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'N')
                    int_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 62), 'int')
                    # Applying the binary operator '-=' (line 136)
                    result_isub_448 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 59), '-=', N_446, int_447)
                    # Assigning a type to the variable 'N' (line 136)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'N', result_isub_448)
                    
                    # SSA branch for the else part of an if statement (line 133)
                    module_type_store.open_ssa_branch('else')
                    pass
                    # SSA join for if statement (line 133)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 129)
                if_condition_412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 12), result_le_411)
                # Assigning a type to the variable 'if_condition_412' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'if_condition_412', if_condition_412)
                # SSA begins for if statement (line 129)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 130):
                # Getting the type of 'ans' (line 130)
                ans_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'ans')
                str_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 28), 'str', '1')
                # Applying the binary operator '+' (line 130)
                result_add_415 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 22), '+', ans_413, str_414)
                
                # Assigning a type to the variable 'ans' (line 130)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'ans', result_add_415)
                
                # Getting the type of 'tot1' (line 130)
                tot1_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 45), 'tot1')
                int_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 52), 'int')
                # Applying the binary operator '+=' (line 130)
                result_iadd_418 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 45), '+=', tot1_416, int_417)
                # Assigning a type to the variable 'tot1' (line 130)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 45), 'tot1', result_iadd_418)
                
                # Getting the type of 'adaptive' (line 131)
                adaptive_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'adaptive')
                # Testing if the type of an if condition is none (line 131)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 131, 16), adaptive_419):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 131)
                    if_condition_420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 16), adaptive_419)
                    # Assigning a type to the variable 'if_condition_420' (line 131)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'if_condition_420', if_condition_420)
                    # SSA begins for if statement (line 131)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'c1' (line 131)
                    c1_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'c1')
                    float_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 35), 'float')
                    # Applying the binary operator '+=' (line 131)
                    result_iadd_423 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 29), '+=', c1_421, float_422)
                    # Assigning a type to the variable 'c1' (line 131)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'c1', result_iadd_423)
                    
                    pass
                    # SSA join for if statement (line 131)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 132):
                # Getting the type of 'boundary' (line 132)
                boundary_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'boundary')
                # Assigning a type to the variable 'a' (line 132)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'a', boundary_424)
                
                # Assigning a Num to a Name (line 132):
                int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'int')
                # Assigning a type to the variable 'model_needs_updating' (line 132)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'model_needs_updating', int_425)
                
                # Getting the type of 'N' (line 132)
                N_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 59), 'N')
                int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 62), 'int')
                # Applying the binary operator '-=' (line 132)
                result_isub_428 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 59), '-=', N_426, int_427)
                # Assigning a type to the variable 'N' (line 132)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 59), 'N', result_isub_428)
                
                # SSA branch for the else part of an if statement (line 129)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'boundary' (line 133)
                boundary_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'boundary')
                # Getting the type of 'v' (line 133)
                v_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'v')
                # Applying the binary operator '>=' (line 133)
                result_ge_431 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '>=', boundary_429, v_430)
                
                # Testing if the type of an if condition is none (line 133)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 133, 17), result_ge_431):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 133)
                    if_condition_432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 17), result_ge_431)
                    # Assigning a type to the variable 'if_condition_432' (line 133)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'if_condition_432', if_condition_432)
                    # SSA begins for if statement (line 133)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 134):
                    # Getting the type of 'ans' (line 134)
                    ans_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'ans')
                    str_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 28), 'str', '0')
                    # Applying the binary operator '+' (line 134)
                    result_add_435 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 22), '+', ans_433, str_434)
                    
                    # Assigning a type to the variable 'ans' (line 134)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'ans', result_add_435)
                    
                    # Getting the type of 'tot0' (line 134)
                    tot0_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'tot0')
                    int_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 52), 'int')
                    # Applying the binary operator '+=' (line 134)
                    result_iadd_438 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 45), '+=', tot0_436, int_437)
                    # Assigning a type to the variable 'tot0' (line 134)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'tot0', result_iadd_438)
                    
                    # Getting the type of 'adaptive' (line 135)
                    adaptive_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'adaptive')
                    # Testing if the type of an if condition is none (line 135)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 16), adaptive_439):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 135)
                        if_condition_440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 16), adaptive_439)
                        # Assigning a type to the variable 'if_condition_440' (line 135)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'if_condition_440', if_condition_440)
                        # SSA begins for if statement (line 135)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'c0' (line 135)
                        c0_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'c0')
                        float_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'float')
                        # Applying the binary operator '+=' (line 135)
                        result_iadd_443 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 29), '+=', c0_441, float_442)
                        # Assigning a type to the variable 'c0' (line 135)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'c0', result_iadd_443)
                        
                        pass
                        # SSA join for if statement (line 135)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Name to a Name (line 136):
                    # Getting the type of 'boundary' (line 136)
                    boundary_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'boundary')
                    # Assigning a type to the variable 'b' (line 136)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'b', boundary_444)
                    
                    # Assigning a Num to a Name (line 136):
                    int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 54), 'int')
                    # Assigning a type to the variable 'model_needs_updating' (line 136)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'model_needs_updating', int_445)
                    
                    # Getting the type of 'N' (line 136)
                    N_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'N')
                    int_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 62), 'int')
                    # Applying the binary operator '-=' (line 136)
                    result_isub_448 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 59), '-=', N_446, int_447)
                    # Assigning a type to the variable 'N' (line 136)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'N', result_isub_448)
                    
                    # SSA branch for the else part of an if statement (line 133)
                    module_type_store.open_ssa_branch('else')
                    pass
                    # SSA join for if statement (line 133)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 129)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'a' (line 144)
            a_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'a')
            # Getting the type of 'HALF' (line 144)
            HALF_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'HALF')
            # Applying the binary operator '>=' (line 144)
            result_ge_451 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 21), '>=', a_449, HALF_450)
            
            
            # Getting the type of 'b' (line 144)
            b_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'b')
            # Getting the type of 'HALF' (line 144)
            HALF_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'HALF')
            # Applying the binary operator '<=' (line 144)
            result_le_454 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 34), '<=', b_452, HALF_453)
            
            # Applying the binary operator 'or' (line 144)
            result_or_keyword_455 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), 'or', result_ge_451, result_le_454)
            
            # Assigning a type to the variable 'result_or_keyword_455' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'result_or_keyword_455', result_or_keyword_455)
            # Testing if the while is going to be iterated (line 144)
            # Testing the type of an if condition (line 144)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), result_or_keyword_455)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 144, 12), result_or_keyword_455):
                # SSA begins for while statement (line 144)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Getting the type of 'a' (line 145)
                a_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'a')
                # Getting the type of 'HALF' (line 145)
                HALF_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 23), 'HALF')
                # Applying the binary operator '>=' (line 145)
                result_ge_458 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 20), '>=', a_456, HALF_457)
                
                # Testing if the type of an if condition is none (line 145)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 145, 16), result_ge_458):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 145)
                    if_condition_459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 16), result_ge_458)
                    # Assigning a type to the variable 'if_condition_459' (line 145)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'if_condition_459', if_condition_459)
                    # SSA begins for if statement (line 145)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 146):
                    # Getting the type of 'a' (line 146)
                    a_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'a')
                    # Getting the type of 'HALF' (line 146)
                    HALF_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'HALF')
                    # Applying the binary operator '-' (line 146)
                    result_sub_462 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 24), '-', a_460, HALF_461)
                    
                    # Assigning a type to the variable 'a' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'a', result_sub_462)
                    
                    # Assigning a BinOp to a Name (line 146):
                    # Getting the type of 'b' (line 146)
                    b_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'b')
                    # Getting the type of 'HALF' (line 146)
                    HALF_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'HALF')
                    # Applying the binary operator '-' (line 146)
                    result_sub_465 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 38), '-', b_463, HALF_464)
                    
                    # Assigning a type to the variable 'b' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'b', result_sub_465)
                    
                    # Assigning a BinOp to a Name (line 146):
                    # Getting the type of 'u' (line 146)
                    u_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 54), 'u')
                    # Getting the type of 'HALF' (line 146)
                    HALF_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 56), 'HALF')
                    # Applying the binary operator '-' (line 146)
                    result_sub_468 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 54), '-', u_466, HALF_467)
                    
                    # Assigning a type to the variable 'u' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 50), 'u', result_sub_468)
                    
                    # Assigning a BinOp to a Name (line 146):
                    # Getting the type of 'v' (line 146)
                    v_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 71), 'v')
                    # Getting the type of 'HALF' (line 146)
                    HALF_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 73), 'HALF')
                    # Applying the binary operator '-' (line 146)
                    result_sub_471 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 71), '-', v_469, HALF_470)
                    
                    # Assigning a type to the variable 'v' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 67), 'v', result_sub_471)
                    pass
                    # SSA branch for the else part of an if statement (line 145)
                    module_type_store.open_ssa_branch('else')
                    pass
                    # SSA join for if statement (line 145)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'a' (line 150)
                a_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'a')
                int_473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 21), 'int')
                # Applying the binary operator '*=' (line 150)
                result_imul_474 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 16), '*=', a_472, int_473)
                # Assigning a type to the variable 'a' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'a', result_imul_474)
                
                
                # Getting the type of 'b' (line 150)
                b_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'b')
                int_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 35), 'int')
                # Applying the binary operator '*=' (line 150)
                result_imul_477 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 30), '*=', b_475, int_476)
                # Assigning a type to the variable 'b' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'b', result_imul_477)
                
                
                # Getting the type of 'u' (line 150)
                u_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'u')
                int_479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 49), 'int')
                # Applying the binary operator '*=' (line 150)
                result_imul_480 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 44), '*=', u_478, int_479)
                # Assigning a type to the variable 'u' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'u', result_imul_480)
                
                
                # Getting the type of 'v' (line 150)
                v_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 58), 'v')
                int_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 63), 'int')
                # Applying the binary operator '*=' (line 150)
                result_imul_483 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 58), '*=', v_481, int_482)
                # Assigning a type to the variable 'v' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 58), 'v', result_imul_483)
                
                
                # Assigning a Num to a Name (line 151):
                int_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 39), 'int')
                # Assigning a type to the variable 'model_needs_updating' (line 151)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'model_needs_updating', int_484)
                pass
                # SSA join for while statement (line 144)
                module_type_store = module_type_store.join_ssa_context()

            
            # Evaluating assert statement condition
            
            # Getting the type of 'a' (line 154)
            a_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'a')
            # Getting the type of 'HALF' (line 154)
            HALF_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'HALF')
            # Applying the binary operator '<=' (line 154)
            result_le_487 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '<=', a_485, HALF_486)
            
            assert_488 = result_le_487
            # Assigning a type to the variable 'assert_488' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'assert_488', result_le_487)
            # Evaluating assert statement condition
            
            # Getting the type of 'b' (line 154)
            b_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 46), 'b')
            # Getting the type of 'HALF' (line 154)
            HALF_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'HALF')
            # Applying the binary operator '>=' (line 154)
            result_ge_491 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 46), '>=', b_489, HALF_490)
            
            assert_492 = result_ge_491
            # Assigning a type to the variable 'assert_492' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'assert_492', result_ge_491)
            # Evaluating assert statement condition
            
            # Getting the type of 'a' (line 154)
            a_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 73), 'a')
            int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 76), 'int')
            # Applying the binary operator '>=' (line 154)
            result_ge_495 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 73), '>=', a_493, int_494)
            
            assert_496 = result_ge_495
            # Assigning a type to the variable 'assert_496' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 66), 'assert_496', result_ge_495)
            # Evaluating assert statement condition
            
            # Getting the type of 'b' (line 154)
            b_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 97), 'b')
            # Getting the type of 'ONE' (line 154)
            ONE_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 100), 'ONE')
            # Applying the binary operator '<=' (line 154)
            result_le_499 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 97), '<=', b_497, ONE_498)
            
            assert_500 = result_le_499
            # Assigning a type to the variable 'assert_500' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 90), 'assert_500', result_le_499)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'a' (line 156)
            a_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'a')
            # Getting the type of 'QUARTER' (line 156)
            QUARTER_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'QUARTER')
            # Applying the binary operator '>' (line 156)
            result_gt_503 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 21), '>', a_501, QUARTER_502)
            
            
            # Getting the type of 'b' (line 156)
            b_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'b')
            # Getting the type of 'THREEQU' (line 156)
            THREEQU_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'THREEQU')
            # Applying the binary operator '<' (line 156)
            result_lt_506 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 37), '<', b_504, THREEQU_505)
            
            # Applying the binary operator 'and' (line 156)
            result_and_keyword_507 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 20), 'and', result_gt_503, result_lt_506)
            
            # Assigning a type to the variable 'result_and_keyword_507' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'result_and_keyword_507', result_and_keyword_507)
            # Testing if the while is going to be iterated (line 156)
            # Testing the type of an if condition (line 156)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 12), result_and_keyword_507)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 156, 12), result_and_keyword_507):
                # SSA begins for while statement (line 156)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a BinOp to a Name (line 157):
                int_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 20), 'int')
                # Getting the type of 'a' (line 157)
                a_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'a')
                # Applying the binary operator '*' (line 157)
                result_mul_510 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 20), '*', int_508, a_509)
                
                # Getting the type of 'HALF' (line 157)
                HALF_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'HALF')
                # Applying the binary operator '-' (line 157)
                result_sub_512 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 20), '-', result_mul_510, HALF_511)
                
                # Assigning a type to the variable 'a' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'a', result_sub_512)
                
                # Assigning a BinOp to a Name (line 157):
                int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 35), 'int')
                # Getting the type of 'b' (line 157)
                b_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'b')
                # Applying the binary operator '*' (line 157)
                result_mul_515 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 35), '*', int_513, b_514)
                
                # Getting the type of 'HALF' (line 157)
                HALF_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'HALF')
                # Applying the binary operator '-' (line 157)
                result_sub_517 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 35), '-', result_mul_515, HALF_516)
                
                # Assigning a type to the variable 'b' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'b', result_sub_517)
                
                # Assigning a BinOp to a Name (line 157):
                int_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 50), 'int')
                # Getting the type of 'u' (line 157)
                u_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 52), 'u')
                # Applying the binary operator '*' (line 157)
                result_mul_520 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 50), '*', int_518, u_519)
                
                # Getting the type of 'HALF' (line 157)
                HALF_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 54), 'HALF')
                # Applying the binary operator '-' (line 157)
                result_sub_522 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 50), '-', result_mul_520, HALF_521)
                
                # Assigning a type to the variable 'u' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 46), 'u', result_sub_522)
                
                # Assigning a BinOp to a Name (line 157):
                int_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 66), 'int')
                # Getting the type of 'v' (line 157)
                v_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 68), 'v')
                # Applying the binary operator '*' (line 157)
                result_mul_525 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 66), '*', int_523, v_524)
                
                # Getting the type of 'HALF' (line 157)
                HALF_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 70), 'HALF')
                # Applying the binary operator '-' (line 157)
                result_sub_527 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 66), '-', result_mul_525, HALF_526)
                
                # Assigning a type to the variable 'v' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 62), 'v', result_sub_527)
                pass
                # SSA join for while statement (line 156)
                module_type_store = module_type_store.join_ssa_context()

            
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'N' (line 159)
            N_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'N')
            int_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
            # Applying the binary operator '>' (line 159)
            result_gt_530 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), '>', N_528, int_529)
            
            # Getting the type of 'model_needs_updating' (line 159)
            model_needs_updating_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'model_needs_updating')
            # Applying the binary operator 'and' (line 159)
            result_and_keyword_532 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), 'and', result_gt_530, model_needs_updating_531)
            
            # Applying the 'not' unary operator (line 159)
            result_not__533 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 15), 'not', result_and_keyword_532)
            
            # Testing if the type of an if condition is none (line 159)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 12), result_not__533):
                pass
            else:
                
                # Testing the type of an if condition (line 159)
                if_condition_534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 12), result_not__533)
                # Assigning a type to the variable 'if_condition_534' (line 159)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'if_condition_534', if_condition_534)
                # SSA begins for if statement (line 159)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 159)
                module_type_store = module_type_store.join_ssa_context()
                

            pass
            # SSA join for while statement (line 117)
            module_type_store = module_type_store.join_ssa_context()

        
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'ans' (line 163)
    ans_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'ans')
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type', ans_535)
    pass
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_536

# Assigning a type to the variable 'decode' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'decode', decode)

@norecursion
def hardertest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hardertest'
    module_type_store = module_type_store.open_function_context('hardertest', 166, 0, False)
    
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

    
    # Assigning a Call to a Name (line 168):
    
    # Call to open(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Call to Relative(...): (line 168)
    # Processing the call arguments (line 168)
    str_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 31), 'str', 'testdata/BentCoinFile')
    # Processing the call keyword arguments (line 168)
    kwargs_540 = {}
    # Getting the type of 'Relative' (line 168)
    Relative_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 168)
    Relative_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 168, 22), Relative_538, *[str_539], **kwargs_540)
    
    str_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 58), 'str', 'r')
    # Processing the call keyword arguments (line 168)
    kwargs_543 = {}
    # Getting the type of 'open' (line 168)
    open_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'open', False)
    # Calling open(args, kwargs) (line 168)
    open_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), open_537, *[Relative_call_result_541, str_542], **kwargs_543)
    
    # Assigning a type to the variable 'inputfile' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'inputfile', open_call_result_544)
    
    # Assigning a Call to a Name (line 169):
    
    # Call to open(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Call to Relative(...): (line 169)
    # Processing the call arguments (line 169)
    str_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 169)
    kwargs_548 = {}
    # Getting the type of 'Relative' (line 169)
    Relative_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'Relative', False)
    # Calling Relative(args, kwargs) (line 169)
    Relative_call_result_549 = invoke(stypy.reporting.localization.Localization(__file__, 169, 23), Relative_546, *[str_547], **kwargs_548)
    
    str_550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 45), 'str', 'w')
    # Processing the call keyword arguments (line 169)
    kwargs_551 = {}
    # Getting the type of 'open' (line 169)
    open_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'open', False)
    # Calling open(args, kwargs) (line 169)
    open_call_result_552 = invoke(stypy.reporting.localization.Localization(__file__, 169, 17), open_545, *[Relative_call_result_549, str_550], **kwargs_551)
    
    # Assigning a type to the variable 'outputfile' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'outputfile', open_call_result_552)
    
    # Assigning a Call to a Name (line 172):
    
    # Call to read(...): (line 172)
    # Processing the call keyword arguments (line 172)
    kwargs_555 = {}
    # Getting the type of 'inputfile' (line 172)
    inputfile_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 172)
    read_554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), inputfile_553, 'read')
    # Calling read(args, kwargs) (line 172)
    read_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), read_554, *[], **kwargs_555)
    
    # Assigning a type to the variable 's' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 's', read_call_result_556)
    
    # Assigning a Call to a Name (line 173):
    
    # Call to len(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 's' (line 173)
    s_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 's', False)
    # Processing the call keyword arguments (line 173)
    kwargs_559 = {}
    # Getting the type of 'len' (line 173)
    len_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'len', False)
    # Calling len(args, kwargs) (line 173)
    len_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), len_557, *[s_558], **kwargs_559)
    
    # Assigning a type to the variable 'N' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'N', len_call_result_560)
    
    # Assigning a Call to a Name (line 174):
    
    # Call to encode(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 's' (line 174)
    s_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 's', False)
    int_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'int')
    int_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'int')
    # Processing the call keyword arguments (line 174)
    kwargs_565 = {}
    # Getting the type of 'encode' (line 174)
    encode_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 10), 'encode', False)
    # Calling encode(args, kwargs) (line 174)
    encode_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 174, 10), encode_561, *[s_562, int_563, int_564], **kwargs_565)
    
    # Assigning a type to the variable 'zip' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'zip', encode_call_result_566)
    
    # Call to write(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'zip' (line 175)
    zip_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'zip', False)
    # Processing the call keyword arguments (line 175)
    kwargs_570 = {}
    # Getting the type of 'outputfile' (line 175)
    outputfile_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 175)
    write_568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), outputfile_567, 'write')
    # Calling write(args, kwargs) (line 175)
    write_call_result_571 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), write_568, *[zip_569], **kwargs_570)
    
    
    # Call to close(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_574 = {}
    # Getting the type of 'outputfile' (line 176)
    outputfile_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 176)
    close_573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), outputfile_572, 'close')
    # Calling close(args, kwargs) (line 176)
    close_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), close_573, *[], **kwargs_574)
    
    
    # Call to close(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_578 = {}
    # Getting the type of 'inputfile' (line 176)
    inputfile_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 176)
    close_577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 28), inputfile_576, 'close')
    # Calling close(args, kwargs) (line 176)
    close_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 176, 28), close_577, *[], **kwargs_578)
    
    
    # Assigning a Call to a Name (line 179):
    
    # Call to open(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Call to Relative(...): (line 179)
    # Processing the call arguments (line 179)
    str_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 179)
    kwargs_583 = {}
    # Getting the type of 'Relative' (line 179)
    Relative_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 179)
    Relative_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 179, 22), Relative_581, *[str_582], **kwargs_583)
    
    str_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 44), 'str', 'r')
    # Processing the call keyword arguments (line 179)
    kwargs_586 = {}
    # Getting the type of 'open' (line 179)
    open_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'open', False)
    # Calling open(args, kwargs) (line 179)
    open_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), open_580, *[Relative_call_result_584, str_585], **kwargs_586)
    
    # Assigning a type to the variable 'inputfile' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'inputfile', open_call_result_587)
    
    # Assigning a Call to a Name (line 180):
    
    # Call to open(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Call to Relative(...): (line 180)
    # Processing the call arguments (line 180)
    str_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 32), 'str', 'tmp2')
    # Processing the call keyword arguments (line 180)
    kwargs_591 = {}
    # Getting the type of 'Relative' (line 180)
    Relative_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'Relative', False)
    # Calling Relative(args, kwargs) (line 180)
    Relative_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 180, 23), Relative_589, *[str_590], **kwargs_591)
    
    str_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 42), 'str', 'w')
    # Processing the call keyword arguments (line 180)
    kwargs_594 = {}
    # Getting the type of 'open' (line 180)
    open_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'open', False)
    # Calling open(args, kwargs) (line 180)
    open_call_result_595 = invoke(stypy.reporting.localization.Localization(__file__, 180, 17), open_588, *[Relative_call_result_592, str_593], **kwargs_594)
    
    # Assigning a type to the variable 'outputfile' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'outputfile', open_call_result_595)
    
    # Assigning a Call to a Name (line 182):
    
    # Call to decode(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Call to list(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Call to read(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_600 = {}
    # Getting the type of 'inputfile' (line 182)
    inputfile_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 182)
    read_599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 22), inputfile_598, 'read')
    # Calling read(args, kwargs) (line 182)
    read_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 182, 22), read_599, *[], **kwargs_600)
    
    # Processing the call keyword arguments (line 182)
    kwargs_602 = {}
    # Getting the type of 'list' (line 182)
    list_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'list', False)
    # Calling list(args, kwargs) (line 182)
    list_call_result_603 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), list_597, *[read_call_result_601], **kwargs_602)
    
    # Getting the type of 'N' (line 182)
    N_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 41), 'N', False)
    int_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 44), 'int')
    int_606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 48), 'int')
    # Processing the call keyword arguments (line 182)
    kwargs_607 = {}
    # Getting the type of 'decode' (line 182)
    decode_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 10), 'decode', False)
    # Calling decode(args, kwargs) (line 182)
    decode_call_result_608 = invoke(stypy.reporting.localization.Localization(__file__, 182, 10), decode_596, *[list_call_result_603, N_604, int_605, int_606], **kwargs_607)
    
    # Assigning a type to the variable 'unc' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'unc', decode_call_result_608)
    
    # Call to write(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'unc' (line 183)
    unc_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'unc', False)
    # Processing the call keyword arguments (line 183)
    kwargs_612 = {}
    # Getting the type of 'outputfile' (line 183)
    outputfile_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 183)
    write_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), outputfile_609, 'write')
    # Calling write(args, kwargs) (line 183)
    write_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), write_610, *[unc_611], **kwargs_612)
    
    
    # Call to close(...): (line 184)
    # Processing the call keyword arguments (line 184)
    kwargs_616 = {}
    # Getting the type of 'outputfile' (line 184)
    outputfile_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 184)
    close_615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), outputfile_614, 'close')
    # Calling close(args, kwargs) (line 184)
    close_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), close_615, *[], **kwargs_616)
    
    
    # Call to close(...): (line 184)
    # Processing the call keyword arguments (line 184)
    kwargs_620 = {}
    # Getting the type of 'inputfile' (line 184)
    inputfile_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 184)
    close_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), inputfile_618, 'close')
    # Calling close(args, kwargs) (line 184)
    close_call_result_621 = invoke(stypy.reporting.localization.Localization(__file__, 184, 28), close_619, *[], **kwargs_620)
    
    
    # ################# End of 'hardertest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hardertest' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hardertest'
    return stypy_return_type_622

# Assigning a type to the variable 'hardertest' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'hardertest', hardertest)

@norecursion
def test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test'
    module_type_store = module_type_store.open_function_context('test', 191, 0, False)
    
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

    
    # Assigning a List to a Name (line 192):
    
    # Obtaining an instance of the builtin type 'list' (line 192)
    list_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 7), 'list')
    # Adding type elements to the builtin type 'list' instance (line 192)
    # Adding element type (line 192)
    str_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 8), 'str', '1010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_624)
    # Adding element type (line 192)
    str_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 16), 'str', '111')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_625)
    # Adding element type (line 192)
    str_626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'str', '00001000000000000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_626)
    # Adding element type (line 192)
    str_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_627)
    # Adding element type (line 192)
    str_628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 13), 'str', '10')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_628)
    # Adding element type (line 192)
    str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 20), 'str', '01')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_629)
    # Adding element type (line 192)
    str_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'str', '0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_630)
    # Adding element type (line 192)
    str_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', '0000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_631)
    # Adding element type (line 192)
    str_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 8), 'str', '000000000000000100000000000000000000000000000000100000000000000000011000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_623, str_632)
    
    # Assigning a type to the variable 'sl' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'sl', list_623)
    
    # Getting the type of 'sl' (line 195)
    sl_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 13), 'sl')
    # Assigning a type to the variable 'sl_633' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'sl_633', sl_633)
    # Testing if the for loop is going to be iterated (line 195)
    # Testing the type of a for loop iterable (line 195)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 4), sl_633)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 195, 4), sl_633):
        # Getting the type of the for loop variable (line 195)
        for_loop_var_634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 4), sl_633)
        # Assigning a type to the variable 's' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 's', for_loop_var_634)
        # SSA begins for a for statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 197):
        
        # Call to len(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 's' (line 197)
        s_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 14), 's', False)
        # Processing the call keyword arguments (line 197)
        kwargs_637 = {}
        # Getting the type of 'len' (line 197)
        len_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 10), 'len', False)
        # Calling len(args, kwargs) (line 197)
        len_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 197, 10), len_635, *[s_636], **kwargs_637)
        
        # Assigning a type to the variable 'N' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'N', len_call_result_638)
        
        # Assigning a Call to a Name (line 198):
        
        # Call to encode(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 's' (line 198)
        s_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 's', False)
        int_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 21), 'int')
        int_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 24), 'int')
        # Processing the call keyword arguments (line 198)
        kwargs_643 = {}
        # Getting the type of 'encode' (line 198)
        encode_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'encode', False)
        # Calling encode(args, kwargs) (line 198)
        encode_call_result_644 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), encode_639, *[s_640, int_641, int_642], **kwargs_643)
        
        # Assigning a type to the variable 'e' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'e', encode_call_result_644)
        
        # Assigning a Call to a Name (line 200):
        
        # Call to decode(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'e' (line 200)
        e_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'e', False)
        # Getting the type of 'N' (line 200)
        N_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'N', False)
        int_648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 24), 'int')
        int_649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'int')
        # Processing the call keyword arguments (line 200)
        kwargs_650 = {}
        # Getting the type of 'decode' (line 200)
        decode_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'decode', False)
        # Calling decode(args, kwargs) (line 200)
        decode_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), decode_645, *[e_646, N_647, int_648, int_649], **kwargs_650)
        
        # Assigning a type to the variable 'ds' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'ds', decode_call_result_651)
        
        # Getting the type of 'ds' (line 202)
        ds_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'ds')
        # Getting the type of 's' (line 202)
        s_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 's')
        # Applying the binary operator '!=' (line 202)
        result_ne_654 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 13), '!=', ds_652, s_653)
        
        # Testing if the type of an if condition is none (line 202)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 202, 8), result_ne_654):
            pass
        else:
            
            # Testing the type of an if condition (line 202)
            if_condition_655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), result_ne_654)
            # Assigning a type to the variable 'if_condition_655' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_655', if_condition_655)
            # SSA begins for if statement (line 202)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 202)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 202)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    pass
    
    # ################# End of 'test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test' in the type store
    # Getting the type of 'stypy_return_type' (line 191)
    stypy_return_type_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_656)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test'
    return stypy_return_type_656

# Assigning a type to the variable 'test' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'test', test)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 211, 0, False)
    
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

    
    # Call to test(...): (line 212)
    # Processing the call keyword arguments (line 212)
    kwargs_658 = {}
    # Getting the type of 'test' (line 212)
    test_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'test', False)
    # Calling test(args, kwargs) (line 212)
    test_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), test_657, *[], **kwargs_658)
    
    
    # Call to hardertest(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_661 = {}
    # Getting the type of 'hardertest' (line 213)
    hardertest_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'hardertest', False)
    # Calling hardertest(args, kwargs) (line 213)
    hardertest_call_result_662 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), hardertest_660, *[], **kwargs_661)
    
    # Getting the type of 'True' (line 214)
    True_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', True_663)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 211)
    stypy_return_type_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_664)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_664

# Assigning a type to the variable 'run' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'run', run)

# Call to run(...): (line 216)
# Processing the call keyword arguments (line 216)
kwargs_666 = {}
# Getting the type of 'run' (line 216)
run_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'run', False)
# Calling run(args, kwargs) (line 216)
run_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 216, 0), run_665, *[], **kwargs_666)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

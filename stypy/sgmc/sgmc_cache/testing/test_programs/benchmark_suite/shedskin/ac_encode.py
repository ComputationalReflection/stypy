
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
    
    # Evaluating assert statement condition
    
    # Getting the type of 'c1' (line 22)
    c1_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 56), 'c1')
    int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 59), 'int')
    # Applying the binary operator '>' (line 22)
    result_gt_67 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 56), '>', c1_65, int_66)
    
    
    
    # Getting the type of 'adaptive' (line 23)
    adaptive_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'adaptive')
    int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'int')
    # Applying the binary operator '==' (line 23)
    result_eq_70 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 7), '==', adaptive_68, int_69)
    
    # Testing the type of an if condition (line 23)
    if_condition_71 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), result_eq_70)
    # Assigning a type to the variable 'if_condition_71' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_condition_71', if_condition_71)
    # SSA begins for if statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 24):
    # Getting the type of 'c0' (line 24)
    c0_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'c0')
    float_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'float')
    # Applying the binary operator '*' (line 24)
    result_mul_74 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 13), '*', c0_72, float_73)
    
    # Getting the type of 'c0' (line 24)
    c0_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'c0')
    # Getting the type of 'c1' (line 24)
    c1_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'c1')
    # Applying the binary operator '+' (line 24)
    result_add_77 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 21), '+', c0_75, c1_76)
    
    # Applying the binary operator 'div' (line 24)
    result_div_78 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), 'div', result_mul_74, result_add_77)
    
    # Assigning a type to the variable 'p0' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'p0', result_div_78)
    pass
    # SSA join for if statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 26):
    str_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'str', '')
    # Assigning a type to the variable 'ans' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'ans', str_79)
    
    # Assigning a List to a Name (line 27):
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), list_80, int_81)
    
    # Assigning a type to the variable 'charstack' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'charstack', list_80)
    
    # Getting the type of 'string' (line 28)
    string_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'string')
    # Testing the type of a for loop iterable (line 28)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 4), string_82)
    # Getting the type of the for loop variable (line 28)
    for_loop_var_83 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 4), string_82)
    # Assigning a type to the variable 'c' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'c', for_loop_var_83)
    # SSA begins for a for statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 29):
    # Getting the type of 'b' (line 29)
    b_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'b')
    # Getting the type of 'a' (line 29)
    a_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'a')
    # Applying the binary operator '-' (line 29)
    result_sub_86 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 10), '-', b_84, a_85)
    
    # Assigning a type to the variable 'w' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'w', result_sub_86)
    
    # Getting the type of 'adaptive' (line 30)
    adaptive_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'adaptive')
    # Testing the type of an if condition (line 30)
    if_condition_88 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), adaptive_87)
    # Assigning a type to the variable 'if_condition_88' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_88', if_condition_88)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 31):
    # Getting the type of 'c0' (line 31)
    c0_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'c0')
    # Getting the type of 'c1' (line 31)
    c1_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'c1')
    # Applying the binary operator '+' (line 31)
    result_add_91 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 17), '+', c0_89, c1_90)
    
    # Assigning a type to the variable 'cT' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'cT', result_add_91)
    
    # Assigning a BinOp to a Name (line 32):
    # Getting the type of 'c0' (line 32)
    c0_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'c0')
    float_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'float')
    # Applying the binary operator '*' (line 32)
    result_mul_94 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 17), '*', c0_92, float_93)
    
    # Getting the type of 'cT' (line 32)
    cT_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'cT')
    # Applying the binary operator 'div' (line 32)
    result_div_96 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 23), 'div', result_mul_94, cT_95)
    
    # Assigning a type to the variable 'p0' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'p0', result_div_96)
    pass
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 34):
    # Getting the type of 'a' (line 34)
    a_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'a')
    
    # Call to int(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'p0' (line 34)
    p0_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'p0', False)
    # Getting the type of 'w' (line 34)
    w_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'w', False)
    # Applying the binary operator '*' (line 34)
    result_mul_101 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 27), '*', p0_99, w_100)
    
    # Processing the call keyword arguments (line 34)
    kwargs_102 = {}
    # Getting the type of 'int' (line 34)
    int_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'int', False)
    # Calling int(args, kwargs) (line 34)
    int_call_result_103 = invoke(stypy.reporting.localization.Localization(__file__, 34, 23), int_98, *[result_mul_101], **kwargs_102)
    
    # Applying the binary operator '+' (line 34)
    result_add_104 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 19), '+', a_97, int_call_result_103)
    
    # Assigning a type to the variable 'boundary' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'boundary', result_add_104)
    
    
    # Getting the type of 'boundary' (line 35)
    boundary_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'boundary')
    # Getting the type of 'a' (line 35)
    a_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'a')
    # Applying the binary operator '==' (line 35)
    result_eq_107 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), '==', boundary_105, a_106)
    
    # Testing the type of an if condition (line 35)
    if_condition_108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_eq_107)
    # Assigning a type to the variable 'if_condition_108' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_108', if_condition_108)
    # SSA begins for if statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'boundary' (line 35)
    boundary_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'boundary')
    int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'int')
    # Applying the binary operator '+=' (line 35)
    result_iadd_111 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 28), '+=', boundary_109, int_110)
    # Assigning a type to the variable 'boundary' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'boundary', result_iadd_111)
    
    str_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 49), 'str', 'warningA')
    pass
    # SSA join for if statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'boundary' (line 36)
    boundary_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'boundary')
    # Getting the type of 'b' (line 36)
    b_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'b')
    # Applying the binary operator '==' (line 36)
    result_eq_115 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '==', boundary_113, b_114)
    
    # Testing the type of an if condition (line 36)
    if_condition_116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), result_eq_115)
    # Assigning a type to the variable 'if_condition_116' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_116', if_condition_116)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'boundary' (line 36)
    boundary_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'boundary')
    int_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 40), 'int')
    # Applying the binary operator '-=' (line 36)
    result_isub_119 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 28), '-=', boundary_117, int_118)
    # Assigning a type to the variable 'boundary' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'boundary', result_isub_119)
    
    str_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 49), 'str', 'warningB')
    pass
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'c' (line 38)
    c_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'c')
    str_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'str', '1')
    # Applying the binary operator '==' (line 38)
    result_eq_123 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 12), '==', c_121, str_122)
    
    # Testing the type of an if condition (line 38)
    if_condition_124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), result_eq_123)
    # Assigning a type to the variable 'if_condition_124' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_124', if_condition_124)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 39):
    # Getting the type of 'boundary' (line 39)
    boundary_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'boundary')
    # Assigning a type to the variable 'a' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'a', boundary_125)
    
    # Getting the type of 'tot1' (line 40)
    tot1_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tot1')
    int_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'int')
    # Applying the binary operator '+=' (line 40)
    result_iadd_128 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 12), '+=', tot1_126, int_127)
    # Assigning a type to the variable 'tot1' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'tot1', result_iadd_128)
    
    
    # Getting the type of 'adaptive' (line 41)
    adaptive_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'adaptive')
    # Testing the type of an if condition (line 41)
    if_condition_130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), adaptive_129)
    # Assigning a type to the variable 'if_condition_130' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_130', if_condition_130)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 41)
    c1_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'c1')
    float_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'float')
    # Applying the binary operator '+=' (line 41)
    result_iadd_133 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '+=', c1_131, float_132)
    # Assigning a type to the variable 'c1' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'c1', result_iadd_133)
    
    pass
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'c' (line 42)
    c_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'c')
    str_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'str', '0')
    # Applying the binary operator '==' (line 42)
    result_eq_136 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 14), '==', c_134, str_135)
    
    # Testing the type of an if condition (line 42)
    if_condition_137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 13), result_eq_136)
    # Assigning a type to the variable 'if_condition_137' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'if_condition_137', if_condition_137)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 43):
    # Getting the type of 'boundary' (line 43)
    boundary_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'boundary')
    # Assigning a type to the variable 'b' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'b', boundary_138)
    
    # Getting the type of 'tot0' (line 44)
    tot0_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'tot0')
    int_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
    # Applying the binary operator '+=' (line 44)
    result_iadd_141 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '+=', tot0_139, int_140)
    # Assigning a type to the variable 'tot0' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'tot0', result_iadd_141)
    
    
    # Getting the type of 'adaptive' (line 45)
    adaptive_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'adaptive')
    # Testing the type of an if condition (line 45)
    if_condition_143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 12), adaptive_142)
    # Assigning a type to the variable 'if_condition_143' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'if_condition_143', if_condition_143)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c0' (line 45)
    c0_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'c0')
    float_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'float')
    # Applying the binary operator '+=' (line 45)
    result_iadd_146 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 25), '+=', c0_144, float_145)
    # Assigning a type to the variable 'c0' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'c0', result_iadd_146)
    
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
    a_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'a')
    # Getting the type of 'HALF' (line 48)
    HALF_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'HALF')
    # Applying the binary operator '>=' (line 48)
    result_ge_149 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 17), '>=', a_147, HALF_148)
    
    
    # Getting the type of 'b' (line 48)
    b_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'b')
    # Getting the type of 'HALF' (line 48)
    HALF_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'HALF')
    # Applying the binary operator '<=' (line 48)
    result_le_152 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 30), '<=', b_150, HALF_151)
    
    # Applying the binary operator 'or' (line 48)
    result_or_keyword_153 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 16), 'or', result_ge_149, result_le_152)
    
    # Testing the type of an if condition (line 48)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_or_keyword_153)
    # SSA begins for while statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'a' (line 49)
    a_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'a')
    # Getting the type of 'HALF' (line 49)
    HALF_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'HALF')
    # Applying the binary operator '>=' (line 49)
    result_ge_156 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '>=', a_154, HALF_155)
    
    # Testing the type of an if condition (line 49)
    if_condition_157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 12), result_ge_156)
    # Assigning a type to the variable 'if_condition_157' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'if_condition_157', if_condition_157)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 50):
    # Getting the type of 'ans' (line 50)
    ans_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'ans')
    
    # Call to clear(...): (line 50)
    # Processing the call arguments (line 50)
    int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'int')
    # Getting the type of 'charstack' (line 50)
    charstack_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'charstack', False)
    # Processing the call keyword arguments (line 50)
    kwargs_162 = {}
    # Getting the type of 'clear' (line 50)
    clear_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'clear', False)
    # Calling clear(args, kwargs) (line 50)
    clear_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 50, 28), clear_159, *[int_160, charstack_161], **kwargs_162)
    
    # Applying the binary operator '+' (line 50)
    result_add_164 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), '+', ans_158, clear_call_result_163)
    
    # Assigning a type to the variable 'ans' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'ans', result_add_164)
    
    # Assigning a BinOp to a Name (line 51):
    # Getting the type of 'a' (line 51)
    a_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'a')
    # Getting the type of 'HALF' (line 51)
    HALF_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'HALF')
    # Applying the binary operator '-' (line 51)
    result_sub_167 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 20), '-', a_165, HALF_166)
    
    # Assigning a type to the variable 'a' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'a', result_sub_167)
    
    # Assigning a BinOp to a Name (line 52):
    # Getting the type of 'b' (line 52)
    b_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'b')
    # Getting the type of 'HALF' (line 52)
    HALF_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'HALF')
    # Applying the binary operator '-' (line 52)
    result_sub_170 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), '-', b_168, HALF_169)
    
    # Assigning a type to the variable 'b' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'b', result_sub_170)
    # SSA branch for the else part of an if statement (line 49)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 54):
    # Getting the type of 'ans' (line 54)
    ans_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'ans')
    
    # Call to clear(...): (line 54)
    # Processing the call arguments (line 54)
    int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'int')
    # Getting the type of 'charstack' (line 54)
    charstack_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'charstack', False)
    # Processing the call keyword arguments (line 54)
    kwargs_175 = {}
    # Getting the type of 'clear' (line 54)
    clear_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'clear', False)
    # Calling clear(args, kwargs) (line 54)
    clear_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), clear_172, *[int_173, charstack_174], **kwargs_175)
    
    # Applying the binary operator '+' (line 54)
    result_add_177 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 22), '+', ans_171, clear_call_result_176)
    
    # Assigning a type to the variable 'ans' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'ans', result_add_177)
    pass
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'a' (line 56)
    a_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'a')
    int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'int')
    # Applying the binary operator '*=' (line 56)
    result_imul_180 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '*=', a_178, int_179)
    # Assigning a type to the variable 'a' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'a', result_imul_180)
    
    
    # Getting the type of 'b' (line 56)
    b_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'b')
    int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'int')
    # Applying the binary operator '*=' (line 56)
    result_imul_183 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 26), '*=', b_181, int_182)
    # Assigning a type to the variable 'b' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'b', result_imul_183)
    
    pass
    # SSA join for while statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    # Evaluating assert statement condition
    
    # Getting the type of 'a' (line 59)
    a_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'a')
    # Getting the type of 'HALF' (line 59)
    HALF_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'HALF')
    # Applying the binary operator '<=' (line 59)
    result_le_186 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), '<=', a_184, HALF_185)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'b' (line 59)
    b_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'b')
    # Getting the type of 'HALF' (line 59)
    HALF_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'HALF')
    # Applying the binary operator '>=' (line 59)
    result_ge_189 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 31), '>=', b_187, HALF_188)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'a' (line 59)
    a_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 47), 'a')
    int_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 50), 'int')
    # Applying the binary operator '>=' (line 59)
    result_ge_192 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 47), '>=', a_190, int_191)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'b' (line 59)
    b_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 60), 'b')
    # Getting the type of 'ONE' (line 59)
    ONE_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 63), 'ONE')
    # Applying the binary operator '<=' (line 59)
    result_le_195 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 60), '<=', b_193, ONE_194)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 61)
    a_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'a')
    # Getting the type of 'QUARTER' (line 61)
    QUARTER_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'QUARTER')
    # Applying the binary operator '>' (line 61)
    result_gt_198 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 17), '>', a_196, QUARTER_197)
    
    
    # Getting the type of 'b' (line 61)
    b_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'b')
    # Getting the type of 'THREEQU' (line 61)
    THREEQU_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'THREEQU')
    # Applying the binary operator '<' (line 61)
    result_lt_201 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 33), '<', b_199, THREEQU_200)
    
    # Applying the binary operator 'and' (line 61)
    result_and_keyword_202 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 16), 'and', result_gt_198, result_lt_201)
    
    # Testing the type of an if condition (line 61)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), result_and_keyword_202)
    # SSA begins for while statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'charstack' (line 62)
    charstack_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'charstack')
    
    # Obtaining the type of the subscript
    int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'int')
    # Getting the type of 'charstack' (line 62)
    charstack_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'charstack')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), charstack_205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_207 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), getitem___206, int_204)
    
    int_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'int')
    # Applying the binary operator '+=' (line 62)
    result_iadd_209 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '+=', subscript_call_result_207, int_208)
    # Getting the type of 'charstack' (line 62)
    charstack_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'charstack')
    int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'int')
    # Storing an element on a container (line 62)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 12), charstack_210, (int_211, result_iadd_209))
    
    
    # Assigning a BinOp to a Name (line 63):
    int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'int')
    # Getting the type of 'a' (line 63)
    a_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'a')
    # Applying the binary operator '*' (line 63)
    result_mul_214 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '*', int_212, a_213)
    
    # Getting the type of 'HALF' (line 63)
    HALF_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'HALF')
    # Applying the binary operator '-' (line 63)
    result_sub_216 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '-', result_mul_214, HALF_215)
    
    # Assigning a type to the variable 'a' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'a', result_sub_216)
    
    # Assigning a BinOp to a Name (line 64):
    int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'int')
    # Getting the type of 'b' (line 64)
    b_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'b')
    # Applying the binary operator '*' (line 64)
    result_mul_219 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '*', int_217, b_218)
    
    # Getting the type of 'HALF' (line 64)
    HALF_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'HALF')
    # Applying the binary operator '-' (line 64)
    result_sub_221 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '-', result_mul_219, HALF_220)
    
    # Assigning a type to the variable 'b' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'b', result_sub_221)
    pass
    # SSA join for while statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    # Evaluating assert statement condition
    
    # Getting the type of 'a' (line 67)
    a_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'a')
    # Getting the type of 'HALF' (line 67)
    HALF_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'HALF')
    # Applying the binary operator '<=' (line 67)
    result_le_224 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), '<=', a_222, HALF_223)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'b' (line 67)
    b_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'b')
    # Getting the type of 'HALF' (line 67)
    HALF_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'HALF')
    # Applying the binary operator '>=' (line 67)
    result_ge_227 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 31), '>=', b_225, HALF_226)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'a' (line 67)
    a_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 47), 'a')
    int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 50), 'int')
    # Applying the binary operator '>=' (line 67)
    result_ge_230 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 47), '>=', a_228, int_229)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'b' (line 67)
    b_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 60), 'b')
    # Getting the type of 'ONE' (line 67)
    ONE_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 63), 'ONE')
    # Applying the binary operator '<=' (line 67)
    result_le_233 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 60), '<=', b_231, ONE_232)
    
    pass
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'HALF' (line 71)
    HALF_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'HALF')
    # Getting the type of 'a' (line 71)
    a_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'a')
    # Applying the binary operator '-' (line 71)
    result_sub_236 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 10), '-', HALF_234, a_235)
    
    # Getting the type of 'b' (line 71)
    b_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'b')
    # Getting the type of 'HALF' (line 71)
    HALF_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'HALF')
    # Applying the binary operator '-' (line 71)
    result_sub_239 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 21), '-', b_237, HALF_238)
    
    # Applying the binary operator '>' (line 71)
    result_gt_240 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), '>', result_sub_236, result_sub_239)
    
    # Testing the type of an if condition (line 71)
    if_condition_241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_gt_240)
    # Assigning a type to the variable 'if_condition_241' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_241', if_condition_241)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 72):
    # Getting the type of 'HALF' (line 72)
    HALF_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'HALF')
    # Getting the type of 'a' (line 72)
    a_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'a')
    # Applying the binary operator '-' (line 72)
    result_sub_244 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '-', HALF_242, a_243)
    
    # Assigning a type to the variable 'w' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'w', result_sub_244)
    
    # Assigning a BinOp to a Name (line 73):
    # Getting the type of 'ans' (line 73)
    ans_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'ans')
    
    # Call to clear(...): (line 73)
    # Processing the call arguments (line 73)
    int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'int')
    # Getting the type of 'charstack' (line 73)
    charstack_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'charstack', False)
    # Processing the call keyword arguments (line 73)
    kwargs_249 = {}
    # Getting the type of 'clear' (line 73)
    clear_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'clear', False)
    # Calling clear(args, kwargs) (line 73)
    clear_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), clear_246, *[int_247, charstack_248], **kwargs_249)
    
    # Applying the binary operator '+' (line 73)
    result_add_251 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 14), '+', ans_245, clear_call_result_250)
    
    # Assigning a type to the variable 'ans' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'ans', result_add_251)
    
    
    # Getting the type of 'w' (line 74)
    w_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'w')
    # Getting the type of 'HALF' (line 74)
    HALF_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'HALF')
    # Applying the binary operator '<' (line 74)
    result_lt_254 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 16), '<', w_252, HALF_253)
    
    # Testing the type of an if condition (line 74)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), result_lt_254)
    # SSA begins for while statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 75):
    # Getting the type of 'ans' (line 75)
    ans_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'ans')
    
    # Call to clear(...): (line 75)
    # Processing the call arguments (line 75)
    int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 30), 'int')
    # Getting the type of 'charstack' (line 75)
    charstack_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'charstack', False)
    # Processing the call keyword arguments (line 75)
    kwargs_259 = {}
    # Getting the type of 'clear' (line 75)
    clear_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'clear', False)
    # Calling clear(args, kwargs) (line 75)
    clear_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 75, 24), clear_256, *[int_257, charstack_258], **kwargs_259)
    
    # Applying the binary operator '+' (line 75)
    result_add_261 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 18), '+', ans_255, clear_call_result_260)
    
    # Assigning a type to the variable 'ans' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'ans', result_add_261)
    
    # Getting the type of 'w' (line 76)
    w_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'w')
    int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 16), 'int')
    # Applying the binary operator '*=' (line 76)
    result_imul_264 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 12), '*=', w_262, int_263)
    # Assigning a type to the variable 'w' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'w', result_imul_264)
    
    pass
    # SSA join for while statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    pass
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 80):
    # Getting the type of 'b' (line 80)
    b_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'b')
    # Getting the type of 'HALF' (line 80)
    HALF_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'HALF')
    # Applying the binary operator '-' (line 80)
    result_sub_267 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '-', b_265, HALF_266)
    
    # Assigning a type to the variable 'w' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'w', result_sub_267)
    
    # Assigning a BinOp to a Name (line 81):
    # Getting the type of 'ans' (line 81)
    ans_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'ans')
    
    # Call to clear(...): (line 81)
    # Processing the call arguments (line 81)
    int_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'int')
    # Getting the type of 'charstack' (line 81)
    charstack_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'charstack', False)
    # Processing the call keyword arguments (line 81)
    kwargs_272 = {}
    # Getting the type of 'clear' (line 81)
    clear_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'clear', False)
    # Calling clear(args, kwargs) (line 81)
    clear_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 81, 20), clear_269, *[int_270, charstack_271], **kwargs_272)
    
    # Applying the binary operator '+' (line 81)
    result_add_274 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 14), '+', ans_268, clear_call_result_273)
    
    # Assigning a type to the variable 'ans' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'ans', result_add_274)
    
    
    # Getting the type of 'w' (line 82)
    w_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'w')
    # Getting the type of 'HALF' (line 82)
    HALF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'HALF')
    # Applying the binary operator '<' (line 82)
    result_lt_277 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 16), '<', w_275, HALF_276)
    
    # Testing the type of an if condition (line 82)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_lt_277)
    # SSA begins for while statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 83):
    # Getting the type of 'ans' (line 83)
    ans_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'ans')
    
    # Call to clear(...): (line 83)
    # Processing the call arguments (line 83)
    int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'int')
    # Getting the type of 'charstack' (line 83)
    charstack_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'charstack', False)
    # Processing the call keyword arguments (line 83)
    kwargs_282 = {}
    # Getting the type of 'clear' (line 83)
    clear_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'clear', False)
    # Calling clear(args, kwargs) (line 83)
    clear_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), clear_279, *[int_280, charstack_281], **kwargs_282)
    
    # Applying the binary operator '+' (line 83)
    result_add_284 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 18), '+', ans_278, clear_call_result_283)
    
    # Assigning a type to the variable 'ans' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'ans', result_add_284)
    
    # Getting the type of 'w' (line 84)
    w_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'w')
    int_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'int')
    # Applying the binary operator '*=' (line 84)
    result_imul_287 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '*=', w_285, int_286)
    # Assigning a type to the variable 'w' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'w', result_imul_287)
    
    pass
    # SSA join for while statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    pass
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ans' (line 87)
    ans_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'ans')
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type', ans_288)
    pass
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_289

# Assigning a type to the variable 'encode' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'encode', encode)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'int')
    # Getting the type of 'BETA0' (line 92)
    BETA0_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'BETA0')
    # Getting the type of 'BETA1' (line 92)
    BETA1_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'BETA1')
    int_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 58), 'int')
    int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 68), 'int')
    defaults = [int_290, BETA0_291, BETA1_292, int_293, int_294]
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
    ONE_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 6), 'ONE')
    # Assigning a type to the variable 'b' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'b', ONE_295)
    
    # Assigning a Num to a Name (line 94):
    int_296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 14), 'int')
    # Assigning a type to the variable 'a' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'a', int_296)
    
    # Assigning a Num to a Name (line 94):
    int_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'int')
    # Assigning a type to the variable 'tot0' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'tot0', int_297)
    
    # Assigning a Num to a Name (line 94):
    int_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 35), 'int')
    # Assigning a type to the variable 'tot1' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), 'tot1', int_298)
    # Evaluating assert statement condition
    
    # Getting the type of 'c0' (line 94)
    c0_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 51), 'c0')
    int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 54), 'int')
    # Applying the binary operator '>' (line 94)
    result_gt_301 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 51), '>', c0_299, int_300)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'c1' (line 94)
    c1_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 65), 'c1')
    int_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 68), 'int')
    # Applying the binary operator '>' (line 94)
    result_gt_304 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 65), '>', c1_302, int_303)
    
    
    # Assigning a Num to a Name (line 95):
    int_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 27), 'int')
    # Assigning a type to the variable 'model_needs_updating' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'model_needs_updating', int_305)
    
    
    # Getting the type of 'adaptive' (line 96)
    adaptive_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'adaptive')
    int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'int')
    # Applying the binary operator '==' (line 96)
    result_eq_308 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '==', adaptive_306, int_307)
    
    # Testing the type of an if condition (line 96)
    if_condition_309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_eq_308)
    # Assigning a type to the variable 'if_condition_309' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_309', if_condition_309)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 97):
    # Getting the type of 'c0' (line 97)
    c0_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'c0')
    float_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'float')
    # Applying the binary operator '*' (line 97)
    result_mul_312 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 13), '*', c0_310, float_311)
    
    # Getting the type of 'c0' (line 97)
    c0_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'c0')
    # Getting the type of 'c1' (line 97)
    c1_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'c1')
    # Applying the binary operator '+' (line 97)
    result_add_315 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 21), '+', c0_313, c1_314)
    
    # Applying the binary operator 'div' (line 97)
    result_div_316 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 19), 'div', result_mul_312, result_add_315)
    
    # Assigning a type to the variable 'p0' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'p0', result_div_316)
    pass
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 99):
    str_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'str', '')
    # Assigning a type to the variable 'ans' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'ans', str_317)
    
    # Assigning a Num to a Name (line 100):
    int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 6), 'int')
    # Assigning a type to the variable 'u' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'u', int_318)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'ONE' (line 100)
    ONE_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ONE')
    # Assigning a type to the variable 'v' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 10), 'v', ONE_319)
    
    # Getting the type of 'string' (line 101)
    string_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'string')
    # Testing the type of a for loop iterable (line 101)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 4), string_320)
    # Getting the type of the for loop variable (line 101)
    for_loop_var_321 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 4), string_320)
    # Assigning a type to the variable 'c' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'c', for_loop_var_321)
    # SSA begins for a for statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'N' (line 102)
    N_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'N')
    int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 14), 'int')
    # Applying the binary operator '<=' (line 102)
    result_le_324 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '<=', N_322, int_323)
    
    # Testing the type of an if condition (line 102)
    if_condition_325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_le_324)
    # Assigning a type to the variable 'if_condition_325' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_325', if_condition_325)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    # Evaluating assert statement condition
    
    # Getting the type of 'N' (line 104)
    N_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'N')
    int_327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'int')
    # Applying the binary operator '>' (line 104)
    result_gt_328 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '>', N_326, int_327)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'u' (line 107)
    u_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'u')
    int_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'int')
    # Applying the binary operator '>=' (line 107)
    result_ge_331 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), '>=', u_329, int_330)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'v' (line 107)
    v_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'v')
    # Getting the type of 'ONE' (line 107)
    ONE_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'ONE')
    # Applying the binary operator '<=' (line 107)
    result_le_334 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 29), '<=', v_332, ONE_333)
    
    
    # Assigning a BinOp to a Name (line 108):
    # Getting the type of 'u' (line 108)
    u_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'u')
    # Getting the type of 'v' (line 108)
    v_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'v')
    # Getting the type of 'u' (line 108)
    u_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'u')
    # Applying the binary operator '-' (line 108)
    result_sub_338 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 23), '-', v_336, u_337)
    
    int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'int')
    # Applying the binary operator 'div' (line 108)
    result_div_340 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 22), 'div', result_sub_338, int_339)
    
    # Applying the binary operator '+' (line 108)
    result_add_341 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 18), '+', u_335, result_div_340)
    
    # Assigning a type to the variable 'halfway' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'halfway', result_add_341)
    
    
    # Getting the type of 'c' (line 109)
    c_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'c')
    str_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'str', '1')
    # Applying the binary operator '==' (line 109)
    result_eq_344 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 12), '==', c_342, str_343)
    
    # Testing the type of an if condition (line 109)
    if_condition_345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_344)
    # Assigning a type to the variable 'if_condition_345' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_345', if_condition_345)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 110):
    # Getting the type of 'halfway' (line 110)
    halfway_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'halfway')
    # Assigning a type to the variable 'u' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'u', halfway_346)
    # SSA branch for the else part of an if statement (line 109)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'c' (line 111)
    c_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'c')
    str_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 18), 'str', '0')
    # Applying the binary operator '==' (line 111)
    result_eq_349 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '==', c_347, str_348)
    
    # Testing the type of an if condition (line 111)
    if_condition_350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 13), result_eq_349)
    # Assigning a type to the variable 'if_condition_350' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'if_condition_350', if_condition_350)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'halfway' (line 112)
    halfway_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'halfway')
    # Assigning a type to the variable 'v' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'v', halfway_351)
    # SSA branch for the else part of an if statement (line 111)
    module_type_store.open_ssa_branch('else')
    pass
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    int_352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 15), 'int')
    # Testing the type of an if condition (line 117)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), int_352)
    # SSA begins for while statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Num to a Name (line 118):
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'int')
    # Assigning a type to the variable 'firsttime' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'firsttime', int_353)
    
    # Getting the type of 'model_needs_updating' (line 119)
    model_needs_updating_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'model_needs_updating')
    # Testing the type of an if condition (line 119)
    if_condition_355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), model_needs_updating_354)
    # Assigning a type to the variable 'if_condition_355' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'if_condition_355', if_condition_355)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 120):
    # Getting the type of 'b' (line 120)
    b_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'b')
    # Getting the type of 'a' (line 120)
    a_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'a')
    # Applying the binary operator '-' (line 120)
    result_sub_358 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 20), '-', b_356, a_357)
    
    # Assigning a type to the variable 'w' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'w', result_sub_358)
    
    # Getting the type of 'adaptive' (line 121)
    adaptive_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'adaptive')
    # Testing the type of an if condition (line 121)
    if_condition_360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), adaptive_359)
    # Assigning a type to the variable 'if_condition_360' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_360', if_condition_360)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 122):
    # Getting the type of 'c0' (line 122)
    c0_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'c0')
    # Getting the type of 'c1' (line 122)
    c1_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'c1')
    # Applying the binary operator '+' (line 122)
    result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 25), '+', c0_361, c1_362)
    
    # Assigning a type to the variable 'cT' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'cT', result_add_363)
    
    # Assigning a BinOp to a Name (line 122):
    # Getting the type of 'c0' (line 122)
    c0_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 42), 'c0')
    float_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 46), 'float')
    # Applying the binary operator '*' (line 122)
    result_mul_366 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 42), '*', c0_364, float_365)
    
    # Getting the type of 'cT' (line 122)
    cT_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 50), 'cT')
    # Applying the binary operator 'div' (line 122)
    result_div_368 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 49), 'div', result_mul_366, cT_367)
    
    # Assigning a type to the variable 'p0' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'p0', result_div_368)
    pass
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 124):
    # Getting the type of 'a' (line 124)
    a_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'a')
    
    # Call to int(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'p0' (line 124)
    p0_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'p0', False)
    # Getting the type of 'w' (line 124)
    w_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 38), 'w', False)
    # Applying the binary operator '*' (line 124)
    result_mul_373 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 35), '*', p0_371, w_372)
    
    # Processing the call keyword arguments (line 124)
    kwargs_374 = {}
    # Getting the type of 'int' (line 124)
    int_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'int', False)
    # Calling int(args, kwargs) (line 124)
    int_call_result_375 = invoke(stypy.reporting.localization.Localization(__file__, 124, 31), int_370, *[result_mul_373], **kwargs_374)
    
    # Applying the binary operator '+' (line 124)
    result_add_376 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 27), '+', a_369, int_call_result_375)
    
    # Assigning a type to the variable 'boundary' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'boundary', result_add_376)
    
    
    # Getting the type of 'boundary' (line 125)
    boundary_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'boundary')
    # Getting the type of 'a' (line 125)
    a_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'a')
    # Applying the binary operator '==' (line 125)
    result_eq_379 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 20), '==', boundary_377, a_378)
    
    # Testing the type of an if condition (line 125)
    if_condition_380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_eq_379)
    # Assigning a type to the variable 'if_condition_380' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'if_condition_380', if_condition_380)
    # SSA begins for if statement (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'boundary' (line 125)
    boundary_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'boundary')
    int_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 48), 'int')
    # Applying the binary operator '+=' (line 125)
    result_iadd_383 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 36), '+=', boundary_381, int_382)
    # Assigning a type to the variable 'boundary' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'boundary', result_iadd_383)
    
    str_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 57), 'str', 'warningA')
    pass
    # SSA join for if statement (line 125)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'boundary' (line 126)
    boundary_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'boundary')
    # Getting the type of 'b' (line 126)
    b_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'b')
    # Applying the binary operator '==' (line 126)
    result_eq_387 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 20), '==', boundary_385, b_386)
    
    # Testing the type of an if condition (line 126)
    if_condition_388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 16), result_eq_387)
    # Assigning a type to the variable 'if_condition_388' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'if_condition_388', if_condition_388)
    # SSA begins for if statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'boundary' (line 126)
    boundary_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'boundary')
    int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 48), 'int')
    # Applying the binary operator '-=' (line 126)
    result_isub_391 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 36), '-=', boundary_389, int_390)
    # Assigning a type to the variable 'boundary' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'boundary', result_isub_391)
    
    str_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 57), 'str', 'warningB')
    pass
    # SSA join for if statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 127):
    int_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'int')
    # Assigning a type to the variable 'model_needs_updating' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'model_needs_updating', int_393)
    pass
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'boundary' (line 129)
    boundary_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'boundary')
    # Getting the type of 'u' (line 129)
    u_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'u')
    # Applying the binary operator '<=' (line 129)
    result_le_396 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 18), '<=', boundary_394, u_395)
    
    # Testing the type of an if condition (line 129)
    if_condition_397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 12), result_le_396)
    # Assigning a type to the variable 'if_condition_397' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'if_condition_397', if_condition_397)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 130):
    # Getting the type of 'ans' (line 130)
    ans_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'ans')
    str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 28), 'str', '1')
    # Applying the binary operator '+' (line 130)
    result_add_400 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 22), '+', ans_398, str_399)
    
    # Assigning a type to the variable 'ans' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'ans', result_add_400)
    
    # Getting the type of 'tot1' (line 130)
    tot1_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 45), 'tot1')
    int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 52), 'int')
    # Applying the binary operator '+=' (line 130)
    result_iadd_403 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 45), '+=', tot1_401, int_402)
    # Assigning a type to the variable 'tot1' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 45), 'tot1', result_iadd_403)
    
    
    # Getting the type of 'adaptive' (line 131)
    adaptive_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'adaptive')
    # Testing the type of an if condition (line 131)
    if_condition_405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 16), adaptive_404)
    # Assigning a type to the variable 'if_condition_405' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'if_condition_405', if_condition_405)
    # SSA begins for if statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 131)
    c1_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'c1')
    float_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 35), 'float')
    # Applying the binary operator '+=' (line 131)
    result_iadd_408 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 29), '+=', c1_406, float_407)
    # Assigning a type to the variable 'c1' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'c1', result_iadd_408)
    
    pass
    # SSA join for if statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'boundary' (line 132)
    boundary_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'boundary')
    # Assigning a type to the variable 'a' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'a', boundary_409)
    
    # Assigning a Num to a Name (line 132):
    int_410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'int')
    # Assigning a type to the variable 'model_needs_updating' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'model_needs_updating', int_410)
    
    # Getting the type of 'N' (line 132)
    N_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 59), 'N')
    int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 62), 'int')
    # Applying the binary operator '-=' (line 132)
    result_isub_413 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 59), '-=', N_411, int_412)
    # Assigning a type to the variable 'N' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 59), 'N', result_isub_413)
    
    # SSA branch for the else part of an if statement (line 129)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'boundary' (line 133)
    boundary_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'boundary')
    # Getting the type of 'v' (line 133)
    v_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'v')
    # Applying the binary operator '>=' (line 133)
    result_ge_416 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '>=', boundary_414, v_415)
    
    # Testing the type of an if condition (line 133)
    if_condition_417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 17), result_ge_416)
    # Assigning a type to the variable 'if_condition_417' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'if_condition_417', if_condition_417)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 134):
    # Getting the type of 'ans' (line 134)
    ans_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'ans')
    str_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 28), 'str', '0')
    # Applying the binary operator '+' (line 134)
    result_add_420 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 22), '+', ans_418, str_419)
    
    # Assigning a type to the variable 'ans' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'ans', result_add_420)
    
    # Getting the type of 'tot0' (line 134)
    tot0_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'tot0')
    int_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 52), 'int')
    # Applying the binary operator '+=' (line 134)
    result_iadd_423 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 45), '+=', tot0_421, int_422)
    # Assigning a type to the variable 'tot0' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'tot0', result_iadd_423)
    
    
    # Getting the type of 'adaptive' (line 135)
    adaptive_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'adaptive')
    # Testing the type of an if condition (line 135)
    if_condition_425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 16), adaptive_424)
    # Assigning a type to the variable 'if_condition_425' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'if_condition_425', if_condition_425)
    # SSA begins for if statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c0' (line 135)
    c0_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'c0')
    float_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'float')
    # Applying the binary operator '+=' (line 135)
    result_iadd_428 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 29), '+=', c0_426, float_427)
    # Assigning a type to the variable 'c0' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'c0', result_iadd_428)
    
    pass
    # SSA join for if statement (line 135)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'boundary' (line 136)
    boundary_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'boundary')
    # Assigning a type to the variable 'b' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'b', boundary_429)
    
    # Assigning a Num to a Name (line 136):
    int_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 54), 'int')
    # Assigning a type to the variable 'model_needs_updating' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'model_needs_updating', int_430)
    
    # Getting the type of 'N' (line 136)
    N_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'N')
    int_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 62), 'int')
    # Applying the binary operator '-=' (line 136)
    result_isub_433 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 59), '-=', N_431, int_432)
    # Assigning a type to the variable 'N' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'N', result_isub_433)
    
    # SSA branch for the else part of an if statement (line 133)
    module_type_store.open_ssa_branch('else')
    pass
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 144)
    a_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'a')
    # Getting the type of 'HALF' (line 144)
    HALF_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'HALF')
    # Applying the binary operator '>=' (line 144)
    result_ge_436 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 21), '>=', a_434, HALF_435)
    
    
    # Getting the type of 'b' (line 144)
    b_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'b')
    # Getting the type of 'HALF' (line 144)
    HALF_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'HALF')
    # Applying the binary operator '<=' (line 144)
    result_le_439 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 34), '<=', b_437, HALF_438)
    
    # Applying the binary operator 'or' (line 144)
    result_or_keyword_440 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), 'or', result_ge_436, result_le_439)
    
    # Testing the type of an if condition (line 144)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), result_or_keyword_440)
    # SSA begins for while statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'a' (line 145)
    a_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'a')
    # Getting the type of 'HALF' (line 145)
    HALF_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 23), 'HALF')
    # Applying the binary operator '>=' (line 145)
    result_ge_443 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 20), '>=', a_441, HALF_442)
    
    # Testing the type of an if condition (line 145)
    if_condition_444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 16), result_ge_443)
    # Assigning a type to the variable 'if_condition_444' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'if_condition_444', if_condition_444)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 146):
    # Getting the type of 'a' (line 146)
    a_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'a')
    # Getting the type of 'HALF' (line 146)
    HALF_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'HALF')
    # Applying the binary operator '-' (line 146)
    result_sub_447 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 24), '-', a_445, HALF_446)
    
    # Assigning a type to the variable 'a' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'a', result_sub_447)
    
    # Assigning a BinOp to a Name (line 146):
    # Getting the type of 'b' (line 146)
    b_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'b')
    # Getting the type of 'HALF' (line 146)
    HALF_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'HALF')
    # Applying the binary operator '-' (line 146)
    result_sub_450 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 38), '-', b_448, HALF_449)
    
    # Assigning a type to the variable 'b' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'b', result_sub_450)
    
    # Assigning a BinOp to a Name (line 146):
    # Getting the type of 'u' (line 146)
    u_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 54), 'u')
    # Getting the type of 'HALF' (line 146)
    HALF_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 56), 'HALF')
    # Applying the binary operator '-' (line 146)
    result_sub_453 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 54), '-', u_451, HALF_452)
    
    # Assigning a type to the variable 'u' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 50), 'u', result_sub_453)
    
    # Assigning a BinOp to a Name (line 146):
    # Getting the type of 'v' (line 146)
    v_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 71), 'v')
    # Getting the type of 'HALF' (line 146)
    HALF_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 73), 'HALF')
    # Applying the binary operator '-' (line 146)
    result_sub_456 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 71), '-', v_454, HALF_455)
    
    # Assigning a type to the variable 'v' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 67), 'v', result_sub_456)
    pass
    # SSA branch for the else part of an if statement (line 145)
    module_type_store.open_ssa_branch('else')
    pass
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'a' (line 150)
    a_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'a')
    int_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 21), 'int')
    # Applying the binary operator '*=' (line 150)
    result_imul_459 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 16), '*=', a_457, int_458)
    # Assigning a type to the variable 'a' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'a', result_imul_459)
    
    
    # Getting the type of 'b' (line 150)
    b_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'b')
    int_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 35), 'int')
    # Applying the binary operator '*=' (line 150)
    result_imul_462 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 30), '*=', b_460, int_461)
    # Assigning a type to the variable 'b' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'b', result_imul_462)
    
    
    # Getting the type of 'u' (line 150)
    u_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'u')
    int_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 49), 'int')
    # Applying the binary operator '*=' (line 150)
    result_imul_465 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 44), '*=', u_463, int_464)
    # Assigning a type to the variable 'u' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'u', result_imul_465)
    
    
    # Getting the type of 'v' (line 150)
    v_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 58), 'v')
    int_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 63), 'int')
    # Applying the binary operator '*=' (line 150)
    result_imul_468 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 58), '*=', v_466, int_467)
    # Assigning a type to the variable 'v' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 58), 'v', result_imul_468)
    
    
    # Assigning a Num to a Name (line 151):
    int_469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 39), 'int')
    # Assigning a type to the variable 'model_needs_updating' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'model_needs_updating', int_469)
    pass
    # SSA join for while statement (line 144)
    module_type_store = module_type_store.join_ssa_context()
    
    # Evaluating assert statement condition
    
    # Getting the type of 'a' (line 154)
    a_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'a')
    # Getting the type of 'HALF' (line 154)
    HALF_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'HALF')
    # Applying the binary operator '<=' (line 154)
    result_le_472 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '<=', a_470, HALF_471)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'b' (line 154)
    b_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 46), 'b')
    # Getting the type of 'HALF' (line 154)
    HALF_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'HALF')
    # Applying the binary operator '>=' (line 154)
    result_ge_475 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 46), '>=', b_473, HALF_474)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'a' (line 154)
    a_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 73), 'a')
    int_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 76), 'int')
    # Applying the binary operator '>=' (line 154)
    result_ge_478 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 73), '>=', a_476, int_477)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'b' (line 154)
    b_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 97), 'b')
    # Getting the type of 'ONE' (line 154)
    ONE_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 100), 'ONE')
    # Applying the binary operator '<=' (line 154)
    result_le_481 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 97), '<=', b_479, ONE_480)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 156)
    a_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'a')
    # Getting the type of 'QUARTER' (line 156)
    QUARTER_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'QUARTER')
    # Applying the binary operator '>' (line 156)
    result_gt_484 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 21), '>', a_482, QUARTER_483)
    
    
    # Getting the type of 'b' (line 156)
    b_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'b')
    # Getting the type of 'THREEQU' (line 156)
    THREEQU_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'THREEQU')
    # Applying the binary operator '<' (line 156)
    result_lt_487 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 37), '<', b_485, THREEQU_486)
    
    # Applying the binary operator 'and' (line 156)
    result_and_keyword_488 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 20), 'and', result_gt_484, result_lt_487)
    
    # Testing the type of an if condition (line 156)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 12), result_and_keyword_488)
    # SSA begins for while statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 157):
    int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 20), 'int')
    # Getting the type of 'a' (line 157)
    a_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'a')
    # Applying the binary operator '*' (line 157)
    result_mul_491 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 20), '*', int_489, a_490)
    
    # Getting the type of 'HALF' (line 157)
    HALF_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'HALF')
    # Applying the binary operator '-' (line 157)
    result_sub_493 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 20), '-', result_mul_491, HALF_492)
    
    # Assigning a type to the variable 'a' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'a', result_sub_493)
    
    # Assigning a BinOp to a Name (line 157):
    int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 35), 'int')
    # Getting the type of 'b' (line 157)
    b_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'b')
    # Applying the binary operator '*' (line 157)
    result_mul_496 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 35), '*', int_494, b_495)
    
    # Getting the type of 'HALF' (line 157)
    HALF_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'HALF')
    # Applying the binary operator '-' (line 157)
    result_sub_498 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 35), '-', result_mul_496, HALF_497)
    
    # Assigning a type to the variable 'b' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'b', result_sub_498)
    
    # Assigning a BinOp to a Name (line 157):
    int_499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 50), 'int')
    # Getting the type of 'u' (line 157)
    u_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 52), 'u')
    # Applying the binary operator '*' (line 157)
    result_mul_501 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 50), '*', int_499, u_500)
    
    # Getting the type of 'HALF' (line 157)
    HALF_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 54), 'HALF')
    # Applying the binary operator '-' (line 157)
    result_sub_503 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 50), '-', result_mul_501, HALF_502)
    
    # Assigning a type to the variable 'u' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 46), 'u', result_sub_503)
    
    # Assigning a BinOp to a Name (line 157):
    int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 66), 'int')
    # Getting the type of 'v' (line 157)
    v_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 68), 'v')
    # Applying the binary operator '*' (line 157)
    result_mul_506 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 66), '*', int_504, v_505)
    
    # Getting the type of 'HALF' (line 157)
    HALF_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 70), 'HALF')
    # Applying the binary operator '-' (line 157)
    result_sub_508 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 66), '-', result_mul_506, HALF_507)
    
    # Assigning a type to the variable 'v' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 62), 'v', result_sub_508)
    pass
    # SSA join for while statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'N' (line 159)
    N_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'N')
    int_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
    # Applying the binary operator '>' (line 159)
    result_gt_511 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), '>', N_509, int_510)
    
    # Getting the type of 'model_needs_updating' (line 159)
    model_needs_updating_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'model_needs_updating')
    # Applying the binary operator 'and' (line 159)
    result_and_keyword_513 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), 'and', result_gt_511, model_needs_updating_512)
    
    # Applying the 'not' unary operator (line 159)
    result_not__514 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 15), 'not', result_and_keyword_513)
    
    # Testing the type of an if condition (line 159)
    if_condition_515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 12), result_not__514)
    # Assigning a type to the variable 'if_condition_515' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'if_condition_515', if_condition_515)
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
    ans_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'ans')
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type', ans_516)
    pass
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_517

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
    str_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 31), 'str', 'testdata/BentCoinFile')
    # Processing the call keyword arguments (line 168)
    kwargs_521 = {}
    # Getting the type of 'Relative' (line 168)
    Relative_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 168)
    Relative_call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 168, 22), Relative_519, *[str_520], **kwargs_521)
    
    str_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 58), 'str', 'r')
    # Processing the call keyword arguments (line 168)
    kwargs_524 = {}
    # Getting the type of 'open' (line 168)
    open_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'open', False)
    # Calling open(args, kwargs) (line 168)
    open_call_result_525 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), open_518, *[Relative_call_result_522, str_523], **kwargs_524)
    
    # Assigning a type to the variable 'inputfile' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'inputfile', open_call_result_525)
    
    # Assigning a Call to a Name (line 169):
    
    # Call to open(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Call to Relative(...): (line 169)
    # Processing the call arguments (line 169)
    str_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 169)
    kwargs_529 = {}
    # Getting the type of 'Relative' (line 169)
    Relative_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'Relative', False)
    # Calling Relative(args, kwargs) (line 169)
    Relative_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 169, 23), Relative_527, *[str_528], **kwargs_529)
    
    str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 45), 'str', 'w')
    # Processing the call keyword arguments (line 169)
    kwargs_532 = {}
    # Getting the type of 'open' (line 169)
    open_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'open', False)
    # Calling open(args, kwargs) (line 169)
    open_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 169, 17), open_526, *[Relative_call_result_530, str_531], **kwargs_532)
    
    # Assigning a type to the variable 'outputfile' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'outputfile', open_call_result_533)
    
    # Assigning a Call to a Name (line 172):
    
    # Call to read(...): (line 172)
    # Processing the call keyword arguments (line 172)
    kwargs_536 = {}
    # Getting the type of 'inputfile' (line 172)
    inputfile_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 172)
    read_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), inputfile_534, 'read')
    # Calling read(args, kwargs) (line 172)
    read_call_result_537 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), read_535, *[], **kwargs_536)
    
    # Assigning a type to the variable 's' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 's', read_call_result_537)
    
    # Assigning a Call to a Name (line 173):
    
    # Call to len(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 's' (line 173)
    s_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 's', False)
    # Processing the call keyword arguments (line 173)
    kwargs_540 = {}
    # Getting the type of 'len' (line 173)
    len_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'len', False)
    # Calling len(args, kwargs) (line 173)
    len_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), len_538, *[s_539], **kwargs_540)
    
    # Assigning a type to the variable 'N' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'N', len_call_result_541)
    
    # Assigning a Call to a Name (line 174):
    
    # Call to encode(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 's' (line 174)
    s_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 's', False)
    int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'int')
    int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'int')
    # Processing the call keyword arguments (line 174)
    kwargs_546 = {}
    # Getting the type of 'encode' (line 174)
    encode_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 10), 'encode', False)
    # Calling encode(args, kwargs) (line 174)
    encode_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 174, 10), encode_542, *[s_543, int_544, int_545], **kwargs_546)
    
    # Assigning a type to the variable 'zip' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'zip', encode_call_result_547)
    
    # Call to write(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'zip' (line 175)
    zip_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'zip', False)
    # Processing the call keyword arguments (line 175)
    kwargs_551 = {}
    # Getting the type of 'outputfile' (line 175)
    outputfile_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 175)
    write_549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), outputfile_548, 'write')
    # Calling write(args, kwargs) (line 175)
    write_call_result_552 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), write_549, *[zip_550], **kwargs_551)
    
    
    # Call to close(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_555 = {}
    # Getting the type of 'outputfile' (line 176)
    outputfile_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 176)
    close_554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), outputfile_553, 'close')
    # Calling close(args, kwargs) (line 176)
    close_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), close_554, *[], **kwargs_555)
    
    
    # Call to close(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_559 = {}
    # Getting the type of 'inputfile' (line 176)
    inputfile_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 176)
    close_558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 28), inputfile_557, 'close')
    # Calling close(args, kwargs) (line 176)
    close_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 176, 28), close_558, *[], **kwargs_559)
    
    
    # Assigning a Call to a Name (line 179):
    
    # Call to open(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Call to Relative(...): (line 179)
    # Processing the call arguments (line 179)
    str_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'str', 'tmp.zip')
    # Processing the call keyword arguments (line 179)
    kwargs_564 = {}
    # Getting the type of 'Relative' (line 179)
    Relative_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'Relative', False)
    # Calling Relative(args, kwargs) (line 179)
    Relative_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 179, 22), Relative_562, *[str_563], **kwargs_564)
    
    str_566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 44), 'str', 'r')
    # Processing the call keyword arguments (line 179)
    kwargs_567 = {}
    # Getting the type of 'open' (line 179)
    open_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'open', False)
    # Calling open(args, kwargs) (line 179)
    open_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), open_561, *[Relative_call_result_565, str_566], **kwargs_567)
    
    # Assigning a type to the variable 'inputfile' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'inputfile', open_call_result_568)
    
    # Assigning a Call to a Name (line 180):
    
    # Call to open(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Call to Relative(...): (line 180)
    # Processing the call arguments (line 180)
    str_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 32), 'str', 'tmp2')
    # Processing the call keyword arguments (line 180)
    kwargs_572 = {}
    # Getting the type of 'Relative' (line 180)
    Relative_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'Relative', False)
    # Calling Relative(args, kwargs) (line 180)
    Relative_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 180, 23), Relative_570, *[str_571], **kwargs_572)
    
    str_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 42), 'str', 'w')
    # Processing the call keyword arguments (line 180)
    kwargs_575 = {}
    # Getting the type of 'open' (line 180)
    open_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'open', False)
    # Calling open(args, kwargs) (line 180)
    open_call_result_576 = invoke(stypy.reporting.localization.Localization(__file__, 180, 17), open_569, *[Relative_call_result_573, str_574], **kwargs_575)
    
    # Assigning a type to the variable 'outputfile' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'outputfile', open_call_result_576)
    
    # Assigning a Call to a Name (line 182):
    
    # Call to decode(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Call to list(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Call to read(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_581 = {}
    # Getting the type of 'inputfile' (line 182)
    inputfile_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'inputfile', False)
    # Obtaining the member 'read' of a type (line 182)
    read_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 22), inputfile_579, 'read')
    # Calling read(args, kwargs) (line 182)
    read_call_result_582 = invoke(stypy.reporting.localization.Localization(__file__, 182, 22), read_580, *[], **kwargs_581)
    
    # Processing the call keyword arguments (line 182)
    kwargs_583 = {}
    # Getting the type of 'list' (line 182)
    list_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'list', False)
    # Calling list(args, kwargs) (line 182)
    list_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), list_578, *[read_call_result_582], **kwargs_583)
    
    # Getting the type of 'N' (line 182)
    N_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 41), 'N', False)
    int_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 44), 'int')
    int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 48), 'int')
    # Processing the call keyword arguments (line 182)
    kwargs_588 = {}
    # Getting the type of 'decode' (line 182)
    decode_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 10), 'decode', False)
    # Calling decode(args, kwargs) (line 182)
    decode_call_result_589 = invoke(stypy.reporting.localization.Localization(__file__, 182, 10), decode_577, *[list_call_result_584, N_585, int_586, int_587], **kwargs_588)
    
    # Assigning a type to the variable 'unc' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'unc', decode_call_result_589)
    
    # Call to write(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'unc' (line 183)
    unc_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'unc', False)
    # Processing the call keyword arguments (line 183)
    kwargs_593 = {}
    # Getting the type of 'outputfile' (line 183)
    outputfile_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'outputfile', False)
    # Obtaining the member 'write' of a type (line 183)
    write_591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), outputfile_590, 'write')
    # Calling write(args, kwargs) (line 183)
    write_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), write_591, *[unc_592], **kwargs_593)
    
    
    # Call to close(...): (line 184)
    # Processing the call keyword arguments (line 184)
    kwargs_597 = {}
    # Getting the type of 'outputfile' (line 184)
    outputfile_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'outputfile', False)
    # Obtaining the member 'close' of a type (line 184)
    close_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), outputfile_595, 'close')
    # Calling close(args, kwargs) (line 184)
    close_call_result_598 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), close_596, *[], **kwargs_597)
    
    
    # Call to close(...): (line 184)
    # Processing the call keyword arguments (line 184)
    kwargs_601 = {}
    # Getting the type of 'inputfile' (line 184)
    inputfile_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'inputfile', False)
    # Obtaining the member 'close' of a type (line 184)
    close_600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), inputfile_599, 'close')
    # Calling close(args, kwargs) (line 184)
    close_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 184, 28), close_600, *[], **kwargs_601)
    
    
    # ################# End of 'hardertest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hardertest' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hardertest'
    return stypy_return_type_603

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
    list_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 7), 'list')
    # Adding type elements to the builtin type 'list' instance (line 192)
    # Adding element type (line 192)
    str_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 8), 'str', '1010')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_605)
    # Adding element type (line 192)
    str_606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 16), 'str', '111')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_606)
    # Adding element type (line 192)
    str_607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'str', '00001000000000000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_607)
    # Adding element type (line 192)
    str_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_608)
    # Adding element type (line 192)
    str_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 13), 'str', '10')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_609)
    # Adding element type (line 192)
    str_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 20), 'str', '01')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_610)
    # Adding element type (line 192)
    str_611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'str', '0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_611)
    # Adding element type (line 192)
    str_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', '0000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_612)
    # Adding element type (line 192)
    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 8), 'str', '000000000000000100000000000000000000000000000000100000000000000000011000000')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 7), list_604, str_613)
    
    # Assigning a type to the variable 'sl' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'sl', list_604)
    
    # Getting the type of 'sl' (line 195)
    sl_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 13), 'sl')
    # Testing the type of a for loop iterable (line 195)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 4), sl_614)
    # Getting the type of the for loop variable (line 195)
    for_loop_var_615 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 4), sl_614)
    # Assigning a type to the variable 's' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 's', for_loop_var_615)
    # SSA begins for a for statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 197):
    
    # Call to len(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 's' (line 197)
    s_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 14), 's', False)
    # Processing the call keyword arguments (line 197)
    kwargs_618 = {}
    # Getting the type of 'len' (line 197)
    len_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 10), 'len', False)
    # Calling len(args, kwargs) (line 197)
    len_call_result_619 = invoke(stypy.reporting.localization.Localization(__file__, 197, 10), len_616, *[s_617], **kwargs_618)
    
    # Assigning a type to the variable 'N' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'N', len_call_result_619)
    
    # Assigning a Call to a Name (line 198):
    
    # Call to encode(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 's' (line 198)
    s_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 's', False)
    int_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 21), 'int')
    int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 24), 'int')
    # Processing the call keyword arguments (line 198)
    kwargs_624 = {}
    # Getting the type of 'encode' (line 198)
    encode_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'encode', False)
    # Calling encode(args, kwargs) (line 198)
    encode_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), encode_620, *[s_621, int_622, int_623], **kwargs_624)
    
    # Assigning a type to the variable 'e' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'e', encode_call_result_625)
    
    # Assigning a Call to a Name (line 200):
    
    # Call to decode(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'e' (line 200)
    e_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'e', False)
    # Getting the type of 'N' (line 200)
    N_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'N', False)
    int_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 24), 'int')
    int_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'int')
    # Processing the call keyword arguments (line 200)
    kwargs_631 = {}
    # Getting the type of 'decode' (line 200)
    decode_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'decode', False)
    # Calling decode(args, kwargs) (line 200)
    decode_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), decode_626, *[e_627, N_628, int_629, int_630], **kwargs_631)
    
    # Assigning a type to the variable 'ds' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'ds', decode_call_result_632)
    
    
    # Getting the type of 'ds' (line 202)
    ds_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'ds')
    # Getting the type of 's' (line 202)
    s_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 's')
    # Applying the binary operator '!=' (line 202)
    result_ne_635 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 13), '!=', ds_633, s_634)
    
    # Testing the type of an if condition (line 202)
    if_condition_636 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), result_ne_635)
    # Assigning a type to the variable 'if_condition_636' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_636', if_condition_636)
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
    stypy_return_type_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_637)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test'
    return stypy_return_type_637

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
    kwargs_639 = {}
    # Getting the type of 'test' (line 212)
    test_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'test', False)
    # Calling test(args, kwargs) (line 212)
    test_call_result_640 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), test_638, *[], **kwargs_639)
    
    
    # Call to hardertest(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_642 = {}
    # Getting the type of 'hardertest' (line 213)
    hardertest_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'hardertest', False)
    # Calling hardertest(args, kwargs) (line 213)
    hardertest_call_result_643 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), hardertest_641, *[], **kwargs_642)
    
    # Getting the type of 'True' (line 214)
    True_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', True_644)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 211)
    stypy_return_type_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_645)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_645

# Assigning a type to the variable 'run' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'run', run)

# Call to run(...): (line 216)
# Processing the call keyword arguments (line 216)
kwargs_647 = {}
# Getting the type of 'run' (line 216)
run_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'run', False)
# Calling run(args, kwargs) (line 216)
run_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 216, 0), run_646, *[], **kwargs_647)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

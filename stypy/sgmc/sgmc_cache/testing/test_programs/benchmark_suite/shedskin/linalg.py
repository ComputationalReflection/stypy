
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # (c) Mladen Bestvina
2: #
3: # linear algebra..
4: 
5: import copy
6: 
7: 
8: def inner_prod(v1, v2):
9:     'inner production of two vectors.'
10:     sum = 0
11:     for i in xrange(len(v1)):
12:         sum += v1[i] * v2[i]
13:     return sum
14: 
15: 
16: def matmulttransp(M, N):
17:     'M*N^t.'
18:     return [[inner_prod(v, w) for w in N] for v in M]
19: 
20: 
21: def col(M, j):
22:     v = []
23:     rows = len(M)
24:     for i in xrange(rows):
25:         v.append(M[i][j])
26:     return v
27: 
28: 
29: def Transpose(M):
30:     N = []
31:     cols = len(M[0])
32:     for i in xrange(cols):
33:         N.append(col(M, i))
34:     return N
35: 
36: 
37: def Minor(M, i, j):
38:     M1 = copy.deepcopy(M)
39:     N = [v.pop(j) for v in M1]
40:     M1.pop(i)
41:     return M1
42: 
43: 
44: def sign(n):
45:     return 1 - 2 * (n - 2 * (n / 2))
46: 
47: 
48: def determinant(M):
49:     size = len(M)
50:     if size == 1: return M[0][0]
51:     if size == 2: return M[0][0] * M[1][1] - M[0][1] * M[1][0]  # 1x1 Minors don't work
52:     det = 0
53:     for i in xrange(size):
54:         det += sign(i) * M[0][i] * determinant(Minor(M, 0, i))
55:     return det
56: 
57: 
58: def inverse(M):
59:     size = len(M)
60:     det = determinant(M)
61:     if abs(det) != 1: pass  # print "error, determinant is not 1 or -1"
62:     N = []
63:     for i in xrange(size):
64:         v = []
65:         for j in xrange(size):
66:             v.append(det * sign(i + j) * determinant(Minor(M, j, i)))
67:         N.append(v)
68:     return N
69: 
70: 
71: def iterate_sort(list1, A, B, C, D, E, F):
72:     n = len(list1)
73:     for i in range(n):
74:         z = matmulttransp(list1[i], A)
75:         list1.append(z)
76:         z = matmulttransp(list1[i], B)
77: 
78:         list1.append(z)
79:         z = matmulttransp(list1[i], C)
80: 
81:         list1.append(z)
82:         z = matmulttransp(list1[i], D)
83: 
84:         list1.append(z)
85:         z = matmulttransp(list1[i], E)
86: 
87:         list1.append(z)
88:         z = matmulttransp(list1[i], F)
89: 
90:         list1.append(z)
91: 
92:     list1.sort()
93:     n = len(list1)
94:     last = list1[0]
95:     lasti = i = 1
96:     while i < n:
97:         if list1[i] != last:
98:             list1[lasti] = last = list1[i]
99:             lasti += 1
100:         i += 1
101:     list1.__delslice__(lasti, n)
102: 
103: 
104: def gen(n, list1, A, B, C, D, E, F):
105:     for i in range(n): iterate_sort(list1, A, B, C, D, E, F)
106: 
107: 
108: def inward(U):
109:     b01 = (abs(U[0][0]) < abs(U[0][1])) or ((abs(U[0][0]) == abs(U[0][1]) and abs(U[1][0]) < abs(U[1][1]))) or (
110:     (abs(U[0][0]) == abs(U[0][1]) and abs(U[1][0]) == abs(U[1][1]) and abs(U[2][0]) < abs(U[2][1])))
111: 
112:     b12 = (abs(U[0][1]) < abs(U[0][2])) or ((abs(U[0][1]) == abs(U[0][2]) and abs(U[1][1]) < abs(U[1][2]))) or (
113:     (abs(U[0][1]) == abs(U[0][2]) and abs(U[1][1]) == abs(U[1][2]) and abs(U[2][1]) < abs(U[2][2])))
114: 
115:     return b01 and b12
116: 
117: 
118: def examine(U, i, j):
119:     row1 = abs(i) - 1
120:     row2 = j - 1
121:     s = 1
122:     if i < 0: s = -1
123:     diff = abs(U[0][row1] + s * U[0][row2]) - abs(U[0][row2])
124:     if diff < 0: return -1
125:     if diff > 0:
126:         return 1
127:     else:
128:         diff = abs(U[1][row1] + s * U[1][row2]) - abs(U[1][row2])
129:         if diff < 0: return -1
130:         if diff > 0:
131:             return 1
132:         else:
133:             diff = abs(U[2][row1] + s * U[2][row2]) - abs(U[2][row2])
134:             if diff < 0: return -1
135:             if diff > 0:
136:                 return 1
137:             else:
138:                 return 0
139: 
140: 
141: def examine3(U, i, j, k):
142:     row1 = abs(i) - 1
143:     row2 = abs(j) - 1
144:     row3 = k - 1
145:     s1 = 1
146:     s2 = 1
147:     if i < 0: s1 = -1
148:     if j < 0: s2 = -1
149:     diff = abs(s1 * U[0][row1] + s2 * U[0][row2] + U[0][row3]) - abs(U[0][row3])
150:     if diff < 0: return -1
151:     if diff > 0:
152:         return 1
153:     else:
154:         diff = abs(s1 * U[1][row1] + s2 * U[1][row2] + U[1][row3]) - abs(U[1][row3])
155:         if diff < 0: return -1
156:         if diff > 0:
157:             return 1
158:         else:
159:             diff = abs(s1 * U[2][row1] + s2 * U[2][row2] + U[2][row3]) - abs(U[2][row3])
160:             if diff < 0: return -1
161:             if diff > 0:
162:                 return 1
163:             else:
164:                 return 0
165: 
166: 
167: def binary(n):
168:     if n == 0: return 0
169:     if n == 1: return 1
170:     m = n / 2
171:     if 2 * m == n:
172:         return 10 * binary(m)
173:     else:
174:         return 10 * binary(m) + 1
175: 
176: 
177: length = 6  # wordlength
178: 
179: b = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
180: 
181: A = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
182: B = inverse(A)
183: C = [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
184: D = inverse(B)
185: E = [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
186: F = inverse(E)
187: 
188: At = Transpose(A)
189: Bt = Transpose(B)
190: Ct = Transpose(C)
191: Dt = Transpose(D)
192: Et = Transpose(E)
193: Ft = Transpose(F)
194: 
195: bt = Transpose(b)
196: 
197: 
198: def descending(U):
199:     type = 0
200: 
201:     r = examine(U, 1, 2)
202:     if r == 0: return 1024
203:     if r == -1: type = type + 1
204: 
205:     r = examine(U, -1, 2)
206:     if r == 0: return 1024
207:     if r == -1: type = type + 2
208: 
209:     r = examine(U, 1, 3)
210:     if r == 0: return 1024
211:     if r == -1: type = type + 4
212: 
213:     r = examine(U, -1, 3)
214:     if r == 0: return 1024
215:     if r == -1: type = type + 8
216: 
217:     r = examine(U, 2, 3)
218:     if r == 0: return 1024
219:     if r == -1: type = type + 16
220: 
221:     r = examine(U, -2, 3)
222:     if r == 0: return 1024
223:     if r == -1: type = type + 32
224: 
225:     r = examine3(U, 1, 2, 3)
226:     if r == 0: return 1024
227:     if r == -1: type = type + 64
228: 
229:     r = examine3(U, -1, -2, 3)
230:     if r == 0: return 1024
231:     if r == -1: type = type + 128
232: 
233:     r = examine3(U, -1, 2, 3)
234:     if r == 0: return 1024
235:     if r == -1: type = type + 256
236: 
237:     r = examine3(U, 1, -2, 3)
238:     if r == 0: return 1024
239:     if r == -1: type = type + 512
240: 
241:     return type
242: 
243: 
244: def main2():
245:     list1 = [bt]
246:     gen(length, list1, A, B, C, D, E, F)
247:     inlist = [x for x in list1 if inward(x)]
248:     types = [0] * 1025
249:     for U in inlist:
250:         t = descending(U)
251:         types[t] += 1
252:         if t in [22, 25, 37, 42, 6, 9, 73, 262]:
253:             pass  # print t,U
254:     # print
255:     for t in reversed(range(1025)):
256:         if types[t] > 0:
257:             binary(t)
258:             # print t, binary(t), types[t]
259:             break
260:         # print(' %03i   %012i   %i  ' %(t,binary(t),types[t]))
261: 
262: 
263: def run():
264:     for x in range(10):
265:         main2()
266:     return True
267: 
268: 
269: run()
270: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import copy' statement (line 5)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'copy', copy, module_type_store)


@norecursion
def inner_prod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'inner_prod'
    module_type_store = module_type_store.open_function_context('inner_prod', 8, 0, False)
    
    # Passed parameters checking function
    inner_prod.stypy_localization = localization
    inner_prod.stypy_type_of_self = None
    inner_prod.stypy_type_store = module_type_store
    inner_prod.stypy_function_name = 'inner_prod'
    inner_prod.stypy_param_names_list = ['v1', 'v2']
    inner_prod.stypy_varargs_param_name = None
    inner_prod.stypy_kwargs_param_name = None
    inner_prod.stypy_call_defaults = defaults
    inner_prod.stypy_call_varargs = varargs
    inner_prod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inner_prod', ['v1', 'v2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inner_prod', localization, ['v1', 'v2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inner_prod(...)' code ##################

    str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'inner production of two vectors.')
    
    # Assigning a Num to a Name (line 10):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'int')
    # Assigning a type to the variable 'sum' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'sum', int_2)
    
    
    # Call to xrange(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to len(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'v1' (line 11)
    v1_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'v1', False)
    # Processing the call keyword arguments (line 11)
    kwargs_6 = {}
    # Getting the type of 'len' (line 11)
    len_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'len', False)
    # Calling len(args, kwargs) (line 11)
    len_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 11, 20), len_4, *[v1_5], **kwargs_6)
    
    # Processing the call keyword arguments (line 11)
    kwargs_8 = {}
    # Getting the type of 'xrange' (line 11)
    xrange_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 11)
    xrange_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), xrange_3, *[len_call_result_7], **kwargs_8)
    
    # Testing if the for loop is going to be iterated (line 11)
    # Testing the type of a for loop iterable (line 11)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 11, 4), xrange_call_result_9)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 11, 4), xrange_call_result_9):
        # Getting the type of the for loop variable (line 11)
        for_loop_var_10 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 11, 4), xrange_call_result_9)
        # Assigning a type to the variable 'i' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'i', for_loop_var_10)
        # SSA begins for a for statement (line 11)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'sum' (line 12)
        sum_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'sum')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 12)
        i_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'i')
        # Getting the type of 'v1' (line 12)
        v1_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'v1')
        # Obtaining the member '__getitem__' of a type (line 12)
        getitem___14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 15), v1_13, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 12)
        subscript_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 12, 15), getitem___14, i_12)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 12)
        i_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 26), 'i')
        # Getting the type of 'v2' (line 12)
        v2_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 23), 'v2')
        # Obtaining the member '__getitem__' of a type (line 12)
        getitem___18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 23), v2_17, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 12)
        subscript_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 12, 23), getitem___18, i_16)
        
        # Applying the binary operator '*' (line 12)
        result_mul_20 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 15), '*', subscript_call_result_15, subscript_call_result_19)
        
        # Applying the binary operator '+=' (line 12)
        result_iadd_21 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 8), '+=', sum_11, result_mul_20)
        # Assigning a type to the variable 'sum' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'sum', result_iadd_21)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'sum' (line 13)
    sum_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'sum')
    # Assigning a type to the variable 'stypy_return_type' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type', sum_22)
    
    # ################# End of 'inner_prod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inner_prod' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inner_prod'
    return stypy_return_type_23

# Assigning a type to the variable 'inner_prod' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'inner_prod', inner_prod)

@norecursion
def matmulttransp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matmulttransp'
    module_type_store = module_type_store.open_function_context('matmulttransp', 16, 0, False)
    
    # Passed parameters checking function
    matmulttransp.stypy_localization = localization
    matmulttransp.stypy_type_of_self = None
    matmulttransp.stypy_type_store = module_type_store
    matmulttransp.stypy_function_name = 'matmulttransp'
    matmulttransp.stypy_param_names_list = ['M', 'N']
    matmulttransp.stypy_varargs_param_name = None
    matmulttransp.stypy_kwargs_param_name = None
    matmulttransp.stypy_call_defaults = defaults
    matmulttransp.stypy_call_varargs = varargs
    matmulttransp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matmulttransp', ['M', 'N'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matmulttransp', localization, ['M', 'N'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matmulttransp(...)' code ##################

    str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'M*N^t.')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'M' (line 18)
    M_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 51), 'M')
    comprehension_34 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 12), M_33)
    # Assigning a type to the variable 'v' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'v', comprehension_34)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'N' (line 18)
    N_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 39), 'N')
    comprehension_31 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), N_30)
    # Assigning a type to the variable 'w' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'w', comprehension_31)
    
    # Call to inner_prod(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'v' (line 18)
    v_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'v', False)
    # Getting the type of 'w' (line 18)
    w_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'w', False)
    # Processing the call keyword arguments (line 18)
    kwargs_28 = {}
    # Getting the type of 'inner_prod' (line 18)
    inner_prod_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'inner_prod', False)
    # Calling inner_prod(args, kwargs) (line 18)
    inner_prod_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), inner_prod_25, *[v_26, w_27], **kwargs_28)
    
    list_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_32, inner_prod_call_result_29)
    list_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 12), list_35, list_32)
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', list_35)
    
    # ################# End of 'matmulttransp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matmulttransp' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matmulttransp'
    return stypy_return_type_36

# Assigning a type to the variable 'matmulttransp' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matmulttransp', matmulttransp)

@norecursion
def col(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'col'
    module_type_store = module_type_store.open_function_context('col', 21, 0, False)
    
    # Passed parameters checking function
    col.stypy_localization = localization
    col.stypy_type_of_self = None
    col.stypy_type_store = module_type_store
    col.stypy_function_name = 'col'
    col.stypy_param_names_list = ['M', 'j']
    col.stypy_varargs_param_name = None
    col.stypy_kwargs_param_name = None
    col.stypy_call_defaults = defaults
    col.stypy_call_varargs = varargs
    col.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'col', ['M', 'j'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'col', localization, ['M', 'j'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'col(...)' code ##################

    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    
    # Assigning a type to the variable 'v' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'v', list_37)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to len(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'M' (line 23)
    M_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'M', False)
    # Processing the call keyword arguments (line 23)
    kwargs_40 = {}
    # Getting the type of 'len' (line 23)
    len_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'len', False)
    # Calling len(args, kwargs) (line 23)
    len_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), len_38, *[M_39], **kwargs_40)
    
    # Assigning a type to the variable 'rows' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'rows', len_call_result_41)
    
    
    # Call to xrange(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'rows' (line 24)
    rows_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'rows', False)
    # Processing the call keyword arguments (line 24)
    kwargs_44 = {}
    # Getting the type of 'xrange' (line 24)
    xrange_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 24)
    xrange_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), xrange_42, *[rows_43], **kwargs_44)
    
    # Testing if the for loop is going to be iterated (line 24)
    # Testing the type of a for loop iterable (line 24)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 24, 4), xrange_call_result_45)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 24, 4), xrange_call_result_45):
        # Getting the type of the for loop variable (line 24)
        for_loop_var_46 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 24, 4), xrange_call_result_45)
        # Assigning a type to the variable 'i' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'i', for_loop_var_46)
        # SSA begins for a for statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 25)
        j_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'j', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 25)
        i_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'i', False)
        # Getting the type of 'M' (line 25)
        M_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'M', False)
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 17), M_51, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), getitem___52, i_50)
        
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 17), subscript_call_result_53, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), getitem___54, j_49)
        
        # Processing the call keyword arguments (line 25)
        kwargs_56 = {}
        # Getting the type of 'v' (line 25)
        v_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'v', False)
        # Obtaining the member 'append' of a type (line 25)
        append_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), v_47, 'append')
        # Calling append(args, kwargs) (line 25)
        append_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), append_48, *[subscript_call_result_55], **kwargs_56)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'v' (line 26)
    v_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'v')
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', v_58)
    
    # ################# End of 'col(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'col' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'col'
    return stypy_return_type_59

# Assigning a type to the variable 'col' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'col', col)

@norecursion
def Transpose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Transpose'
    module_type_store = module_type_store.open_function_context('Transpose', 29, 0, False)
    
    # Passed parameters checking function
    Transpose.stypy_localization = localization
    Transpose.stypy_type_of_self = None
    Transpose.stypy_type_store = module_type_store
    Transpose.stypy_function_name = 'Transpose'
    Transpose.stypy_param_names_list = ['M']
    Transpose.stypy_varargs_param_name = None
    Transpose.stypy_kwargs_param_name = None
    Transpose.stypy_call_defaults = defaults
    Transpose.stypy_call_varargs = varargs
    Transpose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Transpose', ['M'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Transpose', localization, ['M'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Transpose(...)' code ##################

    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    
    # Assigning a type to the variable 'N' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'N', list_60)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to len(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Obtaining the type of the subscript
    int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'int')
    # Getting the type of 'M' (line 31)
    M_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'M', False)
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), M_63, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), getitem___64, int_62)
    
    # Processing the call keyword arguments (line 31)
    kwargs_66 = {}
    # Getting the type of 'len' (line 31)
    len_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'len', False)
    # Calling len(args, kwargs) (line 31)
    len_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), len_61, *[subscript_call_result_65], **kwargs_66)
    
    # Assigning a type to the variable 'cols' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'cols', len_call_result_67)
    
    
    # Call to xrange(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'cols' (line 32)
    cols_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'cols', False)
    # Processing the call keyword arguments (line 32)
    kwargs_70 = {}
    # Getting the type of 'xrange' (line 32)
    xrange_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 32)
    xrange_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 32, 13), xrange_68, *[cols_69], **kwargs_70)
    
    # Testing if the for loop is going to be iterated (line 32)
    # Testing the type of a for loop iterable (line 32)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 4), xrange_call_result_71)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 32, 4), xrange_call_result_71):
        # Getting the type of the for loop variable (line 32)
        for_loop_var_72 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 4), xrange_call_result_71)
        # Assigning a type to the variable 'i' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'i', for_loop_var_72)
        # SSA begins for a for statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to col(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'M' (line 33)
        M_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'M', False)
        # Getting the type of 'i' (line 33)
        i_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'i', False)
        # Processing the call keyword arguments (line 33)
        kwargs_78 = {}
        # Getting the type of 'col' (line 33)
        col_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'col', False)
        # Calling col(args, kwargs) (line 33)
        col_call_result_79 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), col_75, *[M_76, i_77], **kwargs_78)
        
        # Processing the call keyword arguments (line 33)
        kwargs_80 = {}
        # Getting the type of 'N' (line 33)
        N_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'N', False)
        # Obtaining the member 'append' of a type (line 33)
        append_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), N_73, 'append')
        # Calling append(args, kwargs) (line 33)
        append_call_result_81 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), append_74, *[col_call_result_79], **kwargs_80)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'N' (line 34)
    N_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'N')
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', N_82)
    
    # ################# End of 'Transpose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Transpose' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_83)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Transpose'
    return stypy_return_type_83

# Assigning a type to the variable 'Transpose' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'Transpose', Transpose)

@norecursion
def Minor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Minor'
    module_type_store = module_type_store.open_function_context('Minor', 37, 0, False)
    
    # Passed parameters checking function
    Minor.stypy_localization = localization
    Minor.stypy_type_of_self = None
    Minor.stypy_type_store = module_type_store
    Minor.stypy_function_name = 'Minor'
    Minor.stypy_param_names_list = ['M', 'i', 'j']
    Minor.stypy_varargs_param_name = None
    Minor.stypy_kwargs_param_name = None
    Minor.stypy_call_defaults = defaults
    Minor.stypy_call_varargs = varargs
    Minor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Minor', ['M', 'i', 'j'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Minor', localization, ['M', 'i', 'j'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Minor(...)' code ##################

    
    # Assigning a Call to a Name (line 38):
    
    # Call to deepcopy(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'M' (line 38)
    M_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'M', False)
    # Processing the call keyword arguments (line 38)
    kwargs_87 = {}
    # Getting the type of 'copy' (line 38)
    copy_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 38)
    deepcopy_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), copy_84, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 38)
    deepcopy_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), deepcopy_85, *[M_86], **kwargs_87)
    
    # Assigning a type to the variable 'M1' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'M1', deepcopy_call_result_88)
    
    # Assigning a ListComp to a Name (line 39):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'M1' (line 39)
    M1_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'M1')
    comprehension_95 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), M1_94)
    # Assigning a type to the variable 'v' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'v', comprehension_95)
    
    # Call to pop(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'j' (line 39)
    j_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'j', False)
    # Processing the call keyword arguments (line 39)
    kwargs_92 = {}
    # Getting the type of 'v' (line 39)
    v_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'v', False)
    # Obtaining the member 'pop' of a type (line 39)
    pop_90 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 9), v_89, 'pop')
    # Calling pop(args, kwargs) (line 39)
    pop_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 39, 9), pop_90, *[j_91], **kwargs_92)
    
    list_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), list_96, pop_call_result_93)
    # Assigning a type to the variable 'N' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'N', list_96)
    
    # Call to pop(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'i' (line 40)
    i_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'i', False)
    # Processing the call keyword arguments (line 40)
    kwargs_100 = {}
    # Getting the type of 'M1' (line 40)
    M1_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'M1', False)
    # Obtaining the member 'pop' of a type (line 40)
    pop_98 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), M1_97, 'pop')
    # Calling pop(args, kwargs) (line 40)
    pop_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), pop_98, *[i_99], **kwargs_100)
    
    # Getting the type of 'M1' (line 41)
    M1_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'M1')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', M1_102)
    
    # ################# End of 'Minor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Minor' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_103)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Minor'
    return stypy_return_type_103

# Assigning a type to the variable 'Minor' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'Minor', Minor)

@norecursion
def sign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sign'
    module_type_store = module_type_store.open_function_context('sign', 44, 0, False)
    
    # Passed parameters checking function
    sign.stypy_localization = localization
    sign.stypy_type_of_self = None
    sign.stypy_type_store = module_type_store
    sign.stypy_function_name = 'sign'
    sign.stypy_param_names_list = ['n']
    sign.stypy_varargs_param_name = None
    sign.stypy_kwargs_param_name = None
    sign.stypy_call_defaults = defaults
    sign.stypy_call_varargs = varargs
    sign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sign', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sign', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sign(...)' code ##################

    int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'int')
    int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'int')
    # Getting the type of 'n' (line 45)
    n_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'n')
    int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'int')
    # Getting the type of 'n' (line 45)
    n_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'n')
    int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'int')
    # Applying the binary operator 'div' (line 45)
    result_div_110 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 29), 'div', n_108, int_109)
    
    # Applying the binary operator '*' (line 45)
    result_mul_111 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 24), '*', int_107, result_div_110)
    
    # Applying the binary operator '-' (line 45)
    result_sub_112 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 20), '-', n_106, result_mul_111)
    
    # Applying the binary operator '*' (line 45)
    result_mul_113 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 15), '*', int_105, result_sub_112)
    
    # Applying the binary operator '-' (line 45)
    result_sub_114 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '-', int_104, result_mul_113)
    
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type', result_sub_114)
    
    # ################# End of 'sign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sign' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sign'
    return stypy_return_type_115

# Assigning a type to the variable 'sign' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'sign', sign)

@norecursion
def determinant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'determinant'
    module_type_store = module_type_store.open_function_context('determinant', 48, 0, False)
    
    # Passed parameters checking function
    determinant.stypy_localization = localization
    determinant.stypy_type_of_self = None
    determinant.stypy_type_store = module_type_store
    determinant.stypy_function_name = 'determinant'
    determinant.stypy_param_names_list = ['M']
    determinant.stypy_varargs_param_name = None
    determinant.stypy_kwargs_param_name = None
    determinant.stypy_call_defaults = defaults
    determinant.stypy_call_varargs = varargs
    determinant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'determinant', ['M'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'determinant', localization, ['M'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'determinant(...)' code ##################

    
    # Assigning a Call to a Name (line 49):
    
    # Call to len(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'M' (line 49)
    M_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'M', False)
    # Processing the call keyword arguments (line 49)
    kwargs_118 = {}
    # Getting the type of 'len' (line 49)
    len_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'len', False)
    # Calling len(args, kwargs) (line 49)
    len_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), len_116, *[M_117], **kwargs_118)
    
    # Assigning a type to the variable 'size' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'size', len_call_result_119)
    
    # Getting the type of 'size' (line 50)
    size_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'size')
    int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 15), 'int')
    # Applying the binary operator '==' (line 50)
    result_eq_122 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), '==', size_120, int_121)
    
    # Testing if the type of an if condition is none (line 50)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 50, 4), result_eq_122):
        pass
    else:
        
        # Testing the type of an if condition (line 50)
        if_condition_123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), result_eq_122)
        # Assigning a type to the variable 'if_condition_123' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_123', if_condition_123)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
        
        # Obtaining the type of the subscript
        int_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 27), 'int')
        # Getting the type of 'M' (line 50)
        M_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'M')
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), M_126, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 50, 25), getitem___127, int_125)
        
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), subscript_call_result_128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 50, 25), getitem___129, int_124)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'stypy_return_type', subscript_call_result_130)
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'size' (line 51)
    size_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'size')
    int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'int')
    # Applying the binary operator '==' (line 51)
    result_eq_133 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), '==', size_131, int_132)
    
    # Testing if the type of an if condition is none (line 51)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 51, 4), result_eq_133):
        pass
    else:
        
        # Testing the type of an if condition (line 51)
        if_condition_134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 4), result_eq_133)
        # Assigning a type to the variable 'if_condition_134' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'if_condition_134', if_condition_134)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'int')
        
        # Obtaining the type of the subscript
        int_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'int')
        # Getting the type of 'M' (line 51)
        M_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'M')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), M_137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), getitem___138, int_136)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), subscript_call_result_139, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), getitem___140, int_135)
        
        
        # Obtaining the type of the subscript
        int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 40), 'int')
        
        # Obtaining the type of the subscript
        int_143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 37), 'int')
        # Getting the type of 'M' (line 51)
        M_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 35), 'M')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 35), M_144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_146 = invoke(stypy.reporting.localization.Localization(__file__, 51, 35), getitem___145, int_143)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 35), subscript_call_result_146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_148 = invoke(stypy.reporting.localization.Localization(__file__, 51, 35), getitem___147, int_142)
        
        # Applying the binary operator '*' (line 51)
        result_mul_149 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 25), '*', subscript_call_result_141, subscript_call_result_148)
        
        
        # Obtaining the type of the subscript
        int_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 50), 'int')
        
        # Obtaining the type of the subscript
        int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'int')
        # Getting the type of 'M' (line 51)
        M_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 45), 'M')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 45), M_152, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 51, 45), getitem___153, int_151)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 45), subscript_call_result_154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 51, 45), getitem___155, int_150)
        
        
        # Obtaining the type of the subscript
        int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 60), 'int')
        
        # Obtaining the type of the subscript
        int_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 57), 'int')
        # Getting the type of 'M' (line 51)
        M_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 55), 'M')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 55), M_159, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 51, 55), getitem___160, int_158)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 55), subscript_call_result_161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 51, 55), getitem___162, int_157)
        
        # Applying the binary operator '*' (line 51)
        result_mul_164 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 45), '*', subscript_call_result_156, subscript_call_result_163)
        
        # Applying the binary operator '-' (line 51)
        result_sub_165 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 25), '-', result_mul_149, result_mul_164)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'stypy_return_type', result_sub_165)
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Num to a Name (line 52):
    int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 10), 'int')
    # Assigning a type to the variable 'det' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'det', int_166)
    
    
    # Call to xrange(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'size' (line 53)
    size_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'size', False)
    # Processing the call keyword arguments (line 53)
    kwargs_169 = {}
    # Getting the type of 'xrange' (line 53)
    xrange_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 53)
    xrange_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), xrange_167, *[size_168], **kwargs_169)
    
    # Testing if the for loop is going to be iterated (line 53)
    # Testing the type of a for loop iterable (line 53)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 4), xrange_call_result_170)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 53, 4), xrange_call_result_170):
        # Getting the type of the for loop variable (line 53)
        for_loop_var_171 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 4), xrange_call_result_170)
        # Assigning a type to the variable 'i' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'i', for_loop_var_171)
        # SSA begins for a for statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'det' (line 54)
        det_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'det')
        
        # Call to sign(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'i' (line 54)
        i_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'i', False)
        # Processing the call keyword arguments (line 54)
        kwargs_175 = {}
        # Getting the type of 'sign' (line 54)
        sign_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'sign', False)
        # Calling sign(args, kwargs) (line 54)
        sign_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 54, 15), sign_173, *[i_174], **kwargs_175)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 54)
        i_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'i')
        
        # Obtaining the type of the subscript
        int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
        # Getting the type of 'M' (line 54)
        M_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'M')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), M_179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 54, 25), getitem___180, int_178)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), subscript_call_result_181, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 54, 25), getitem___182, i_177)
        
        # Applying the binary operator '*' (line 54)
        result_mul_184 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), '*', sign_call_result_176, subscript_call_result_183)
        
        
        # Call to determinant(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to Minor(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'M' (line 54)
        M_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'M', False)
        int_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 56), 'int')
        # Getting the type of 'i' (line 54)
        i_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 59), 'i', False)
        # Processing the call keyword arguments (line 54)
        kwargs_190 = {}
        # Getting the type of 'Minor' (line 54)
        Minor_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'Minor', False)
        # Calling Minor(args, kwargs) (line 54)
        Minor_call_result_191 = invoke(stypy.reporting.localization.Localization(__file__, 54, 47), Minor_186, *[M_187, int_188, i_189], **kwargs_190)
        
        # Processing the call keyword arguments (line 54)
        kwargs_192 = {}
        # Getting the type of 'determinant' (line 54)
        determinant_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 35), 'determinant', False)
        # Calling determinant(args, kwargs) (line 54)
        determinant_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 54, 35), determinant_185, *[Minor_call_result_191], **kwargs_192)
        
        # Applying the binary operator '*' (line 54)
        result_mul_194 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 33), '*', result_mul_184, determinant_call_result_193)
        
        # Applying the binary operator '+=' (line 54)
        result_iadd_195 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 8), '+=', det_172, result_mul_194)
        # Assigning a type to the variable 'det' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'det', result_iadd_195)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'det' (line 55)
    det_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'det')
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type', det_196)
    
    # ################# End of 'determinant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'determinant' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_197)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'determinant'
    return stypy_return_type_197

# Assigning a type to the variable 'determinant' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'determinant', determinant)

@norecursion
def inverse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'inverse'
    module_type_store = module_type_store.open_function_context('inverse', 58, 0, False)
    
    # Passed parameters checking function
    inverse.stypy_localization = localization
    inverse.stypy_type_of_self = None
    inverse.stypy_type_store = module_type_store
    inverse.stypy_function_name = 'inverse'
    inverse.stypy_param_names_list = ['M']
    inverse.stypy_varargs_param_name = None
    inverse.stypy_kwargs_param_name = None
    inverse.stypy_call_defaults = defaults
    inverse.stypy_call_varargs = varargs
    inverse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inverse', ['M'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inverse', localization, ['M'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inverse(...)' code ##################

    
    # Assigning a Call to a Name (line 59):
    
    # Call to len(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'M' (line 59)
    M_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'M', False)
    # Processing the call keyword arguments (line 59)
    kwargs_200 = {}
    # Getting the type of 'len' (line 59)
    len_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'len', False)
    # Calling len(args, kwargs) (line 59)
    len_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 59, 11), len_198, *[M_199], **kwargs_200)
    
    # Assigning a type to the variable 'size' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'size', len_call_result_201)
    
    # Assigning a Call to a Name (line 60):
    
    # Call to determinant(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'M' (line 60)
    M_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'M', False)
    # Processing the call keyword arguments (line 60)
    kwargs_204 = {}
    # Getting the type of 'determinant' (line 60)
    determinant_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 10), 'determinant', False)
    # Calling determinant(args, kwargs) (line 60)
    determinant_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 60, 10), determinant_202, *[M_203], **kwargs_204)
    
    # Assigning a type to the variable 'det' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'det', determinant_call_result_205)
    
    
    # Call to abs(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'det' (line 61)
    det_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'det', False)
    # Processing the call keyword arguments (line 61)
    kwargs_208 = {}
    # Getting the type of 'abs' (line 61)
    abs_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'abs', False)
    # Calling abs(args, kwargs) (line 61)
    abs_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 61, 7), abs_206, *[det_207], **kwargs_208)
    
    int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'int')
    # Applying the binary operator '!=' (line 61)
    result_ne_211 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), '!=', abs_call_result_209, int_210)
    
    # Testing if the type of an if condition is none (line 61)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 61, 4), result_ne_211):
        pass
    else:
        
        # Testing the type of an if condition (line 61)
        if_condition_212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), result_ne_211)
        # Assigning a type to the variable 'if_condition_212' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'if_condition_212', if_condition_212)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 62):
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    
    # Assigning a type to the variable 'N' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'N', list_213)
    
    
    # Call to xrange(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'size' (line 63)
    size_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'size', False)
    # Processing the call keyword arguments (line 63)
    kwargs_216 = {}
    # Getting the type of 'xrange' (line 63)
    xrange_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 63)
    xrange_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), xrange_214, *[size_215], **kwargs_216)
    
    # Testing if the for loop is going to be iterated (line 63)
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 4), xrange_call_result_217)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 63, 4), xrange_call_result_217):
        # Getting the type of the for loop variable (line 63)
        for_loop_var_218 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 4), xrange_call_result_217)
        # Assigning a type to the variable 'i' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'i', for_loop_var_218)
        # SSA begins for a for statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 64):
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        
        # Assigning a type to the variable 'v' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'v', list_219)
        
        
        # Call to xrange(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'size' (line 65)
        size_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'size', False)
        # Processing the call keyword arguments (line 65)
        kwargs_222 = {}
        # Getting the type of 'xrange' (line 65)
        xrange_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 65)
        xrange_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), xrange_220, *[size_221], **kwargs_222)
        
        # Testing if the for loop is going to be iterated (line 65)
        # Testing the type of a for loop iterable (line 65)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 8), xrange_call_result_223)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 65, 8), xrange_call_result_223):
            # Getting the type of the for loop variable (line 65)
            for_loop_var_224 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 8), xrange_call_result_223)
            # Assigning a type to the variable 'j' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'j', for_loop_var_224)
            # SSA begins for a for statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'det' (line 66)
            det_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'det', False)
            
            # Call to sign(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'i' (line 66)
            i_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'i', False)
            # Getting the type of 'j' (line 66)
            j_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 36), 'j', False)
            # Applying the binary operator '+' (line 66)
            result_add_231 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 32), '+', i_229, j_230)
            
            # Processing the call keyword arguments (line 66)
            kwargs_232 = {}
            # Getting the type of 'sign' (line 66)
            sign_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 'sign', False)
            # Calling sign(args, kwargs) (line 66)
            sign_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 66, 27), sign_228, *[result_add_231], **kwargs_232)
            
            # Applying the binary operator '*' (line 66)
            result_mul_234 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 21), '*', det_227, sign_call_result_233)
            
            
            # Call to determinant(...): (line 66)
            # Processing the call arguments (line 66)
            
            # Call to Minor(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'M' (line 66)
            M_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 59), 'M', False)
            # Getting the type of 'j' (line 66)
            j_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 62), 'j', False)
            # Getting the type of 'i' (line 66)
            i_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 65), 'i', False)
            # Processing the call keyword arguments (line 66)
            kwargs_240 = {}
            # Getting the type of 'Minor' (line 66)
            Minor_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 53), 'Minor', False)
            # Calling Minor(args, kwargs) (line 66)
            Minor_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 66, 53), Minor_236, *[M_237, j_238, i_239], **kwargs_240)
            
            # Processing the call keyword arguments (line 66)
            kwargs_242 = {}
            # Getting the type of 'determinant' (line 66)
            determinant_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 41), 'determinant', False)
            # Calling determinant(args, kwargs) (line 66)
            determinant_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 66, 41), determinant_235, *[Minor_call_result_241], **kwargs_242)
            
            # Applying the binary operator '*' (line 66)
            result_mul_244 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 39), '*', result_mul_234, determinant_call_result_243)
            
            # Processing the call keyword arguments (line 66)
            kwargs_245 = {}
            # Getting the type of 'v' (line 66)
            v_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'v', False)
            # Obtaining the member 'append' of a type (line 66)
            append_226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), v_225, 'append')
            # Calling append(args, kwargs) (line 66)
            append_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), append_226, *[result_mul_244], **kwargs_245)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'v' (line 67)
        v_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'v', False)
        # Processing the call keyword arguments (line 67)
        kwargs_250 = {}
        # Getting the type of 'N' (line 67)
        N_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'N', False)
        # Obtaining the member 'append' of a type (line 67)
        append_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), N_247, 'append')
        # Calling append(args, kwargs) (line 67)
        append_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), append_248, *[v_249], **kwargs_250)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'N' (line 68)
    N_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'N')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', N_252)
    
    # ################# End of 'inverse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inverse' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_253)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inverse'
    return stypy_return_type_253

# Assigning a type to the variable 'inverse' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'inverse', inverse)

@norecursion
def iterate_sort(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterate_sort'
    module_type_store = module_type_store.open_function_context('iterate_sort', 71, 0, False)
    
    # Passed parameters checking function
    iterate_sort.stypy_localization = localization
    iterate_sort.stypy_type_of_self = None
    iterate_sort.stypy_type_store = module_type_store
    iterate_sort.stypy_function_name = 'iterate_sort'
    iterate_sort.stypy_param_names_list = ['list1', 'A', 'B', 'C', 'D', 'E', 'F']
    iterate_sort.stypy_varargs_param_name = None
    iterate_sort.stypy_kwargs_param_name = None
    iterate_sort.stypy_call_defaults = defaults
    iterate_sort.stypy_call_varargs = varargs
    iterate_sort.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterate_sort', ['list1', 'A', 'B', 'C', 'D', 'E', 'F'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterate_sort', localization, ['list1', 'A', 'B', 'C', 'D', 'E', 'F'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterate_sort(...)' code ##################

    
    # Assigning a Call to a Name (line 72):
    
    # Call to len(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'list1' (line 72)
    list1_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'list1', False)
    # Processing the call keyword arguments (line 72)
    kwargs_256 = {}
    # Getting the type of 'len' (line 72)
    len_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'len', False)
    # Calling len(args, kwargs) (line 72)
    len_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), len_254, *[list1_255], **kwargs_256)
    
    # Assigning a type to the variable 'n' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'n', len_call_result_257)
    
    
    # Call to range(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'n' (line 73)
    n_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'n', False)
    # Processing the call keyword arguments (line 73)
    kwargs_260 = {}
    # Getting the type of 'range' (line 73)
    range_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'range', False)
    # Calling range(args, kwargs) (line 73)
    range_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), range_258, *[n_259], **kwargs_260)
    
    # Testing if the for loop is going to be iterated (line 73)
    # Testing the type of a for loop iterable (line 73)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 4), range_call_result_261)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 73, 4), range_call_result_261):
        # Getting the type of the for loop variable (line 73)
        for_loop_var_262 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 4), range_call_result_261)
        # Assigning a type to the variable 'i' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'i', for_loop_var_262)
        # SSA begins for a for statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 74):
        
        # Call to matmulttransp(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 74)
        i_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'i', False)
        # Getting the type of 'list1' (line 74)
        list1_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'list1', False)
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 26), list1_265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 74, 26), getitem___266, i_264)
        
        # Getting the type of 'A' (line 74)
        A_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 36), 'A', False)
        # Processing the call keyword arguments (line 74)
        kwargs_269 = {}
        # Getting the type of 'matmulttransp' (line 74)
        matmulttransp_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'matmulttransp', False)
        # Calling matmulttransp(args, kwargs) (line 74)
        matmulttransp_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), matmulttransp_263, *[subscript_call_result_267, A_268], **kwargs_269)
        
        # Assigning a type to the variable 'z' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'z', matmulttransp_call_result_270)
        
        # Call to append(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'z' (line 75)
        z_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'z', False)
        # Processing the call keyword arguments (line 75)
        kwargs_274 = {}
        # Getting the type of 'list1' (line 75)
        list1_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'list1', False)
        # Obtaining the member 'append' of a type (line 75)
        append_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), list1_271, 'append')
        # Calling append(args, kwargs) (line 75)
        append_call_result_275 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), append_272, *[z_273], **kwargs_274)
        
        
        # Assigning a Call to a Name (line 76):
        
        # Call to matmulttransp(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 76)
        i_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'i', False)
        # Getting the type of 'list1' (line 76)
        list1_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'list1', False)
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 26), list1_278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_280 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), getitem___279, i_277)
        
        # Getting the type of 'B' (line 76)
        B_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'B', False)
        # Processing the call keyword arguments (line 76)
        kwargs_282 = {}
        # Getting the type of 'matmulttransp' (line 76)
        matmulttransp_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'matmulttransp', False)
        # Calling matmulttransp(args, kwargs) (line 76)
        matmulttransp_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), matmulttransp_276, *[subscript_call_result_280, B_281], **kwargs_282)
        
        # Assigning a type to the variable 'z' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'z', matmulttransp_call_result_283)
        
        # Call to append(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'z' (line 78)
        z_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'z', False)
        # Processing the call keyword arguments (line 78)
        kwargs_287 = {}
        # Getting the type of 'list1' (line 78)
        list1_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'list1', False)
        # Obtaining the member 'append' of a type (line 78)
        append_285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), list1_284, 'append')
        # Calling append(args, kwargs) (line 78)
        append_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), append_285, *[z_286], **kwargs_287)
        
        
        # Assigning a Call to a Name (line 79):
        
        # Call to matmulttransp(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 79)
        i_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'i', False)
        # Getting the type of 'list1' (line 79)
        list1_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'list1', False)
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 26), list1_291, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 79, 26), getitem___292, i_290)
        
        # Getting the type of 'C' (line 79)
        C_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 36), 'C', False)
        # Processing the call keyword arguments (line 79)
        kwargs_295 = {}
        # Getting the type of 'matmulttransp' (line 79)
        matmulttransp_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'matmulttransp', False)
        # Calling matmulttransp(args, kwargs) (line 79)
        matmulttransp_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), matmulttransp_289, *[subscript_call_result_293, C_294], **kwargs_295)
        
        # Assigning a type to the variable 'z' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'z', matmulttransp_call_result_296)
        
        # Call to append(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'z' (line 81)
        z_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'z', False)
        # Processing the call keyword arguments (line 81)
        kwargs_300 = {}
        # Getting the type of 'list1' (line 81)
        list1_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'list1', False)
        # Obtaining the member 'append' of a type (line 81)
        append_298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), list1_297, 'append')
        # Calling append(args, kwargs) (line 81)
        append_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), append_298, *[z_299], **kwargs_300)
        
        
        # Assigning a Call to a Name (line 82):
        
        # Call to matmulttransp(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 82)
        i_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'i', False)
        # Getting the type of 'list1' (line 82)
        list1_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'list1', False)
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), list1_304, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), getitem___305, i_303)
        
        # Getting the type of 'D' (line 82)
        D_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 36), 'D', False)
        # Processing the call keyword arguments (line 82)
        kwargs_308 = {}
        # Getting the type of 'matmulttransp' (line 82)
        matmulttransp_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'matmulttransp', False)
        # Calling matmulttransp(args, kwargs) (line 82)
        matmulttransp_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), matmulttransp_302, *[subscript_call_result_306, D_307], **kwargs_308)
        
        # Assigning a type to the variable 'z' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'z', matmulttransp_call_result_309)
        
        # Call to append(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'z' (line 84)
        z_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'z', False)
        # Processing the call keyword arguments (line 84)
        kwargs_313 = {}
        # Getting the type of 'list1' (line 84)
        list1_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'list1', False)
        # Obtaining the member 'append' of a type (line 84)
        append_311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), list1_310, 'append')
        # Calling append(args, kwargs) (line 84)
        append_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), append_311, *[z_312], **kwargs_313)
        
        
        # Assigning a Call to a Name (line 85):
        
        # Call to matmulttransp(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 85)
        i_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 32), 'i', False)
        # Getting the type of 'list1' (line 85)
        list1_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'list1', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 26), list1_317, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 85, 26), getitem___318, i_316)
        
        # Getting the type of 'E' (line 85)
        E_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'E', False)
        # Processing the call keyword arguments (line 85)
        kwargs_321 = {}
        # Getting the type of 'matmulttransp' (line 85)
        matmulttransp_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'matmulttransp', False)
        # Calling matmulttransp(args, kwargs) (line 85)
        matmulttransp_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), matmulttransp_315, *[subscript_call_result_319, E_320], **kwargs_321)
        
        # Assigning a type to the variable 'z' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'z', matmulttransp_call_result_322)
        
        # Call to append(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'z' (line 87)
        z_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'z', False)
        # Processing the call keyword arguments (line 87)
        kwargs_326 = {}
        # Getting the type of 'list1' (line 87)
        list1_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'list1', False)
        # Obtaining the member 'append' of a type (line 87)
        append_324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), list1_323, 'append')
        # Calling append(args, kwargs) (line 87)
        append_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), append_324, *[z_325], **kwargs_326)
        
        
        # Assigning a Call to a Name (line 88):
        
        # Call to matmulttransp(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 88)
        i_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'i', False)
        # Getting the type of 'list1' (line 88)
        list1_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'list1', False)
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 26), list1_330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_332 = invoke(stypy.reporting.localization.Localization(__file__, 88, 26), getitem___331, i_329)
        
        # Getting the type of 'F' (line 88)
        F_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'F', False)
        # Processing the call keyword arguments (line 88)
        kwargs_334 = {}
        # Getting the type of 'matmulttransp' (line 88)
        matmulttransp_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'matmulttransp', False)
        # Calling matmulttransp(args, kwargs) (line 88)
        matmulttransp_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), matmulttransp_328, *[subscript_call_result_332, F_333], **kwargs_334)
        
        # Assigning a type to the variable 'z' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'z', matmulttransp_call_result_335)
        
        # Call to append(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'z' (line 90)
        z_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'z', False)
        # Processing the call keyword arguments (line 90)
        kwargs_339 = {}
        # Getting the type of 'list1' (line 90)
        list1_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'list1', False)
        # Obtaining the member 'append' of a type (line 90)
        append_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), list1_336, 'append')
        # Calling append(args, kwargs) (line 90)
        append_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), append_337, *[z_338], **kwargs_339)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to sort(...): (line 92)
    # Processing the call keyword arguments (line 92)
    kwargs_343 = {}
    # Getting the type of 'list1' (line 92)
    list1_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'list1', False)
    # Obtaining the member 'sort' of a type (line 92)
    sort_342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), list1_341, 'sort')
    # Calling sort(args, kwargs) (line 92)
    sort_call_result_344 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), sort_342, *[], **kwargs_343)
    
    
    # Assigning a Call to a Name (line 93):
    
    # Call to len(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'list1' (line 93)
    list1_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'list1', False)
    # Processing the call keyword arguments (line 93)
    kwargs_347 = {}
    # Getting the type of 'len' (line 93)
    len_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'len', False)
    # Calling len(args, kwargs) (line 93)
    len_call_result_348 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), len_345, *[list1_346], **kwargs_347)
    
    # Assigning a type to the variable 'n' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'n', len_call_result_348)
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 17), 'int')
    # Getting the type of 'list1' (line 94)
    list1_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'list1')
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), list1_350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), getitem___351, int_349)
    
    # Assigning a type to the variable 'last' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'last', subscript_call_result_352)
    
    # Multiple assignment of 2 elements.
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'int')
    # Assigning a type to the variable 'i' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'i', int_353)
    # Getting the type of 'i' (line 95)
    i_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'i')
    # Assigning a type to the variable 'lasti' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'lasti', i_354)
    
    
    # Getting the type of 'i' (line 96)
    i_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 10), 'i')
    # Getting the type of 'n' (line 96)
    n_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'n')
    # Applying the binary operator '<' (line 96)
    result_lt_357 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 10), '<', i_355, n_356)
    
    # Testing if the while is going to be iterated (line 96)
    # Testing the type of an if condition (line 96)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_lt_357)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 96, 4), result_lt_357):
        # SSA begins for while statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 97)
        i_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'i')
        # Getting the type of 'list1' (line 97)
        list1_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'list1')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), list1_359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), getitem___360, i_358)
        
        # Getting the type of 'last' (line 97)
        last_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'last')
        # Applying the binary operator '!=' (line 97)
        result_ne_363 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '!=', subscript_call_result_361, last_362)
        
        # Testing if the type of an if condition is none (line 97)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 8), result_ne_363):
            pass
        else:
            
            # Testing the type of an if condition (line 97)
            if_condition_364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_ne_363)
            # Assigning a type to the variable 'if_condition_364' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_364', if_condition_364)
            # SSA begins for if statement (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Multiple assignment of 2 elements.
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 98)
            i_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'i')
            # Getting the type of 'list1' (line 98)
            list1_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'list1')
            # Obtaining the member '__getitem__' of a type (line 98)
            getitem___367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 34), list1_366, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 98)
            subscript_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 98, 34), getitem___367, i_365)
            
            # Assigning a type to the variable 'last' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'last', subscript_call_result_368)
            # Getting the type of 'last' (line 98)
            last_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'last')
            # Getting the type of 'list1' (line 98)
            list1_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'list1')
            # Getting the type of 'lasti' (line 98)
            lasti_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'lasti')
            # Storing an element on a container (line 98)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), list1_370, (lasti_371, last_369))
            
            # Getting the type of 'lasti' (line 99)
            lasti_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'lasti')
            int_373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'int')
            # Applying the binary operator '+=' (line 99)
            result_iadd_374 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 12), '+=', lasti_372, int_373)
            # Assigning a type to the variable 'lasti' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'lasti', result_iadd_374)
            
            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'i' (line 100)
        i_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'i')
        int_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 13), 'int')
        # Applying the binary operator '+=' (line 100)
        result_iadd_377 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 8), '+=', i_375, int_376)
        # Assigning a type to the variable 'i' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'i', result_iadd_377)
        
        # SSA join for while statement (line 96)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to __delslice__(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'lasti' (line 101)
    lasti_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'lasti', False)
    # Getting the type of 'n' (line 101)
    n_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'n', False)
    # Processing the call keyword arguments (line 101)
    kwargs_382 = {}
    # Getting the type of 'list1' (line 101)
    list1_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'list1', False)
    # Obtaining the member '__delslice__' of a type (line 101)
    delslice___379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), list1_378, '__delslice__')
    # Calling __delslice__(args, kwargs) (line 101)
    delslice___call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), delslice___379, *[lasti_380, n_381], **kwargs_382)
    
    
    # ################# End of 'iterate_sort(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterate_sort' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterate_sort'
    return stypy_return_type_384

# Assigning a type to the variable 'iterate_sort' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'iterate_sort', iterate_sort)

@norecursion
def gen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gen'
    module_type_store = module_type_store.open_function_context('gen', 104, 0, False)
    
    # Passed parameters checking function
    gen.stypy_localization = localization
    gen.stypy_type_of_self = None
    gen.stypy_type_store = module_type_store
    gen.stypy_function_name = 'gen'
    gen.stypy_param_names_list = ['n', 'list1', 'A', 'B', 'C', 'D', 'E', 'F']
    gen.stypy_varargs_param_name = None
    gen.stypy_kwargs_param_name = None
    gen.stypy_call_defaults = defaults
    gen.stypy_call_varargs = varargs
    gen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gen', ['n', 'list1', 'A', 'B', 'C', 'D', 'E', 'F'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gen', localization, ['n', 'list1', 'A', 'B', 'C', 'D', 'E', 'F'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gen(...)' code ##################

    
    
    # Call to range(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'n' (line 105)
    n_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'n', False)
    # Processing the call keyword arguments (line 105)
    kwargs_387 = {}
    # Getting the type of 'range' (line 105)
    range_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'range', False)
    # Calling range(args, kwargs) (line 105)
    range_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), range_385, *[n_386], **kwargs_387)
    
    # Testing if the for loop is going to be iterated (line 105)
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_388)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_388):
        # Getting the type of the for loop variable (line 105)
        for_loop_var_389 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_388)
        # Assigning a type to the variable 'i' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'i', for_loop_var_389)
        # SSA begins for a for statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to iterate_sort(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'list1' (line 105)
        list1_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 36), 'list1', False)
        # Getting the type of 'A' (line 105)
        A_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 43), 'A', False)
        # Getting the type of 'B' (line 105)
        B_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'B', False)
        # Getting the type of 'C' (line 105)
        C_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'C', False)
        # Getting the type of 'D' (line 105)
        D_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 52), 'D', False)
        # Getting the type of 'E' (line 105)
        E_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 55), 'E', False)
        # Getting the type of 'F' (line 105)
        F_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 58), 'F', False)
        # Processing the call keyword arguments (line 105)
        kwargs_398 = {}
        # Getting the type of 'iterate_sort' (line 105)
        iterate_sort_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'iterate_sort', False)
        # Calling iterate_sort(args, kwargs) (line 105)
        iterate_sort_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 105, 23), iterate_sort_390, *[list1_391, A_392, B_393, C_394, D_395, E_396, F_397], **kwargs_398)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'gen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gen' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_400)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gen'
    return stypy_return_type_400

# Assigning a type to the variable 'gen' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'gen', gen)

@norecursion
def inward(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'inward'
    module_type_store = module_type_store.open_function_context('inward', 108, 0, False)
    
    # Passed parameters checking function
    inward.stypy_localization = localization
    inward.stypy_type_of_self = None
    inward.stypy_type_store = module_type_store
    inward.stypy_function_name = 'inward'
    inward.stypy_param_names_list = ['U']
    inward.stypy_varargs_param_name = None
    inward.stypy_kwargs_param_name = None
    inward.stypy_call_defaults = defaults
    inward.stypy_call_varargs = varargs
    inward.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inward', ['U'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inward', localization, ['U'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inward(...)' code ##################

    
    # Assigning a BoolOp to a Name (line 109):
    
    # Evaluating a boolean operation
    
    
    # Call to abs(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'int')
    
    # Obtaining the type of the subscript
    int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'int')
    # Getting the type of 'U' (line 109)
    U_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), U_404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_406 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), getitem___405, int_403)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), subscript_call_result_406, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), getitem___407, int_402)
    
    # Processing the call keyword arguments (line 109)
    kwargs_409 = {}
    # Getting the type of 'abs' (line 109)
    abs_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 109)
    abs_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), abs_401, *[subscript_call_result_408], **kwargs_409)
    
    
    # Call to abs(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
    
    # Obtaining the type of the subscript
    int_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 32), 'int')
    # Getting the type of 'U' (line 109)
    U_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 30), U_414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_416 = invoke(stypy.reporting.localization.Localization(__file__, 109, 30), getitem___415, int_413)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 30), subscript_call_result_416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 109, 30), getitem___417, int_412)
    
    # Processing the call keyword arguments (line 109)
    kwargs_419 = {}
    # Getting the type of 'abs' (line 109)
    abs_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'abs', False)
    # Calling abs(args, kwargs) (line 109)
    abs_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 109, 26), abs_411, *[subscript_call_result_418], **kwargs_419)
    
    # Applying the binary operator '<' (line 109)
    result_lt_421 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '<', abs_call_result_410, abs_call_result_420)
    
    
    # Evaluating a boolean operation
    
    
    # Call to abs(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    int_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 54), 'int')
    
    # Obtaining the type of the subscript
    int_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 51), 'int')
    # Getting the type of 'U' (line 109)
    U_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 49), U_425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_427 = invoke(stypy.reporting.localization.Localization(__file__, 109, 49), getitem___426, int_424)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 49), subscript_call_result_427, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 109, 49), getitem___428, int_423)
    
    # Processing the call keyword arguments (line 109)
    kwargs_430 = {}
    # Getting the type of 'abs' (line 109)
    abs_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 45), 'abs', False)
    # Calling abs(args, kwargs) (line 109)
    abs_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 109, 45), abs_422, *[subscript_call_result_429], **kwargs_430)
    
    
    # Call to abs(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    int_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 70), 'int')
    
    # Obtaining the type of the subscript
    int_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 67), 'int')
    # Getting the type of 'U' (line 109)
    U_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 65), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 65), U_435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 109, 65), getitem___436, int_434)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 65), subscript_call_result_437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 109, 65), getitem___438, int_433)
    
    # Processing the call keyword arguments (line 109)
    kwargs_440 = {}
    # Getting the type of 'abs' (line 109)
    abs_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 61), 'abs', False)
    # Calling abs(args, kwargs) (line 109)
    abs_call_result_441 = invoke(stypy.reporting.localization.Localization(__file__, 109, 61), abs_432, *[subscript_call_result_439], **kwargs_440)
    
    # Applying the binary operator '==' (line 109)
    result_eq_442 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 45), '==', abs_call_result_431, abs_call_result_441)
    
    
    
    # Call to abs(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    int_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 87), 'int')
    
    # Obtaining the type of the subscript
    int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 84), 'int')
    # Getting the type of 'U' (line 109)
    U_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 82), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 82), U_446, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 109, 82), getitem___447, int_445)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 82), subscript_call_result_448, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_450 = invoke(stypy.reporting.localization.Localization(__file__, 109, 82), getitem___449, int_444)
    
    # Processing the call keyword arguments (line 109)
    kwargs_451 = {}
    # Getting the type of 'abs' (line 109)
    abs_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 78), 'abs', False)
    # Calling abs(args, kwargs) (line 109)
    abs_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 109, 78), abs_443, *[subscript_call_result_450], **kwargs_451)
    
    
    # Call to abs(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 102), 'int')
    
    # Obtaining the type of the subscript
    int_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 99), 'int')
    # Getting the type of 'U' (line 109)
    U_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 97), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 97), U_456, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 109, 97), getitem___457, int_455)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 97), subscript_call_result_458, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 109, 97), getitem___459, int_454)
    
    # Processing the call keyword arguments (line 109)
    kwargs_461 = {}
    # Getting the type of 'abs' (line 109)
    abs_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 93), 'abs', False)
    # Calling abs(args, kwargs) (line 109)
    abs_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 109, 93), abs_453, *[subscript_call_result_460], **kwargs_461)
    
    # Applying the binary operator '<' (line 109)
    result_lt_463 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 78), '<', abs_call_result_452, abs_call_result_462)
    
    # Applying the binary operator 'and' (line 109)
    result_and_keyword_464 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 45), 'and', result_eq_442, result_lt_463)
    
    # Applying the binary operator 'or' (line 109)
    result_or_keyword_465 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 10), 'or', result_lt_421, result_and_keyword_464)
    
    # Evaluating a boolean operation
    
    
    # Call to abs(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    int_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 14), 'int')
    
    # Obtaining the type of the subscript
    int_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 11), 'int')
    # Getting the type of 'U' (line 110)
    U_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 9), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 9), U_469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 110, 9), getitem___470, int_468)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 9), subscript_call_result_471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 110, 9), getitem___472, int_467)
    
    # Processing the call keyword arguments (line 110)
    kwargs_474 = {}
    # Getting the type of 'abs' (line 110)
    abs_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 5), 'abs', False)
    # Calling abs(args, kwargs) (line 110)
    abs_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 110, 5), abs_466, *[subscript_call_result_473], **kwargs_474)
    
    
    # Call to abs(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    int_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
    
    # Obtaining the type of the subscript
    int_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
    # Getting the type of 'U' (line 110)
    U_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 25), U_479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_481 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), getitem___480, int_478)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 25), subscript_call_result_481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), getitem___482, int_477)
    
    # Processing the call keyword arguments (line 110)
    kwargs_484 = {}
    # Getting the type of 'abs' (line 110)
    abs_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'abs', False)
    # Calling abs(args, kwargs) (line 110)
    abs_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), abs_476, *[subscript_call_result_483], **kwargs_484)
    
    # Applying the binary operator '==' (line 110)
    result_eq_486 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 5), '==', abs_call_result_475, abs_call_result_485)
    
    
    
    # Call to abs(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    int_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 47), 'int')
    
    # Obtaining the type of the subscript
    int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 44), 'int')
    # Getting the type of 'U' (line 110)
    U_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 42), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 42), U_490, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_492 = invoke(stypy.reporting.localization.Localization(__file__, 110, 42), getitem___491, int_489)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 42), subscript_call_result_492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_494 = invoke(stypy.reporting.localization.Localization(__file__, 110, 42), getitem___493, int_488)
    
    # Processing the call keyword arguments (line 110)
    kwargs_495 = {}
    # Getting the type of 'abs' (line 110)
    abs_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 38), 'abs', False)
    # Calling abs(args, kwargs) (line 110)
    abs_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 110, 38), abs_487, *[subscript_call_result_494], **kwargs_495)
    
    
    # Call to abs(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    int_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 63), 'int')
    
    # Obtaining the type of the subscript
    int_499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 60), 'int')
    # Getting the type of 'U' (line 110)
    U_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 58), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 58), U_500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_502 = invoke(stypy.reporting.localization.Localization(__file__, 110, 58), getitem___501, int_499)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 58), subscript_call_result_502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 110, 58), getitem___503, int_498)
    
    # Processing the call keyword arguments (line 110)
    kwargs_505 = {}
    # Getting the type of 'abs' (line 110)
    abs_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 54), 'abs', False)
    # Calling abs(args, kwargs) (line 110)
    abs_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 110, 54), abs_497, *[subscript_call_result_504], **kwargs_505)
    
    # Applying the binary operator '==' (line 110)
    result_eq_507 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 38), '==', abs_call_result_496, abs_call_result_506)
    
    # Applying the binary operator 'and' (line 110)
    result_and_keyword_508 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 5), 'and', result_eq_486, result_eq_507)
    
    
    # Call to abs(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    int_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 80), 'int')
    
    # Obtaining the type of the subscript
    int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 77), 'int')
    # Getting the type of 'U' (line 110)
    U_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 75), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 75), U_512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_514 = invoke(stypy.reporting.localization.Localization(__file__, 110, 75), getitem___513, int_511)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 75), subscript_call_result_514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_516 = invoke(stypy.reporting.localization.Localization(__file__, 110, 75), getitem___515, int_510)
    
    # Processing the call keyword arguments (line 110)
    kwargs_517 = {}
    # Getting the type of 'abs' (line 110)
    abs_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 71), 'abs', False)
    # Calling abs(args, kwargs) (line 110)
    abs_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 110, 71), abs_509, *[subscript_call_result_516], **kwargs_517)
    
    
    # Call to abs(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 95), 'int')
    
    # Obtaining the type of the subscript
    int_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 92), 'int')
    # Getting the type of 'U' (line 110)
    U_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 90), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 90), U_522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 110, 90), getitem___523, int_521)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 90), subscript_call_result_524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_526 = invoke(stypy.reporting.localization.Localization(__file__, 110, 90), getitem___525, int_520)
    
    # Processing the call keyword arguments (line 110)
    kwargs_527 = {}
    # Getting the type of 'abs' (line 110)
    abs_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 86), 'abs', False)
    # Calling abs(args, kwargs) (line 110)
    abs_call_result_528 = invoke(stypy.reporting.localization.Localization(__file__, 110, 86), abs_519, *[subscript_call_result_526], **kwargs_527)
    
    # Applying the binary operator '<' (line 110)
    result_lt_529 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 71), '<', abs_call_result_518, abs_call_result_528)
    
    # Applying the binary operator 'and' (line 110)
    result_and_keyword_530 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 5), 'and', result_and_keyword_508, result_lt_529)
    
    # Applying the binary operator 'or' (line 109)
    result_or_keyword_531 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 10), 'or', result_or_keyword_465, result_and_keyword_530)
    
    # Assigning a type to the variable 'b01' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'b01', result_or_keyword_531)
    
    # Assigning a BoolOp to a Name (line 112):
    
    # Evaluating a boolean operation
    
    
    # Call to abs(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'int')
    
    # Obtaining the type of the subscript
    int_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 17), 'int')
    # Getting the type of 'U' (line 112)
    U_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 15), U_535, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_537 = invoke(stypy.reporting.localization.Localization(__file__, 112, 15), getitem___536, int_534)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 15), subscript_call_result_537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 112, 15), getitem___538, int_533)
    
    # Processing the call keyword arguments (line 112)
    kwargs_540 = {}
    # Getting the type of 'abs' (line 112)
    abs_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 112)
    abs_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), abs_532, *[subscript_call_result_539], **kwargs_540)
    
    
    # Call to abs(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    int_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'int')
    
    # Obtaining the type of the subscript
    int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'int')
    # Getting the type of 'U' (line 112)
    U_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 30), U_545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 112, 30), getitem___546, int_544)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 30), subscript_call_result_547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_549 = invoke(stypy.reporting.localization.Localization(__file__, 112, 30), getitem___548, int_543)
    
    # Processing the call keyword arguments (line 112)
    kwargs_550 = {}
    # Getting the type of 'abs' (line 112)
    abs_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'abs', False)
    # Calling abs(args, kwargs) (line 112)
    abs_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 112, 26), abs_542, *[subscript_call_result_549], **kwargs_550)
    
    # Applying the binary operator '<' (line 112)
    result_lt_552 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), '<', abs_call_result_541, abs_call_result_551)
    
    
    # Evaluating a boolean operation
    
    
    # Call to abs(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    int_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 54), 'int')
    
    # Obtaining the type of the subscript
    int_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 51), 'int')
    # Getting the type of 'U' (line 112)
    U_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 49), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 49), U_556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_558 = invoke(stypy.reporting.localization.Localization(__file__, 112, 49), getitem___557, int_555)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 49), subscript_call_result_558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 112, 49), getitem___559, int_554)
    
    # Processing the call keyword arguments (line 112)
    kwargs_561 = {}
    # Getting the type of 'abs' (line 112)
    abs_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 45), 'abs', False)
    # Calling abs(args, kwargs) (line 112)
    abs_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 112, 45), abs_553, *[subscript_call_result_560], **kwargs_561)
    
    
    # Call to abs(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    int_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 70), 'int')
    
    # Obtaining the type of the subscript
    int_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 67), 'int')
    # Getting the type of 'U' (line 112)
    U_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 65), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 65), U_566, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 112, 65), getitem___567, int_565)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 65), subscript_call_result_568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_570 = invoke(stypy.reporting.localization.Localization(__file__, 112, 65), getitem___569, int_564)
    
    # Processing the call keyword arguments (line 112)
    kwargs_571 = {}
    # Getting the type of 'abs' (line 112)
    abs_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 61), 'abs', False)
    # Calling abs(args, kwargs) (line 112)
    abs_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 112, 61), abs_563, *[subscript_call_result_570], **kwargs_571)
    
    # Applying the binary operator '==' (line 112)
    result_eq_573 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 45), '==', abs_call_result_562, abs_call_result_572)
    
    
    
    # Call to abs(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    int_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 87), 'int')
    
    # Obtaining the type of the subscript
    int_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 84), 'int')
    # Getting the type of 'U' (line 112)
    U_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 82), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 82), U_577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 112, 82), getitem___578, int_576)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 82), subscript_call_result_579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 112, 82), getitem___580, int_575)
    
    # Processing the call keyword arguments (line 112)
    kwargs_582 = {}
    # Getting the type of 'abs' (line 112)
    abs_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 78), 'abs', False)
    # Calling abs(args, kwargs) (line 112)
    abs_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 112, 78), abs_574, *[subscript_call_result_581], **kwargs_582)
    
    
    # Call to abs(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    int_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 102), 'int')
    
    # Obtaining the type of the subscript
    int_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 99), 'int')
    # Getting the type of 'U' (line 112)
    U_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 97), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 97), U_587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_589 = invoke(stypy.reporting.localization.Localization(__file__, 112, 97), getitem___588, int_586)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 97), subscript_call_result_589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 112, 97), getitem___590, int_585)
    
    # Processing the call keyword arguments (line 112)
    kwargs_592 = {}
    # Getting the type of 'abs' (line 112)
    abs_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 93), 'abs', False)
    # Calling abs(args, kwargs) (line 112)
    abs_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 112, 93), abs_584, *[subscript_call_result_591], **kwargs_592)
    
    # Applying the binary operator '<' (line 112)
    result_lt_594 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 78), '<', abs_call_result_583, abs_call_result_593)
    
    # Applying the binary operator 'and' (line 112)
    result_and_keyword_595 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 45), 'and', result_eq_573, result_lt_594)
    
    # Applying the binary operator 'or' (line 112)
    result_or_keyword_596 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 10), 'or', result_lt_552, result_and_keyword_595)
    
    # Evaluating a boolean operation
    
    
    # Call to abs(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    int_598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 14), 'int')
    
    # Obtaining the type of the subscript
    int_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 11), 'int')
    # Getting the type of 'U' (line 113)
    U_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), U_600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), getitem___601, int_599)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), subscript_call_result_602, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), getitem___603, int_598)
    
    # Processing the call keyword arguments (line 113)
    kwargs_605 = {}
    # Getting the type of 'abs' (line 113)
    abs_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 5), 'abs', False)
    # Calling abs(args, kwargs) (line 113)
    abs_call_result_606 = invoke(stypy.reporting.localization.Localization(__file__, 113, 5), abs_597, *[subscript_call_result_604], **kwargs_605)
    
    
    # Call to abs(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    int_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 30), 'int')
    
    # Obtaining the type of the subscript
    int_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'int')
    # Getting the type of 'U' (line 113)
    U_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), U_610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 113, 25), getitem___611, int_609)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), subscript_call_result_612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 113, 25), getitem___613, int_608)
    
    # Processing the call keyword arguments (line 113)
    kwargs_615 = {}
    # Getting the type of 'abs' (line 113)
    abs_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'abs', False)
    # Calling abs(args, kwargs) (line 113)
    abs_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 113, 21), abs_607, *[subscript_call_result_614], **kwargs_615)
    
    # Applying the binary operator '==' (line 113)
    result_eq_617 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 5), '==', abs_call_result_606, abs_call_result_616)
    
    
    
    # Call to abs(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    int_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 47), 'int')
    
    # Obtaining the type of the subscript
    int_620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'int')
    # Getting the type of 'U' (line 113)
    U_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 42), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 42), U_621, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 113, 42), getitem___622, int_620)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 42), subscript_call_result_623, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 113, 42), getitem___624, int_619)
    
    # Processing the call keyword arguments (line 113)
    kwargs_626 = {}
    # Getting the type of 'abs' (line 113)
    abs_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 38), 'abs', False)
    # Calling abs(args, kwargs) (line 113)
    abs_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 113, 38), abs_618, *[subscript_call_result_625], **kwargs_626)
    
    
    # Call to abs(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    int_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 63), 'int')
    
    # Obtaining the type of the subscript
    int_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 60), 'int')
    # Getting the type of 'U' (line 113)
    U_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 58), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 58), U_631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 113, 58), getitem___632, int_630)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 58), subscript_call_result_633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_635 = invoke(stypy.reporting.localization.Localization(__file__, 113, 58), getitem___634, int_629)
    
    # Processing the call keyword arguments (line 113)
    kwargs_636 = {}
    # Getting the type of 'abs' (line 113)
    abs_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 54), 'abs', False)
    # Calling abs(args, kwargs) (line 113)
    abs_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 113, 54), abs_628, *[subscript_call_result_635], **kwargs_636)
    
    # Applying the binary operator '==' (line 113)
    result_eq_638 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 38), '==', abs_call_result_627, abs_call_result_637)
    
    # Applying the binary operator 'and' (line 113)
    result_and_keyword_639 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 5), 'and', result_eq_617, result_eq_638)
    
    
    # Call to abs(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    int_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 80), 'int')
    
    # Obtaining the type of the subscript
    int_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 77), 'int')
    # Getting the type of 'U' (line 113)
    U_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 75), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 75), U_643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_645 = invoke(stypy.reporting.localization.Localization(__file__, 113, 75), getitem___644, int_642)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 75), subscript_call_result_645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 113, 75), getitem___646, int_641)
    
    # Processing the call keyword arguments (line 113)
    kwargs_648 = {}
    # Getting the type of 'abs' (line 113)
    abs_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 71), 'abs', False)
    # Calling abs(args, kwargs) (line 113)
    abs_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 113, 71), abs_640, *[subscript_call_result_647], **kwargs_648)
    
    
    # Call to abs(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    int_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 95), 'int')
    
    # Obtaining the type of the subscript
    int_652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 92), 'int')
    # Getting the type of 'U' (line 113)
    U_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 90), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 90), U_653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 113, 90), getitem___654, int_652)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 90), subscript_call_result_655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_657 = invoke(stypy.reporting.localization.Localization(__file__, 113, 90), getitem___656, int_651)
    
    # Processing the call keyword arguments (line 113)
    kwargs_658 = {}
    # Getting the type of 'abs' (line 113)
    abs_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 86), 'abs', False)
    # Calling abs(args, kwargs) (line 113)
    abs_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 113, 86), abs_650, *[subscript_call_result_657], **kwargs_658)
    
    # Applying the binary operator '<' (line 113)
    result_lt_660 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 71), '<', abs_call_result_649, abs_call_result_659)
    
    # Applying the binary operator 'and' (line 113)
    result_and_keyword_661 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 5), 'and', result_and_keyword_639, result_lt_660)
    
    # Applying the binary operator 'or' (line 112)
    result_or_keyword_662 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 10), 'or', result_or_keyword_596, result_and_keyword_661)
    
    # Assigning a type to the variable 'b12' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'b12', result_or_keyword_662)
    
    # Evaluating a boolean operation
    # Getting the type of 'b01' (line 115)
    b01_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'b01')
    # Getting the type of 'b12' (line 115)
    b12_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'b12')
    # Applying the binary operator 'and' (line 115)
    result_and_keyword_665 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 11), 'and', b01_663, b12_664)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', result_and_keyword_665)
    
    # ################# End of 'inward(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inward' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_666)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inward'
    return stypy_return_type_666

# Assigning a type to the variable 'inward' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'inward', inward)

@norecursion
def examine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'examine'
    module_type_store = module_type_store.open_function_context('examine', 118, 0, False)
    
    # Passed parameters checking function
    examine.stypy_localization = localization
    examine.stypy_type_of_self = None
    examine.stypy_type_store = module_type_store
    examine.stypy_function_name = 'examine'
    examine.stypy_param_names_list = ['U', 'i', 'j']
    examine.stypy_varargs_param_name = None
    examine.stypy_kwargs_param_name = None
    examine.stypy_call_defaults = defaults
    examine.stypy_call_varargs = varargs
    examine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'examine', ['U', 'i', 'j'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'examine', localization, ['U', 'i', 'j'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'examine(...)' code ##################

    
    # Assigning a BinOp to a Name (line 119):
    
    # Call to abs(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'i' (line 119)
    i_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'i', False)
    # Processing the call keyword arguments (line 119)
    kwargs_669 = {}
    # Getting the type of 'abs' (line 119)
    abs_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 119)
    abs_call_result_670 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), abs_667, *[i_668], **kwargs_669)
    
    int_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'int')
    # Applying the binary operator '-' (line 119)
    result_sub_672 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '-', abs_call_result_670, int_671)
    
    # Assigning a type to the variable 'row1' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'row1', result_sub_672)
    
    # Assigning a BinOp to a Name (line 120):
    # Getting the type of 'j' (line 120)
    j_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'j')
    int_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
    # Applying the binary operator '-' (line 120)
    result_sub_675 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), '-', j_673, int_674)
    
    # Assigning a type to the variable 'row2' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'row2', result_sub_675)
    
    # Assigning a Num to a Name (line 121):
    int_676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'int')
    # Assigning a type to the variable 's' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 's', int_676)
    
    # Getting the type of 'i' (line 122)
    i_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 7), 'i')
    int_678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 11), 'int')
    # Applying the binary operator '<' (line 122)
    result_lt_679 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 7), '<', i_677, int_678)
    
    # Testing if the type of an if condition is none (line 122)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 4), result_lt_679):
        pass
    else:
        
        # Testing the type of an if condition (line 122)
        if_condition_680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 4), result_lt_679)
        # Assigning a type to the variable 'if_condition_680' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'if_condition_680', if_condition_680)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 122):
        int_681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 18), 'int')
        # Assigning a type to the variable 's' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 's', int_681)
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 123):
    
    # Call to abs(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row1' (line 123)
    row1_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'row1', False)
    
    # Obtaining the type of the subscript
    int_684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'int')
    # Getting the type of 'U' (line 123)
    U_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 15), U_685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), getitem___686, int_684)
    
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 15), subscript_call_result_687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_689 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), getitem___688, row1_683)
    
    # Getting the type of 's' (line 123)
    s_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 's', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row2' (line 123)
    row2_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'row2', False)
    
    # Obtaining the type of the subscript
    int_692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'int')
    # Getting the type of 'U' (line 123)
    U_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 32), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 32), U_693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 123, 32), getitem___694, int_692)
    
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 32), subscript_call_result_695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 123, 32), getitem___696, row2_691)
    
    # Applying the binary operator '*' (line 123)
    result_mul_698 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 28), '*', s_690, subscript_call_result_697)
    
    # Applying the binary operator '+' (line 123)
    result_add_699 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '+', subscript_call_result_689, result_mul_698)
    
    # Processing the call keyword arguments (line 123)
    kwargs_700 = {}
    # Getting the type of 'abs' (line 123)
    abs_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 123)
    abs_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), abs_682, *[result_add_699], **kwargs_700)
    
    
    # Call to abs(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row2' (line 123)
    row2_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 55), 'row2', False)
    
    # Obtaining the type of the subscript
    int_704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 52), 'int')
    # Getting the type of 'U' (line 123)
    U_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 50), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 50), U_705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 123, 50), getitem___706, int_704)
    
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 50), subscript_call_result_707, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_709 = invoke(stypy.reporting.localization.Localization(__file__, 123, 50), getitem___708, row2_703)
    
    # Processing the call keyword arguments (line 123)
    kwargs_710 = {}
    # Getting the type of 'abs' (line 123)
    abs_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'abs', False)
    # Calling abs(args, kwargs) (line 123)
    abs_call_result_711 = invoke(stypy.reporting.localization.Localization(__file__, 123, 46), abs_702, *[subscript_call_result_709], **kwargs_710)
    
    # Applying the binary operator '-' (line 123)
    result_sub_712 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), '-', abs_call_result_701, abs_call_result_711)
    
    # Assigning a type to the variable 'diff' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'diff', result_sub_712)
    
    # Getting the type of 'diff' (line 124)
    diff_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'diff')
    int_714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'int')
    # Applying the binary operator '<' (line 124)
    result_lt_715 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), '<', diff_713, int_714)
    
    # Testing if the type of an if condition is none (line 124)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 124, 4), result_lt_715):
        pass
    else:
        
        # Testing the type of an if condition (line 124)
        if_condition_716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_lt_715)
        # Assigning a type to the variable 'if_condition_716' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_716', if_condition_716)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'stypy_return_type', int_717)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'diff' (line 125)
    diff_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 7), 'diff')
    int_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 14), 'int')
    # Applying the binary operator '>' (line 125)
    result_gt_720 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 7), '>', diff_718, int_719)
    
    # Testing if the type of an if condition is none (line 125)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 4), result_gt_720):
        
        # Assigning a BinOp to a Name (line 128):
        
        # Call to abs(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 128)
        row1_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'row1', False)
        
        # Obtaining the type of the subscript
        int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 21), 'int')
        # Getting the type of 'U' (line 128)
        U_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), U_726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), getitem___727, int_725)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), subscript_call_result_728, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_730 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), getitem___729, row1_724)
        
        # Getting the type of 's' (line 128)
        s_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 's', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row2' (line 128)
        row2_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'row2', False)
        
        # Obtaining the type of the subscript
        int_733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 38), 'int')
        # Getting the type of 'U' (line 128)
        U_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 36), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 36), U_734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 128, 36), getitem___735, int_733)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 36), subscript_call_result_736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_738 = invoke(stypy.reporting.localization.Localization(__file__, 128, 36), getitem___737, row2_732)
        
        # Applying the binary operator '*' (line 128)
        result_mul_739 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 32), '*', s_731, subscript_call_result_738)
        
        # Applying the binary operator '+' (line 128)
        result_add_740 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 19), '+', subscript_call_result_730, result_mul_739)
        
        # Processing the call keyword arguments (line 128)
        kwargs_741 = {}
        # Getting the type of 'abs' (line 128)
        abs_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 128)
        abs_call_result_742 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), abs_723, *[result_add_740], **kwargs_741)
        
        
        # Call to abs(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row2' (line 128)
        row2_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'row2', False)
        
        # Obtaining the type of the subscript
        int_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 56), 'int')
        # Getting the type of 'U' (line 128)
        U_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 54), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 54), U_746, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 128, 54), getitem___747, int_745)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 54), subscript_call_result_748, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 128, 54), getitem___749, row2_744)
        
        # Processing the call keyword arguments (line 128)
        kwargs_751 = {}
        # Getting the type of 'abs' (line 128)
        abs_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'abs', False)
        # Calling abs(args, kwargs) (line 128)
        abs_call_result_752 = invoke(stypy.reporting.localization.Localization(__file__, 128, 50), abs_743, *[subscript_call_result_750], **kwargs_751)
        
        # Applying the binary operator '-' (line 128)
        result_sub_753 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), '-', abs_call_result_742, abs_call_result_752)
        
        # Assigning a type to the variable 'diff' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'diff', result_sub_753)
        
        # Getting the type of 'diff' (line 129)
        diff_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'diff')
        int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 18), 'int')
        # Applying the binary operator '<' (line 129)
        result_lt_756 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 11), '<', diff_754, int_755)
        
        # Testing if the type of an if condition is none (line 129)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 129, 8), result_lt_756):
            pass
        else:
            
            # Testing the type of an if condition (line 129)
            if_condition_757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), result_lt_756)
            # Assigning a type to the variable 'if_condition_757' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_757', if_condition_757)
            # SSA begins for if statement (line 129)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'stypy_return_type', int_758)
            # SSA join for if statement (line 129)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'diff' (line 130)
        diff_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'diff')
        int_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 18), 'int')
        # Applying the binary operator '>' (line 130)
        result_gt_761 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), '>', diff_759, int_760)
        
        # Testing if the type of an if condition is none (line 130)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 8), result_gt_761):
            
            # Assigning a BinOp to a Name (line 133):
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 133)
            row1_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'row1', False)
            
            # Obtaining the type of the subscript
            int_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
            # Getting the type of 'U' (line 133)
            U_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), U_767, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_769 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___768, int_766)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), subscript_call_result_769, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___770, row1_765)
            
            # Getting the type of 's' (line 133)
            s_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 's', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 45), 'row2', False)
            
            # Obtaining the type of the subscript
            int_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 42), 'int')
            # Getting the type of 'U' (line 133)
            U_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), U_775, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___776, int_774)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), subscript_call_result_777, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___778, row2_773)
            
            # Applying the binary operator '*' (line 133)
            result_mul_780 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 36), '*', s_772, subscript_call_result_779)
            
            # Applying the binary operator '+' (line 133)
            result_add_781 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 23), '+', subscript_call_result_771, result_mul_780)
            
            # Processing the call keyword arguments (line 133)
            kwargs_782 = {}
            # Getting the type of 'abs' (line 133)
            abs_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), abs_764, *[result_add_781], **kwargs_782)
            
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 63), 'row2', False)
            
            # Obtaining the type of the subscript
            int_786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 60), 'int')
            # Getting the type of 'U' (line 133)
            U_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 58), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), U_787, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___788, int_786)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), subscript_call_result_789, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_791 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___790, row2_785)
            
            # Processing the call keyword arguments (line 133)
            kwargs_792 = {}
            # Getting the type of 'abs' (line 133)
            abs_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 54), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 133, 54), abs_784, *[subscript_call_result_791], **kwargs_792)
            
            # Applying the binary operator '-' (line 133)
            result_sub_794 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '-', abs_call_result_783, abs_call_result_793)
            
            # Assigning a type to the variable 'diff' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'diff', result_sub_794)
            
            # Getting the type of 'diff' (line 134)
            diff_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'diff')
            int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'int')
            # Applying the binary operator '<' (line 134)
            result_lt_797 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '<', diff_795, int_796)
            
            # Testing if the type of an if condition is none (line 134)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797):
                pass
            else:
                
                # Testing the type of an if condition (line 134)
                if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797)
                # Assigning a type to the variable 'if_condition_798' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_798', if_condition_798)
                # SSA begins for if statement (line 134)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'stypy_return_type', int_799)
                # SSA join for if statement (line 134)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 135)
            diff_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'diff')
            int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'int')
            # Applying the binary operator '>' (line 135)
            result_gt_802 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '>', diff_800, int_801)
            
            # Testing if the type of an if condition is none (line 135)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802):
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
            else:
                
                # Testing the type of an if condition (line 135)
                if_condition_803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802)
                # Assigning a type to the variable 'if_condition_803' (line 135)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_803', if_condition_803)
                # SSA begins for if statement (line 135)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'stypy_return_type', int_804)
                # SSA branch for the else part of an if statement (line 135)
                module_type_store.open_ssa_branch('else')
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
                # SSA join for if statement (line 135)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 130)
            if_condition_762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_gt_761)
            # Assigning a type to the variable 'if_condition_762' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_762', if_condition_762)
            # SSA begins for if statement (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', int_763)
            # SSA branch for the else part of an if statement (line 130)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 133):
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 133)
            row1_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'row1', False)
            
            # Obtaining the type of the subscript
            int_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
            # Getting the type of 'U' (line 133)
            U_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), U_767, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_769 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___768, int_766)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), subscript_call_result_769, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___770, row1_765)
            
            # Getting the type of 's' (line 133)
            s_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 's', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 45), 'row2', False)
            
            # Obtaining the type of the subscript
            int_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 42), 'int')
            # Getting the type of 'U' (line 133)
            U_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), U_775, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___776, int_774)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), subscript_call_result_777, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___778, row2_773)
            
            # Applying the binary operator '*' (line 133)
            result_mul_780 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 36), '*', s_772, subscript_call_result_779)
            
            # Applying the binary operator '+' (line 133)
            result_add_781 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 23), '+', subscript_call_result_771, result_mul_780)
            
            # Processing the call keyword arguments (line 133)
            kwargs_782 = {}
            # Getting the type of 'abs' (line 133)
            abs_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), abs_764, *[result_add_781], **kwargs_782)
            
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 63), 'row2', False)
            
            # Obtaining the type of the subscript
            int_786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 60), 'int')
            # Getting the type of 'U' (line 133)
            U_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 58), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), U_787, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___788, int_786)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), subscript_call_result_789, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_791 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___790, row2_785)
            
            # Processing the call keyword arguments (line 133)
            kwargs_792 = {}
            # Getting the type of 'abs' (line 133)
            abs_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 54), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 133, 54), abs_784, *[subscript_call_result_791], **kwargs_792)
            
            # Applying the binary operator '-' (line 133)
            result_sub_794 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '-', abs_call_result_783, abs_call_result_793)
            
            # Assigning a type to the variable 'diff' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'diff', result_sub_794)
            
            # Getting the type of 'diff' (line 134)
            diff_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'diff')
            int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'int')
            # Applying the binary operator '<' (line 134)
            result_lt_797 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '<', diff_795, int_796)
            
            # Testing if the type of an if condition is none (line 134)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797):
                pass
            else:
                
                # Testing the type of an if condition (line 134)
                if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797)
                # Assigning a type to the variable 'if_condition_798' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_798', if_condition_798)
                # SSA begins for if statement (line 134)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'stypy_return_type', int_799)
                # SSA join for if statement (line 134)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 135)
            diff_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'diff')
            int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'int')
            # Applying the binary operator '>' (line 135)
            result_gt_802 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '>', diff_800, int_801)
            
            # Testing if the type of an if condition is none (line 135)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802):
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
            else:
                
                # Testing the type of an if condition (line 135)
                if_condition_803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802)
                # Assigning a type to the variable 'if_condition_803' (line 135)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_803', if_condition_803)
                # SSA begins for if statement (line 135)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'stypy_return_type', int_804)
                # SSA branch for the else part of an if statement (line 135)
                module_type_store.open_ssa_branch('else')
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
                # SSA join for if statement (line 135)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 130)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 125)
        if_condition_721 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 4), result_gt_720)
        # Assigning a type to the variable 'if_condition_721' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'if_condition_721', if_condition_721)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', int_722)
        # SSA branch for the else part of an if statement (line 125)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 128):
        
        # Call to abs(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 128)
        row1_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'row1', False)
        
        # Obtaining the type of the subscript
        int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 21), 'int')
        # Getting the type of 'U' (line 128)
        U_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), U_726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), getitem___727, int_725)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), subscript_call_result_728, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_730 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), getitem___729, row1_724)
        
        # Getting the type of 's' (line 128)
        s_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 's', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row2' (line 128)
        row2_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'row2', False)
        
        # Obtaining the type of the subscript
        int_733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 38), 'int')
        # Getting the type of 'U' (line 128)
        U_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 36), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 36), U_734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 128, 36), getitem___735, int_733)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 36), subscript_call_result_736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_738 = invoke(stypy.reporting.localization.Localization(__file__, 128, 36), getitem___737, row2_732)
        
        # Applying the binary operator '*' (line 128)
        result_mul_739 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 32), '*', s_731, subscript_call_result_738)
        
        # Applying the binary operator '+' (line 128)
        result_add_740 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 19), '+', subscript_call_result_730, result_mul_739)
        
        # Processing the call keyword arguments (line 128)
        kwargs_741 = {}
        # Getting the type of 'abs' (line 128)
        abs_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 128)
        abs_call_result_742 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), abs_723, *[result_add_740], **kwargs_741)
        
        
        # Call to abs(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row2' (line 128)
        row2_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'row2', False)
        
        # Obtaining the type of the subscript
        int_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 56), 'int')
        # Getting the type of 'U' (line 128)
        U_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 54), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 54), U_746, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 128, 54), getitem___747, int_745)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 54), subscript_call_result_748, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 128, 54), getitem___749, row2_744)
        
        # Processing the call keyword arguments (line 128)
        kwargs_751 = {}
        # Getting the type of 'abs' (line 128)
        abs_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'abs', False)
        # Calling abs(args, kwargs) (line 128)
        abs_call_result_752 = invoke(stypy.reporting.localization.Localization(__file__, 128, 50), abs_743, *[subscript_call_result_750], **kwargs_751)
        
        # Applying the binary operator '-' (line 128)
        result_sub_753 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), '-', abs_call_result_742, abs_call_result_752)
        
        # Assigning a type to the variable 'diff' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'diff', result_sub_753)
        
        # Getting the type of 'diff' (line 129)
        diff_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'diff')
        int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 18), 'int')
        # Applying the binary operator '<' (line 129)
        result_lt_756 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 11), '<', diff_754, int_755)
        
        # Testing if the type of an if condition is none (line 129)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 129, 8), result_lt_756):
            pass
        else:
            
            # Testing the type of an if condition (line 129)
            if_condition_757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), result_lt_756)
            # Assigning a type to the variable 'if_condition_757' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_757', if_condition_757)
            # SSA begins for if statement (line 129)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'stypy_return_type', int_758)
            # SSA join for if statement (line 129)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'diff' (line 130)
        diff_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'diff')
        int_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 18), 'int')
        # Applying the binary operator '>' (line 130)
        result_gt_761 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), '>', diff_759, int_760)
        
        # Testing if the type of an if condition is none (line 130)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 8), result_gt_761):
            
            # Assigning a BinOp to a Name (line 133):
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 133)
            row1_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'row1', False)
            
            # Obtaining the type of the subscript
            int_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
            # Getting the type of 'U' (line 133)
            U_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), U_767, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_769 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___768, int_766)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), subscript_call_result_769, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___770, row1_765)
            
            # Getting the type of 's' (line 133)
            s_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 's', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 45), 'row2', False)
            
            # Obtaining the type of the subscript
            int_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 42), 'int')
            # Getting the type of 'U' (line 133)
            U_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), U_775, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___776, int_774)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), subscript_call_result_777, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___778, row2_773)
            
            # Applying the binary operator '*' (line 133)
            result_mul_780 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 36), '*', s_772, subscript_call_result_779)
            
            # Applying the binary operator '+' (line 133)
            result_add_781 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 23), '+', subscript_call_result_771, result_mul_780)
            
            # Processing the call keyword arguments (line 133)
            kwargs_782 = {}
            # Getting the type of 'abs' (line 133)
            abs_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), abs_764, *[result_add_781], **kwargs_782)
            
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 63), 'row2', False)
            
            # Obtaining the type of the subscript
            int_786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 60), 'int')
            # Getting the type of 'U' (line 133)
            U_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 58), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), U_787, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___788, int_786)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), subscript_call_result_789, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_791 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___790, row2_785)
            
            # Processing the call keyword arguments (line 133)
            kwargs_792 = {}
            # Getting the type of 'abs' (line 133)
            abs_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 54), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 133, 54), abs_784, *[subscript_call_result_791], **kwargs_792)
            
            # Applying the binary operator '-' (line 133)
            result_sub_794 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '-', abs_call_result_783, abs_call_result_793)
            
            # Assigning a type to the variable 'diff' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'diff', result_sub_794)
            
            # Getting the type of 'diff' (line 134)
            diff_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'diff')
            int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'int')
            # Applying the binary operator '<' (line 134)
            result_lt_797 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '<', diff_795, int_796)
            
            # Testing if the type of an if condition is none (line 134)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797):
                pass
            else:
                
                # Testing the type of an if condition (line 134)
                if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797)
                # Assigning a type to the variable 'if_condition_798' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_798', if_condition_798)
                # SSA begins for if statement (line 134)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'stypy_return_type', int_799)
                # SSA join for if statement (line 134)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 135)
            diff_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'diff')
            int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'int')
            # Applying the binary operator '>' (line 135)
            result_gt_802 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '>', diff_800, int_801)
            
            # Testing if the type of an if condition is none (line 135)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802):
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
            else:
                
                # Testing the type of an if condition (line 135)
                if_condition_803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802)
                # Assigning a type to the variable 'if_condition_803' (line 135)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_803', if_condition_803)
                # SSA begins for if statement (line 135)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'stypy_return_type', int_804)
                # SSA branch for the else part of an if statement (line 135)
                module_type_store.open_ssa_branch('else')
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
                # SSA join for if statement (line 135)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 130)
            if_condition_762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_gt_761)
            # Assigning a type to the variable 'if_condition_762' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_762', if_condition_762)
            # SSA begins for if statement (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', int_763)
            # SSA branch for the else part of an if statement (line 130)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 133):
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 133)
            row1_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'row1', False)
            
            # Obtaining the type of the subscript
            int_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
            # Getting the type of 'U' (line 133)
            U_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), U_767, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_769 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___768, int_766)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), subscript_call_result_769, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), getitem___770, row1_765)
            
            # Getting the type of 's' (line 133)
            s_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 's', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 45), 'row2', False)
            
            # Obtaining the type of the subscript
            int_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 42), 'int')
            # Getting the type of 'U' (line 133)
            U_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), U_775, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___776, int_774)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), subscript_call_result_777, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___778, row2_773)
            
            # Applying the binary operator '*' (line 133)
            result_mul_780 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 36), '*', s_772, subscript_call_result_779)
            
            # Applying the binary operator '+' (line 133)
            result_add_781 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 23), '+', subscript_call_result_771, result_mul_780)
            
            # Processing the call keyword arguments (line 133)
            kwargs_782 = {}
            # Getting the type of 'abs' (line 133)
            abs_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), abs_764, *[result_add_781], **kwargs_782)
            
            
            # Call to abs(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 133)
            row2_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 63), 'row2', False)
            
            # Obtaining the type of the subscript
            int_786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 60), 'int')
            # Getting the type of 'U' (line 133)
            U_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 58), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), U_787, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___788, int_786)
            
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), subscript_call_result_789, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_791 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___790, row2_785)
            
            # Processing the call keyword arguments (line 133)
            kwargs_792 = {}
            # Getting the type of 'abs' (line 133)
            abs_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 54), 'abs', False)
            # Calling abs(args, kwargs) (line 133)
            abs_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 133, 54), abs_784, *[subscript_call_result_791], **kwargs_792)
            
            # Applying the binary operator '-' (line 133)
            result_sub_794 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '-', abs_call_result_783, abs_call_result_793)
            
            # Assigning a type to the variable 'diff' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'diff', result_sub_794)
            
            # Getting the type of 'diff' (line 134)
            diff_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'diff')
            int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'int')
            # Applying the binary operator '<' (line 134)
            result_lt_797 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '<', diff_795, int_796)
            
            # Testing if the type of an if condition is none (line 134)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797):
                pass
            else:
                
                # Testing the type of an if condition (line 134)
                if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_lt_797)
                # Assigning a type to the variable 'if_condition_798' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_798', if_condition_798)
                # SSA begins for if statement (line 134)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'stypy_return_type', int_799)
                # SSA join for if statement (line 134)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 135)
            diff_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'diff')
            int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'int')
            # Applying the binary operator '>' (line 135)
            result_gt_802 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '>', diff_800, int_801)
            
            # Testing if the type of an if condition is none (line 135)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802):
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
            else:
                
                # Testing the type of an if condition (line 135)
                if_condition_803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_802)
                # Assigning a type to the variable 'if_condition_803' (line 135)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_803', if_condition_803)
                # SSA begins for if statement (line 135)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'stypy_return_type', int_804)
                # SSA branch for the else part of an if statement (line 135)
                module_type_store.open_ssa_branch('else')
                int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', int_805)
                # SSA join for if statement (line 135)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 130)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'examine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'examine' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_806)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'examine'
    return stypy_return_type_806

# Assigning a type to the variable 'examine' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'examine', examine)

@norecursion
def examine3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'examine3'
    module_type_store = module_type_store.open_function_context('examine3', 141, 0, False)
    
    # Passed parameters checking function
    examine3.stypy_localization = localization
    examine3.stypy_type_of_self = None
    examine3.stypy_type_store = module_type_store
    examine3.stypy_function_name = 'examine3'
    examine3.stypy_param_names_list = ['U', 'i', 'j', 'k']
    examine3.stypy_varargs_param_name = None
    examine3.stypy_kwargs_param_name = None
    examine3.stypy_call_defaults = defaults
    examine3.stypy_call_varargs = varargs
    examine3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'examine3', ['U', 'i', 'j', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'examine3', localization, ['U', 'i', 'j', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'examine3(...)' code ##################

    
    # Assigning a BinOp to a Name (line 142):
    
    # Call to abs(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'i' (line 142)
    i_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'i', False)
    # Processing the call keyword arguments (line 142)
    kwargs_809 = {}
    # Getting the type of 'abs' (line 142)
    abs_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 142)
    abs_call_result_810 = invoke(stypy.reporting.localization.Localization(__file__, 142, 11), abs_807, *[i_808], **kwargs_809)
    
    int_811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'int')
    # Applying the binary operator '-' (line 142)
    result_sub_812 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), '-', abs_call_result_810, int_811)
    
    # Assigning a type to the variable 'row1' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'row1', result_sub_812)
    
    # Assigning a BinOp to a Name (line 143):
    
    # Call to abs(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'j' (line 143)
    j_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'j', False)
    # Processing the call keyword arguments (line 143)
    kwargs_815 = {}
    # Getting the type of 'abs' (line 143)
    abs_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 143)
    abs_call_result_816 = invoke(stypy.reporting.localization.Localization(__file__, 143, 11), abs_813, *[j_814], **kwargs_815)
    
    int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'int')
    # Applying the binary operator '-' (line 143)
    result_sub_818 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 11), '-', abs_call_result_816, int_817)
    
    # Assigning a type to the variable 'row2' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'row2', result_sub_818)
    
    # Assigning a BinOp to a Name (line 144):
    # Getting the type of 'k' (line 144)
    k_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'k')
    int_820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'int')
    # Applying the binary operator '-' (line 144)
    result_sub_821 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 11), '-', k_819, int_820)
    
    # Assigning a type to the variable 'row3' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'row3', result_sub_821)
    
    # Assigning a Num to a Name (line 145):
    int_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 9), 'int')
    # Assigning a type to the variable 's1' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 's1', int_822)
    
    # Assigning a Num to a Name (line 146):
    int_823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 9), 'int')
    # Assigning a type to the variable 's2' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 's2', int_823)
    
    # Getting the type of 'i' (line 147)
    i_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'i')
    int_825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 11), 'int')
    # Applying the binary operator '<' (line 147)
    result_lt_826 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 7), '<', i_824, int_825)
    
    # Testing if the type of an if condition is none (line 147)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 147, 4), result_lt_826):
        pass
    else:
        
        # Testing the type of an if condition (line 147)
        if_condition_827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), result_lt_826)
        # Assigning a type to the variable 'if_condition_827' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_827', if_condition_827)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 147):
        int_828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 19), 'int')
        # Assigning a type to the variable 's1' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 's1', int_828)
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'j' (line 148)
    j_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'j')
    int_830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 11), 'int')
    # Applying the binary operator '<' (line 148)
    result_lt_831 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 7), '<', j_829, int_830)
    
    # Testing if the type of an if condition is none (line 148)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 4), result_lt_831):
        pass
    else:
        
        # Testing the type of an if condition (line 148)
        if_condition_832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), result_lt_831)
        # Assigning a type to the variable 'if_condition_832' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_832', if_condition_832)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 148):
        int_833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 19), 'int')
        # Assigning a type to the variable 's2' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 's2', int_833)
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 149):
    
    # Call to abs(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 's1' (line 149)
    s1_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 's1', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row1' (line 149)
    row1_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'row1', False)
    
    # Obtaining the type of the subscript
    int_837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 22), 'int')
    # Getting the type of 'U' (line 149)
    U_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), U_838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_840 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), getitem___839, int_837)
    
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), subscript_call_result_840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_842 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), getitem___841, row1_836)
    
    # Applying the binary operator '*' (line 149)
    result_mul_843 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), '*', s1_835, subscript_call_result_842)
    
    # Getting the type of 's2' (line 149)
    s2_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 's2', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row2' (line 149)
    row2_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 43), 'row2', False)
    
    # Obtaining the type of the subscript
    int_846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 40), 'int')
    # Getting the type of 'U' (line 149)
    U_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 38), U_847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_849 = invoke(stypy.reporting.localization.Localization(__file__, 149, 38), getitem___848, int_846)
    
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 38), subscript_call_result_849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_851 = invoke(stypy.reporting.localization.Localization(__file__, 149, 38), getitem___850, row2_845)
    
    # Applying the binary operator '*' (line 149)
    result_mul_852 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 33), '*', s2_844, subscript_call_result_851)
    
    # Applying the binary operator '+' (line 149)
    result_add_853 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), '+', result_mul_843, result_mul_852)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'row3' (line 149)
    row3_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 56), 'row3', False)
    
    # Obtaining the type of the subscript
    int_855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 53), 'int')
    # Getting the type of 'U' (line 149)
    U_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 51), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 51), U_856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_858 = invoke(stypy.reporting.localization.Localization(__file__, 149, 51), getitem___857, int_855)
    
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 51), subscript_call_result_858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_860 = invoke(stypy.reporting.localization.Localization(__file__, 149, 51), getitem___859, row3_854)
    
    # Applying the binary operator '+' (line 149)
    result_add_861 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 49), '+', result_add_853, subscript_call_result_860)
    
    # Processing the call keyword arguments (line 149)
    kwargs_862 = {}
    # Getting the type of 'abs' (line 149)
    abs_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 149)
    abs_call_result_863 = invoke(stypy.reporting.localization.Localization(__file__, 149, 11), abs_834, *[result_add_861], **kwargs_862)
    
    
    # Call to abs(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row3' (line 149)
    row3_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 74), 'row3', False)
    
    # Obtaining the type of the subscript
    int_866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 71), 'int')
    # Getting the type of 'U' (line 149)
    U_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 69), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 69), U_867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_869 = invoke(stypy.reporting.localization.Localization(__file__, 149, 69), getitem___868, int_866)
    
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 69), subscript_call_result_869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_871 = invoke(stypy.reporting.localization.Localization(__file__, 149, 69), getitem___870, row3_865)
    
    # Processing the call keyword arguments (line 149)
    kwargs_872 = {}
    # Getting the type of 'abs' (line 149)
    abs_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 65), 'abs', False)
    # Calling abs(args, kwargs) (line 149)
    abs_call_result_873 = invoke(stypy.reporting.localization.Localization(__file__, 149, 65), abs_864, *[subscript_call_result_871], **kwargs_872)
    
    # Applying the binary operator '-' (line 149)
    result_sub_874 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '-', abs_call_result_863, abs_call_result_873)
    
    # Assigning a type to the variable 'diff' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'diff', result_sub_874)
    
    # Getting the type of 'diff' (line 150)
    diff_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'diff')
    int_876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 14), 'int')
    # Applying the binary operator '<' (line 150)
    result_lt_877 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), '<', diff_875, int_876)
    
    # Testing if the type of an if condition is none (line 150)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 4), result_lt_877):
        pass
    else:
        
        # Testing the type of an if condition (line 150)
        if_condition_878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 4), result_lt_877)
        # Assigning a type to the variable 'if_condition_878' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'if_condition_878', if_condition_878)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 24), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'stypy_return_type', int_879)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'diff' (line 151)
    diff_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'diff')
    int_881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 14), 'int')
    # Applying the binary operator '>' (line 151)
    result_gt_882 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 7), '>', diff_880, int_881)
    
    # Testing if the type of an if condition is none (line 151)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 4), result_gt_882):
        
        # Assigning a BinOp to a Name (line 154):
        
        # Call to abs(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 's1' (line 154)
        s1_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 's1', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 154)
        row1_887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'row1', False)
        
        # Obtaining the type of the subscript
        int_888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 26), 'int')
        # Getting the type of 'U' (line 154)
        U_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 24), U_889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_891 = invoke(stypy.reporting.localization.Localization(__file__, 154, 24), getitem___890, int_888)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 24), subscript_call_result_891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_893 = invoke(stypy.reporting.localization.Localization(__file__, 154, 24), getitem___892, row1_887)
        
        # Applying the binary operator '*' (line 154)
        result_mul_894 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '*', s1_886, subscript_call_result_893)
        
        # Getting the type of 's2' (line 154)
        s2_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 's2', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row2' (line 154)
        row2_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 47), 'row2', False)
        
        # Obtaining the type of the subscript
        int_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 44), 'int')
        # Getting the type of 'U' (line 154)
        U_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), U_898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_900 = invoke(stypy.reporting.localization.Localization(__file__, 154, 42), getitem___899, int_897)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), subscript_call_result_900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_902 = invoke(stypy.reporting.localization.Localization(__file__, 154, 42), getitem___901, row2_896)
        
        # Applying the binary operator '*' (line 154)
        result_mul_903 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 37), '*', s2_895, subscript_call_result_902)
        
        # Applying the binary operator '+' (line 154)
        result_add_904 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '+', result_mul_894, result_mul_903)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'row3' (line 154)
        row3_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 60), 'row3', False)
        
        # Obtaining the type of the subscript
        int_906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 57), 'int')
        # Getting the type of 'U' (line 154)
        U_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 55), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 55), U_907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 154, 55), getitem___908, int_906)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 55), subscript_call_result_909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_911 = invoke(stypy.reporting.localization.Localization(__file__, 154, 55), getitem___910, row3_905)
        
        # Applying the binary operator '+' (line 154)
        result_add_912 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 53), '+', result_add_904, subscript_call_result_911)
        
        # Processing the call keyword arguments (line 154)
        kwargs_913 = {}
        # Getting the type of 'abs' (line 154)
        abs_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 154)
        abs_call_result_914 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), abs_885, *[result_add_912], **kwargs_913)
        
        
        # Call to abs(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row3' (line 154)
        row3_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 78), 'row3', False)
        
        # Obtaining the type of the subscript
        int_917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 75), 'int')
        # Getting the type of 'U' (line 154)
        U_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 73), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 73), U_918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 154, 73), getitem___919, int_917)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 73), subscript_call_result_920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_922 = invoke(stypy.reporting.localization.Localization(__file__, 154, 73), getitem___921, row3_916)
        
        # Processing the call keyword arguments (line 154)
        kwargs_923 = {}
        # Getting the type of 'abs' (line 154)
        abs_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 69), 'abs', False)
        # Calling abs(args, kwargs) (line 154)
        abs_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 154, 69), abs_915, *[subscript_call_result_922], **kwargs_923)
        
        # Applying the binary operator '-' (line 154)
        result_sub_925 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '-', abs_call_result_914, abs_call_result_924)
        
        # Assigning a type to the variable 'diff' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'diff', result_sub_925)
        
        # Getting the type of 'diff' (line 155)
        diff_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'diff')
        int_927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'int')
        # Applying the binary operator '<' (line 155)
        result_lt_928 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), '<', diff_926, int_927)
        
        # Testing if the type of an if condition is none (line 155)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 155, 8), result_lt_928):
            pass
        else:
            
            # Testing the type of an if condition (line 155)
            if_condition_929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_lt_928)
            # Assigning a type to the variable 'if_condition_929' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_929', if_condition_929)
            # SSA begins for if statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 28), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'stypy_return_type', int_930)
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'diff' (line 156)
        diff_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'diff')
        int_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 18), 'int')
        # Applying the binary operator '>' (line 156)
        result_gt_933 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), '>', diff_931, int_932)
        
        # Testing if the type of an if condition is none (line 156)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 156, 8), result_gt_933):
            
            # Assigning a BinOp to a Name (line 159):
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 's1' (line 159)
            s1_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 's1', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 159)
            row1_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'row1', False)
            
            # Obtaining the type of the subscript
            int_939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'int')
            # Getting the type of 'U' (line 159)
            U_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), U_940, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___941, int_939)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), subscript_call_result_942, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_944 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___943, row1_938)
            
            # Applying the binary operator '*' (line 159)
            result_mul_945 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '*', s1_937, subscript_call_result_944)
            
            # Getting the type of 's2' (line 159)
            s2_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 's2', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 159)
            row2_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'row2', False)
            
            # Obtaining the type of the subscript
            int_948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 48), 'int')
            # Getting the type of 'U' (line 159)
            U_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), U_949, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_951 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___950, int_948)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), subscript_call_result_951, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___952, row2_947)
            
            # Applying the binary operator '*' (line 159)
            result_mul_954 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 41), '*', s2_946, subscript_call_result_953)
            
            # Applying the binary operator '+' (line 159)
            result_add_955 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '+', result_mul_945, result_mul_954)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 64), 'row3', False)
            
            # Obtaining the type of the subscript
            int_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 61), 'int')
            # Getting the type of 'U' (line 159)
            U_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), U_958, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_960 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___959, int_957)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), subscript_call_result_960, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_962 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___961, row3_956)
            
            # Applying the binary operator '+' (line 159)
            result_add_963 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 57), '+', result_add_955, subscript_call_result_962)
            
            # Processing the call keyword arguments (line 159)
            kwargs_964 = {}
            # Getting the type of 'abs' (line 159)
            abs_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), abs_936, *[result_add_963], **kwargs_964)
            
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 82), 'row3', False)
            
            # Obtaining the type of the subscript
            int_968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 79), 'int')
            # Getting the type of 'U' (line 159)
            U_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 77), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), U_969, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___970, int_968)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), subscript_call_result_971, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_973 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___972, row3_967)
            
            # Processing the call keyword arguments (line 159)
            kwargs_974 = {}
            # Getting the type of 'abs' (line 159)
            abs_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 73), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_975 = invoke(stypy.reporting.localization.Localization(__file__, 159, 73), abs_966, *[subscript_call_result_973], **kwargs_974)
            
            # Applying the binary operator '-' (line 159)
            result_sub_976 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), '-', abs_call_result_965, abs_call_result_975)
            
            # Assigning a type to the variable 'diff' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'diff', result_sub_976)
            
            # Getting the type of 'diff' (line 160)
            diff_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'diff')
            int_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 22), 'int')
            # Applying the binary operator '<' (line 160)
            result_lt_979 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 15), '<', diff_977, int_978)
            
            # Testing if the type of an if condition is none (line 160)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979):
                pass
            else:
                
                # Testing the type of an if condition (line 160)
                if_condition_980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979)
                # Assigning a type to the variable 'if_condition_980' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'if_condition_980', if_condition_980)
                # SSA begins for if statement (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'stypy_return_type', int_981)
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 161)
            diff_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'diff')
            int_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 22), 'int')
            # Applying the binary operator '>' (line 161)
            result_gt_984 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '>', diff_982, int_983)
            
            # Testing if the type of an if condition is none (line 161)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984):
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
            else:
                
                # Testing the type of an if condition (line 161)
                if_condition_985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984)
                # Assigning a type to the variable 'if_condition_985' (line 161)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_985', if_condition_985)
                # SSA begins for if statement (line 161)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 162)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'stypy_return_type', int_986)
                # SSA branch for the else part of an if statement (line 161)
                module_type_store.open_ssa_branch('else')
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
                # SSA join for if statement (line 161)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 156)
            if_condition_934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_gt_933)
            # Assigning a type to the variable 'if_condition_934' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_934', if_condition_934)
            # SSA begins for if statement (line 156)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'stypy_return_type', int_935)
            # SSA branch for the else part of an if statement (line 156)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 159):
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 's1' (line 159)
            s1_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 's1', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 159)
            row1_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'row1', False)
            
            # Obtaining the type of the subscript
            int_939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'int')
            # Getting the type of 'U' (line 159)
            U_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), U_940, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___941, int_939)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), subscript_call_result_942, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_944 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___943, row1_938)
            
            # Applying the binary operator '*' (line 159)
            result_mul_945 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '*', s1_937, subscript_call_result_944)
            
            # Getting the type of 's2' (line 159)
            s2_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 's2', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 159)
            row2_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'row2', False)
            
            # Obtaining the type of the subscript
            int_948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 48), 'int')
            # Getting the type of 'U' (line 159)
            U_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), U_949, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_951 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___950, int_948)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), subscript_call_result_951, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___952, row2_947)
            
            # Applying the binary operator '*' (line 159)
            result_mul_954 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 41), '*', s2_946, subscript_call_result_953)
            
            # Applying the binary operator '+' (line 159)
            result_add_955 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '+', result_mul_945, result_mul_954)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 64), 'row3', False)
            
            # Obtaining the type of the subscript
            int_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 61), 'int')
            # Getting the type of 'U' (line 159)
            U_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), U_958, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_960 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___959, int_957)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), subscript_call_result_960, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_962 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___961, row3_956)
            
            # Applying the binary operator '+' (line 159)
            result_add_963 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 57), '+', result_add_955, subscript_call_result_962)
            
            # Processing the call keyword arguments (line 159)
            kwargs_964 = {}
            # Getting the type of 'abs' (line 159)
            abs_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), abs_936, *[result_add_963], **kwargs_964)
            
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 82), 'row3', False)
            
            # Obtaining the type of the subscript
            int_968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 79), 'int')
            # Getting the type of 'U' (line 159)
            U_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 77), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), U_969, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___970, int_968)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), subscript_call_result_971, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_973 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___972, row3_967)
            
            # Processing the call keyword arguments (line 159)
            kwargs_974 = {}
            # Getting the type of 'abs' (line 159)
            abs_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 73), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_975 = invoke(stypy.reporting.localization.Localization(__file__, 159, 73), abs_966, *[subscript_call_result_973], **kwargs_974)
            
            # Applying the binary operator '-' (line 159)
            result_sub_976 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), '-', abs_call_result_965, abs_call_result_975)
            
            # Assigning a type to the variable 'diff' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'diff', result_sub_976)
            
            # Getting the type of 'diff' (line 160)
            diff_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'diff')
            int_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 22), 'int')
            # Applying the binary operator '<' (line 160)
            result_lt_979 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 15), '<', diff_977, int_978)
            
            # Testing if the type of an if condition is none (line 160)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979):
                pass
            else:
                
                # Testing the type of an if condition (line 160)
                if_condition_980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979)
                # Assigning a type to the variable 'if_condition_980' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'if_condition_980', if_condition_980)
                # SSA begins for if statement (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'stypy_return_type', int_981)
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 161)
            diff_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'diff')
            int_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 22), 'int')
            # Applying the binary operator '>' (line 161)
            result_gt_984 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '>', diff_982, int_983)
            
            # Testing if the type of an if condition is none (line 161)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984):
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
            else:
                
                # Testing the type of an if condition (line 161)
                if_condition_985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984)
                # Assigning a type to the variable 'if_condition_985' (line 161)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_985', if_condition_985)
                # SSA begins for if statement (line 161)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 162)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'stypy_return_type', int_986)
                # SSA branch for the else part of an if statement (line 161)
                module_type_store.open_ssa_branch('else')
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
                # SSA join for if statement (line 161)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 156)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 151)
        if_condition_883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_gt_882)
        # Assigning a type to the variable 'if_condition_883' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_883', if_condition_883)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', int_884)
        # SSA branch for the else part of an if statement (line 151)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 154):
        
        # Call to abs(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 's1' (line 154)
        s1_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 's1', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 154)
        row1_887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'row1', False)
        
        # Obtaining the type of the subscript
        int_888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 26), 'int')
        # Getting the type of 'U' (line 154)
        U_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 24), U_889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_891 = invoke(stypy.reporting.localization.Localization(__file__, 154, 24), getitem___890, int_888)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 24), subscript_call_result_891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_893 = invoke(stypy.reporting.localization.Localization(__file__, 154, 24), getitem___892, row1_887)
        
        # Applying the binary operator '*' (line 154)
        result_mul_894 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '*', s1_886, subscript_call_result_893)
        
        # Getting the type of 's2' (line 154)
        s2_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 's2', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row2' (line 154)
        row2_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 47), 'row2', False)
        
        # Obtaining the type of the subscript
        int_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 44), 'int')
        # Getting the type of 'U' (line 154)
        U_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), U_898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_900 = invoke(stypy.reporting.localization.Localization(__file__, 154, 42), getitem___899, int_897)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), subscript_call_result_900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_902 = invoke(stypy.reporting.localization.Localization(__file__, 154, 42), getitem___901, row2_896)
        
        # Applying the binary operator '*' (line 154)
        result_mul_903 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 37), '*', s2_895, subscript_call_result_902)
        
        # Applying the binary operator '+' (line 154)
        result_add_904 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '+', result_mul_894, result_mul_903)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'row3' (line 154)
        row3_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 60), 'row3', False)
        
        # Obtaining the type of the subscript
        int_906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 57), 'int')
        # Getting the type of 'U' (line 154)
        U_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 55), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 55), U_907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 154, 55), getitem___908, int_906)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 55), subscript_call_result_909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_911 = invoke(stypy.reporting.localization.Localization(__file__, 154, 55), getitem___910, row3_905)
        
        # Applying the binary operator '+' (line 154)
        result_add_912 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 53), '+', result_add_904, subscript_call_result_911)
        
        # Processing the call keyword arguments (line 154)
        kwargs_913 = {}
        # Getting the type of 'abs' (line 154)
        abs_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 154)
        abs_call_result_914 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), abs_885, *[result_add_912], **kwargs_913)
        
        
        # Call to abs(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row3' (line 154)
        row3_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 78), 'row3', False)
        
        # Obtaining the type of the subscript
        int_917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 75), 'int')
        # Getting the type of 'U' (line 154)
        U_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 73), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 73), U_918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 154, 73), getitem___919, int_917)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 73), subscript_call_result_920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_922 = invoke(stypy.reporting.localization.Localization(__file__, 154, 73), getitem___921, row3_916)
        
        # Processing the call keyword arguments (line 154)
        kwargs_923 = {}
        # Getting the type of 'abs' (line 154)
        abs_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 69), 'abs', False)
        # Calling abs(args, kwargs) (line 154)
        abs_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 154, 69), abs_915, *[subscript_call_result_922], **kwargs_923)
        
        # Applying the binary operator '-' (line 154)
        result_sub_925 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '-', abs_call_result_914, abs_call_result_924)
        
        # Assigning a type to the variable 'diff' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'diff', result_sub_925)
        
        # Getting the type of 'diff' (line 155)
        diff_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'diff')
        int_927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'int')
        # Applying the binary operator '<' (line 155)
        result_lt_928 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), '<', diff_926, int_927)
        
        # Testing if the type of an if condition is none (line 155)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 155, 8), result_lt_928):
            pass
        else:
            
            # Testing the type of an if condition (line 155)
            if_condition_929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_lt_928)
            # Assigning a type to the variable 'if_condition_929' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_929', if_condition_929)
            # SSA begins for if statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 28), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'stypy_return_type', int_930)
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'diff' (line 156)
        diff_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'diff')
        int_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 18), 'int')
        # Applying the binary operator '>' (line 156)
        result_gt_933 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), '>', diff_931, int_932)
        
        # Testing if the type of an if condition is none (line 156)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 156, 8), result_gt_933):
            
            # Assigning a BinOp to a Name (line 159):
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 's1' (line 159)
            s1_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 's1', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 159)
            row1_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'row1', False)
            
            # Obtaining the type of the subscript
            int_939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'int')
            # Getting the type of 'U' (line 159)
            U_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), U_940, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___941, int_939)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), subscript_call_result_942, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_944 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___943, row1_938)
            
            # Applying the binary operator '*' (line 159)
            result_mul_945 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '*', s1_937, subscript_call_result_944)
            
            # Getting the type of 's2' (line 159)
            s2_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 's2', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 159)
            row2_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'row2', False)
            
            # Obtaining the type of the subscript
            int_948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 48), 'int')
            # Getting the type of 'U' (line 159)
            U_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), U_949, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_951 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___950, int_948)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), subscript_call_result_951, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___952, row2_947)
            
            # Applying the binary operator '*' (line 159)
            result_mul_954 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 41), '*', s2_946, subscript_call_result_953)
            
            # Applying the binary operator '+' (line 159)
            result_add_955 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '+', result_mul_945, result_mul_954)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 64), 'row3', False)
            
            # Obtaining the type of the subscript
            int_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 61), 'int')
            # Getting the type of 'U' (line 159)
            U_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), U_958, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_960 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___959, int_957)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), subscript_call_result_960, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_962 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___961, row3_956)
            
            # Applying the binary operator '+' (line 159)
            result_add_963 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 57), '+', result_add_955, subscript_call_result_962)
            
            # Processing the call keyword arguments (line 159)
            kwargs_964 = {}
            # Getting the type of 'abs' (line 159)
            abs_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), abs_936, *[result_add_963], **kwargs_964)
            
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 82), 'row3', False)
            
            # Obtaining the type of the subscript
            int_968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 79), 'int')
            # Getting the type of 'U' (line 159)
            U_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 77), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), U_969, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___970, int_968)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), subscript_call_result_971, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_973 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___972, row3_967)
            
            # Processing the call keyword arguments (line 159)
            kwargs_974 = {}
            # Getting the type of 'abs' (line 159)
            abs_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 73), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_975 = invoke(stypy.reporting.localization.Localization(__file__, 159, 73), abs_966, *[subscript_call_result_973], **kwargs_974)
            
            # Applying the binary operator '-' (line 159)
            result_sub_976 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), '-', abs_call_result_965, abs_call_result_975)
            
            # Assigning a type to the variable 'diff' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'diff', result_sub_976)
            
            # Getting the type of 'diff' (line 160)
            diff_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'diff')
            int_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 22), 'int')
            # Applying the binary operator '<' (line 160)
            result_lt_979 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 15), '<', diff_977, int_978)
            
            # Testing if the type of an if condition is none (line 160)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979):
                pass
            else:
                
                # Testing the type of an if condition (line 160)
                if_condition_980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979)
                # Assigning a type to the variable 'if_condition_980' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'if_condition_980', if_condition_980)
                # SSA begins for if statement (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'stypy_return_type', int_981)
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 161)
            diff_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'diff')
            int_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 22), 'int')
            # Applying the binary operator '>' (line 161)
            result_gt_984 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '>', diff_982, int_983)
            
            # Testing if the type of an if condition is none (line 161)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984):
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
            else:
                
                # Testing the type of an if condition (line 161)
                if_condition_985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984)
                # Assigning a type to the variable 'if_condition_985' (line 161)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_985', if_condition_985)
                # SSA begins for if statement (line 161)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 162)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'stypy_return_type', int_986)
                # SSA branch for the else part of an if statement (line 161)
                module_type_store.open_ssa_branch('else')
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
                # SSA join for if statement (line 161)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 156)
            if_condition_934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_gt_933)
            # Assigning a type to the variable 'if_condition_934' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_934', if_condition_934)
            # SSA begins for if statement (line 156)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'stypy_return_type', int_935)
            # SSA branch for the else part of an if statement (line 156)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 159):
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 's1' (line 159)
            s1_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 's1', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row1' (line 159)
            row1_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'row1', False)
            
            # Obtaining the type of the subscript
            int_939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'int')
            # Getting the type of 'U' (line 159)
            U_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), U_940, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___941, int_939)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), subscript_call_result_942, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_944 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), getitem___943, row1_938)
            
            # Applying the binary operator '*' (line 159)
            result_mul_945 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '*', s1_937, subscript_call_result_944)
            
            # Getting the type of 's2' (line 159)
            s2_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 's2', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row2' (line 159)
            row2_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'row2', False)
            
            # Obtaining the type of the subscript
            int_948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 48), 'int')
            # Getting the type of 'U' (line 159)
            U_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), U_949, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_951 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___950, int_948)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 46), subscript_call_result_951, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), getitem___952, row2_947)
            
            # Applying the binary operator '*' (line 159)
            result_mul_954 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 41), '*', s2_946, subscript_call_result_953)
            
            # Applying the binary operator '+' (line 159)
            result_add_955 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '+', result_mul_945, result_mul_954)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 64), 'row3', False)
            
            # Obtaining the type of the subscript
            int_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 61), 'int')
            # Getting the type of 'U' (line 159)
            U_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), U_958, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_960 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___959, int_957)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), subscript_call_result_960, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_962 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), getitem___961, row3_956)
            
            # Applying the binary operator '+' (line 159)
            result_add_963 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 57), '+', result_add_955, subscript_call_result_962)
            
            # Processing the call keyword arguments (line 159)
            kwargs_964 = {}
            # Getting the type of 'abs' (line 159)
            abs_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), abs_936, *[result_add_963], **kwargs_964)
            
            
            # Call to abs(...): (line 159)
            # Processing the call arguments (line 159)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row3' (line 159)
            row3_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 82), 'row3', False)
            
            # Obtaining the type of the subscript
            int_968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 79), 'int')
            # Getting the type of 'U' (line 159)
            U_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 77), 'U', False)
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), U_969, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___970, int_968)
            
            # Obtaining the member '__getitem__' of a type (line 159)
            getitem___972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 77), subscript_call_result_971, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 159)
            subscript_call_result_973 = invoke(stypy.reporting.localization.Localization(__file__, 159, 77), getitem___972, row3_967)
            
            # Processing the call keyword arguments (line 159)
            kwargs_974 = {}
            # Getting the type of 'abs' (line 159)
            abs_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 73), 'abs', False)
            # Calling abs(args, kwargs) (line 159)
            abs_call_result_975 = invoke(stypy.reporting.localization.Localization(__file__, 159, 73), abs_966, *[subscript_call_result_973], **kwargs_974)
            
            # Applying the binary operator '-' (line 159)
            result_sub_976 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), '-', abs_call_result_965, abs_call_result_975)
            
            # Assigning a type to the variable 'diff' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'diff', result_sub_976)
            
            # Getting the type of 'diff' (line 160)
            diff_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'diff')
            int_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 22), 'int')
            # Applying the binary operator '<' (line 160)
            result_lt_979 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 15), '<', diff_977, int_978)
            
            # Testing if the type of an if condition is none (line 160)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979):
                pass
            else:
                
                # Testing the type of an if condition (line 160)
                if_condition_980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 12), result_lt_979)
                # Assigning a type to the variable 'if_condition_980' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'if_condition_980', if_condition_980)
                # SSA begins for if statement (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'stypy_return_type', int_981)
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'diff' (line 161)
            diff_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'diff')
            int_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 22), 'int')
            # Applying the binary operator '>' (line 161)
            result_gt_984 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '>', diff_982, int_983)
            
            # Testing if the type of an if condition is none (line 161)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984):
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
            else:
                
                # Testing the type of an if condition (line 161)
                if_condition_985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_gt_984)
                # Assigning a type to the variable 'if_condition_985' (line 161)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_985', if_condition_985)
                # SSA begins for if statement (line 161)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 162)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'stypy_return_type', int_986)
                # SSA branch for the else part of an if statement (line 161)
                module_type_store.open_ssa_branch('else')
                int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', int_987)
                # SSA join for if statement (line 161)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 156)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'examine3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'examine3' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_988)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'examine3'
    return stypy_return_type_988

# Assigning a type to the variable 'examine3' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'examine3', examine3)

@norecursion
def binary(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'binary'
    module_type_store = module_type_store.open_function_context('binary', 167, 0, False)
    
    # Passed parameters checking function
    binary.stypy_localization = localization
    binary.stypy_type_of_self = None
    binary.stypy_type_store = module_type_store
    binary.stypy_function_name = 'binary'
    binary.stypy_param_names_list = ['n']
    binary.stypy_varargs_param_name = None
    binary.stypy_kwargs_param_name = None
    binary.stypy_call_defaults = defaults
    binary.stypy_call_varargs = varargs
    binary.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary(...)' code ##################

    
    # Getting the type of 'n' (line 168)
    n_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 7), 'n')
    int_990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'int')
    # Applying the binary operator '==' (line 168)
    result_eq_991 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 7), '==', n_989, int_990)
    
    # Testing if the type of an if condition is none (line 168)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 168, 4), result_eq_991):
        pass
    else:
        
        # Testing the type of an if condition (line 168)
        if_condition_992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 4), result_eq_991)
        # Assigning a type to the variable 'if_condition_992' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'if_condition_992', if_condition_992)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'stypy_return_type', int_993)
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'n' (line 169)
    n_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'n')
    int_995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_996 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 7), '==', n_994, int_995)
    
    # Testing if the type of an if condition is none (line 169)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 169, 4), result_eq_996):
        pass
    else:
        
        # Testing the type of an if condition (line 169)
        if_condition_997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), result_eq_996)
        # Assigning a type to the variable 'if_condition_997' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_997', if_condition_997)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'stypy_return_type', int_998)
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 170):
    # Getting the type of 'n' (line 170)
    n_999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'n')
    int_1000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 12), 'int')
    # Applying the binary operator 'div' (line 170)
    result_div_1001 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 8), 'div', n_999, int_1000)
    
    # Assigning a type to the variable 'm' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'm', result_div_1001)
    
    int_1002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 7), 'int')
    # Getting the type of 'm' (line 171)
    m_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'm')
    # Applying the binary operator '*' (line 171)
    result_mul_1004 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '*', int_1002, m_1003)
    
    # Getting the type of 'n' (line 171)
    n_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'n')
    # Applying the binary operator '==' (line 171)
    result_eq_1006 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '==', result_mul_1004, n_1005)
    
    # Testing if the type of an if condition is none (line 171)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 171, 4), result_eq_1006):
        int_1014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'int')
        
        # Call to binary(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'm' (line 174)
        m_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'm', False)
        # Processing the call keyword arguments (line 174)
        kwargs_1017 = {}
        # Getting the type of 'binary' (line 174)
        binary_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'binary', False)
        # Calling binary(args, kwargs) (line 174)
        binary_call_result_1018 = invoke(stypy.reporting.localization.Localization(__file__, 174, 20), binary_1015, *[m_1016], **kwargs_1017)
        
        # Applying the binary operator '*' (line 174)
        result_mul_1019 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '*', int_1014, binary_call_result_1018)
        
        int_1020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'int')
        # Applying the binary operator '+' (line 174)
        result_add_1021 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '+', result_mul_1019, int_1020)
        
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', result_add_1021)
    else:
        
        # Testing the type of an if condition (line 171)
        if_condition_1007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_eq_1006)
        # Assigning a type to the variable 'if_condition_1007' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_1007', if_condition_1007)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'int')
        
        # Call to binary(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'm' (line 172)
        m_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'm', False)
        # Processing the call keyword arguments (line 172)
        kwargs_1011 = {}
        # Getting the type of 'binary' (line 172)
        binary_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'binary', False)
        # Calling binary(args, kwargs) (line 172)
        binary_call_result_1012 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), binary_1009, *[m_1010], **kwargs_1011)
        
        # Applying the binary operator '*' (line 172)
        result_mul_1013 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), '*', int_1008, binary_call_result_1012)
        
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', result_mul_1013)
        # SSA branch for the else part of an if statement (line 171)
        module_type_store.open_ssa_branch('else')
        int_1014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'int')
        
        # Call to binary(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'm' (line 174)
        m_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'm', False)
        # Processing the call keyword arguments (line 174)
        kwargs_1017 = {}
        # Getting the type of 'binary' (line 174)
        binary_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'binary', False)
        # Calling binary(args, kwargs) (line 174)
        binary_call_result_1018 = invoke(stypy.reporting.localization.Localization(__file__, 174, 20), binary_1015, *[m_1016], **kwargs_1017)
        
        # Applying the binary operator '*' (line 174)
        result_mul_1019 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '*', int_1014, binary_call_result_1018)
        
        int_1020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'int')
        # Applying the binary operator '+' (line 174)
        result_add_1021 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '+', result_mul_1019, int_1020)
        
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', result_add_1021)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'binary(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary'
    return stypy_return_type_1022

# Assigning a type to the variable 'binary' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'binary', binary)

# Assigning a Num to a Name (line 177):
int_1023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 9), 'int')
# Assigning a type to the variable 'length' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'length', int_1023)

# Assigning a List to a Name (line 179):

# Obtaining an instance of the builtin type 'list' (line 179)
list_1024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'list' (line 179)
list_1025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)
# Adding element type (line 179)
int_1026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 5), list_1025, int_1026)
# Adding element type (line 179)
int_1027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 5), list_1025, int_1027)
# Adding element type (line 179)
int_1028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 5), list_1025, int_1028)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 4), list_1024, list_1025)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'list' (line 179)
list_1029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)
# Adding element type (line 179)
int_1030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), list_1029, int_1030)
# Adding element type (line 179)
int_1031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), list_1029, int_1031)
# Adding element type (line 179)
int_1032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), list_1029, int_1032)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 4), list_1024, list_1029)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'list' (line 179)
list_1033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)
# Adding element type (line 179)
int_1034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 27), list_1033, int_1034)
# Adding element type (line 179)
int_1035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 27), list_1033, int_1035)
# Adding element type (line 179)
int_1036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 27), list_1033, int_1036)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 4), list_1024, list_1033)

# Assigning a type to the variable 'b' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'b', list_1024)

# Assigning a List to a Name (line 181):

# Obtaining an instance of the builtin type 'list' (line 181)
list_1037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 181)
# Adding element type (line 181)

# Obtaining an instance of the builtin type 'list' (line 181)
list_1038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 181)
# Adding element type (line 181)
int_1039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 5), list_1038, int_1039)
# Adding element type (line 181)
int_1040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 5), list_1038, int_1040)
# Adding element type (line 181)
int_1041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 5), list_1038, int_1041)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 4), list_1037, list_1038)
# Adding element type (line 181)

# Obtaining an instance of the builtin type 'list' (line 181)
list_1042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 181)
# Adding element type (line 181)
int_1043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 16), list_1042, int_1043)
# Adding element type (line 181)
int_1044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 16), list_1042, int_1044)
# Adding element type (line 181)
int_1045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 16), list_1042, int_1045)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 4), list_1037, list_1042)
# Adding element type (line 181)

# Obtaining an instance of the builtin type 'list' (line 181)
list_1046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 181)
# Adding element type (line 181)
int_1047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 27), list_1046, int_1047)
# Adding element type (line 181)
int_1048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 27), list_1046, int_1048)
# Adding element type (line 181)
int_1049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 27), list_1046, int_1049)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 4), list_1037, list_1046)

# Assigning a type to the variable 'A' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'A', list_1037)

# Assigning a Call to a Name (line 182):

# Call to inverse(...): (line 182)
# Processing the call arguments (line 182)
# Getting the type of 'A' (line 182)
A_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'A', False)
# Processing the call keyword arguments (line 182)
kwargs_1052 = {}
# Getting the type of 'inverse' (line 182)
inverse_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'inverse', False)
# Calling inverse(args, kwargs) (line 182)
inverse_call_result_1053 = invoke(stypy.reporting.localization.Localization(__file__, 182, 4), inverse_1050, *[A_1051], **kwargs_1052)

# Assigning a type to the variable 'B' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'B', inverse_call_result_1053)

# Assigning a List to a Name (line 183):

# Obtaining an instance of the builtin type 'list' (line 183)
list_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 183)
# Adding element type (line 183)

# Obtaining an instance of the builtin type 'list' (line 183)
list_1055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 183)
# Adding element type (line 183)
int_1056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 5), list_1055, int_1056)
# Adding element type (line 183)
int_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 5), list_1055, int_1057)
# Adding element type (line 183)
int_1058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 5), list_1055, int_1058)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), list_1054, list_1055)
# Adding element type (line 183)

# Obtaining an instance of the builtin type 'list' (line 183)
list_1059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 183)
# Adding element type (line 183)
int_1060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), list_1059, int_1060)
# Adding element type (line 183)
int_1061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), list_1059, int_1061)
# Adding element type (line 183)
int_1062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), list_1059, int_1062)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), list_1054, list_1059)
# Adding element type (line 183)

# Obtaining an instance of the builtin type 'list' (line 183)
list_1063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 183)
# Adding element type (line 183)
int_1064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 27), list_1063, int_1064)
# Adding element type (line 183)
int_1065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 27), list_1063, int_1065)
# Adding element type (line 183)
int_1066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 27), list_1063, int_1066)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 4), list_1054, list_1063)

# Assigning a type to the variable 'C' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'C', list_1054)

# Assigning a Call to a Name (line 184):

# Call to inverse(...): (line 184)
# Processing the call arguments (line 184)
# Getting the type of 'B' (line 184)
B_1068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'B', False)
# Processing the call keyword arguments (line 184)
kwargs_1069 = {}
# Getting the type of 'inverse' (line 184)
inverse_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'inverse', False)
# Calling inverse(args, kwargs) (line 184)
inverse_call_result_1070 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), inverse_1067, *[B_1068], **kwargs_1069)

# Assigning a type to the variable 'D' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'D', inverse_call_result_1070)

# Assigning a List to a Name (line 185):

# Obtaining an instance of the builtin type 'list' (line 185)
list_1071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 185)
# Adding element type (line 185)

# Obtaining an instance of the builtin type 'list' (line 185)
list_1072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 185)
# Adding element type (line 185)
int_1073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 5), list_1072, int_1073)
# Adding element type (line 185)
int_1074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 5), list_1072, int_1074)
# Adding element type (line 185)
int_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 5), list_1072, int_1075)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 4), list_1071, list_1072)
# Adding element type (line 185)

# Obtaining an instance of the builtin type 'list' (line 185)
list_1076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 185)
# Adding element type (line 185)
int_1077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 16), list_1076, int_1077)
# Adding element type (line 185)
int_1078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 16), list_1076, int_1078)
# Adding element type (line 185)
int_1079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 16), list_1076, int_1079)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 4), list_1071, list_1076)
# Adding element type (line 185)

# Obtaining an instance of the builtin type 'list' (line 185)
list_1080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 185)
# Adding element type (line 185)
int_1081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 27), list_1080, int_1081)
# Adding element type (line 185)
int_1082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 27), list_1080, int_1082)
# Adding element type (line 185)
int_1083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 27), list_1080, int_1083)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 4), list_1071, list_1080)

# Assigning a type to the variable 'E' (line 185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'E', list_1071)

# Assigning a Call to a Name (line 186):

# Call to inverse(...): (line 186)
# Processing the call arguments (line 186)
# Getting the type of 'E' (line 186)
E_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'E', False)
# Processing the call keyword arguments (line 186)
kwargs_1086 = {}
# Getting the type of 'inverse' (line 186)
inverse_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'inverse', False)
# Calling inverse(args, kwargs) (line 186)
inverse_call_result_1087 = invoke(stypy.reporting.localization.Localization(__file__, 186, 4), inverse_1084, *[E_1085], **kwargs_1086)

# Assigning a type to the variable 'F' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'F', inverse_call_result_1087)

# Assigning a Call to a Name (line 188):

# Call to Transpose(...): (line 188)
# Processing the call arguments (line 188)
# Getting the type of 'A' (line 188)
A_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'A', False)
# Processing the call keyword arguments (line 188)
kwargs_1090 = {}
# Getting the type of 'Transpose' (line 188)
Transpose_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 5), 'Transpose', False)
# Calling Transpose(args, kwargs) (line 188)
Transpose_call_result_1091 = invoke(stypy.reporting.localization.Localization(__file__, 188, 5), Transpose_1088, *[A_1089], **kwargs_1090)

# Assigning a type to the variable 'At' (line 188)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'At', Transpose_call_result_1091)

# Assigning a Call to a Name (line 189):

# Call to Transpose(...): (line 189)
# Processing the call arguments (line 189)
# Getting the type of 'B' (line 189)
B_1093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'B', False)
# Processing the call keyword arguments (line 189)
kwargs_1094 = {}
# Getting the type of 'Transpose' (line 189)
Transpose_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 5), 'Transpose', False)
# Calling Transpose(args, kwargs) (line 189)
Transpose_call_result_1095 = invoke(stypy.reporting.localization.Localization(__file__, 189, 5), Transpose_1092, *[B_1093], **kwargs_1094)

# Assigning a type to the variable 'Bt' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'Bt', Transpose_call_result_1095)

# Assigning a Call to a Name (line 190):

# Call to Transpose(...): (line 190)
# Processing the call arguments (line 190)
# Getting the type of 'C' (line 190)
C_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'C', False)
# Processing the call keyword arguments (line 190)
kwargs_1098 = {}
# Getting the type of 'Transpose' (line 190)
Transpose_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 5), 'Transpose', False)
# Calling Transpose(args, kwargs) (line 190)
Transpose_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 190, 5), Transpose_1096, *[C_1097], **kwargs_1098)

# Assigning a type to the variable 'Ct' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'Ct', Transpose_call_result_1099)

# Assigning a Call to a Name (line 191):

# Call to Transpose(...): (line 191)
# Processing the call arguments (line 191)
# Getting the type of 'D' (line 191)
D_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'D', False)
# Processing the call keyword arguments (line 191)
kwargs_1102 = {}
# Getting the type of 'Transpose' (line 191)
Transpose_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 5), 'Transpose', False)
# Calling Transpose(args, kwargs) (line 191)
Transpose_call_result_1103 = invoke(stypy.reporting.localization.Localization(__file__, 191, 5), Transpose_1100, *[D_1101], **kwargs_1102)

# Assigning a type to the variable 'Dt' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'Dt', Transpose_call_result_1103)

# Assigning a Call to a Name (line 192):

# Call to Transpose(...): (line 192)
# Processing the call arguments (line 192)
# Getting the type of 'E' (line 192)
E_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'E', False)
# Processing the call keyword arguments (line 192)
kwargs_1106 = {}
# Getting the type of 'Transpose' (line 192)
Transpose_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 5), 'Transpose', False)
# Calling Transpose(args, kwargs) (line 192)
Transpose_call_result_1107 = invoke(stypy.reporting.localization.Localization(__file__, 192, 5), Transpose_1104, *[E_1105], **kwargs_1106)

# Assigning a type to the variable 'Et' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'Et', Transpose_call_result_1107)

# Assigning a Call to a Name (line 193):

# Call to Transpose(...): (line 193)
# Processing the call arguments (line 193)
# Getting the type of 'F' (line 193)
F_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'F', False)
# Processing the call keyword arguments (line 193)
kwargs_1110 = {}
# Getting the type of 'Transpose' (line 193)
Transpose_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 5), 'Transpose', False)
# Calling Transpose(args, kwargs) (line 193)
Transpose_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 193, 5), Transpose_1108, *[F_1109], **kwargs_1110)

# Assigning a type to the variable 'Ft' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'Ft', Transpose_call_result_1111)

# Assigning a Call to a Name (line 195):

# Call to Transpose(...): (line 195)
# Processing the call arguments (line 195)
# Getting the type of 'b' (line 195)
b_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'b', False)
# Processing the call keyword arguments (line 195)
kwargs_1114 = {}
# Getting the type of 'Transpose' (line 195)
Transpose_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 5), 'Transpose', False)
# Calling Transpose(args, kwargs) (line 195)
Transpose_call_result_1115 = invoke(stypy.reporting.localization.Localization(__file__, 195, 5), Transpose_1112, *[b_1113], **kwargs_1114)

# Assigning a type to the variable 'bt' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'bt', Transpose_call_result_1115)

@norecursion
def descending(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'descending'
    module_type_store = module_type_store.open_function_context('descending', 198, 0, False)
    
    # Passed parameters checking function
    descending.stypy_localization = localization
    descending.stypy_type_of_self = None
    descending.stypy_type_store = module_type_store
    descending.stypy_function_name = 'descending'
    descending.stypy_param_names_list = ['U']
    descending.stypy_varargs_param_name = None
    descending.stypy_kwargs_param_name = None
    descending.stypy_call_defaults = defaults
    descending.stypy_call_varargs = varargs
    descending.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'descending', ['U'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'descending', localization, ['U'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'descending(...)' code ##################

    
    # Assigning a Num to a Name (line 199):
    int_1116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 11), 'int')
    # Assigning a type to the variable 'type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'type', int_1116)
    
    # Assigning a Call to a Name (line 201):
    
    # Call to examine(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'U' (line 201)
    U_1118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'U', False)
    int_1119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 19), 'int')
    int_1120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 22), 'int')
    # Processing the call keyword arguments (line 201)
    kwargs_1121 = {}
    # Getting the type of 'examine' (line 201)
    examine_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'examine', False)
    # Calling examine(args, kwargs) (line 201)
    examine_call_result_1122 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), examine_1117, *[U_1118, int_1119, int_1120], **kwargs_1121)
    
    # Assigning a type to the variable 'r' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'r', examine_call_result_1122)
    
    # Getting the type of 'r' (line 202)
    r_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 7), 'r')
    int_1124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'int')
    # Applying the binary operator '==' (line 202)
    result_eq_1125 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 7), '==', r_1123, int_1124)
    
    # Testing if the type of an if condition is none (line 202)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 202, 4), result_eq_1125):
        pass
    else:
        
        # Testing the type of an if condition (line 202)
        if_condition_1126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), result_eq_1125)
        # Assigning a type to the variable 'if_condition_1126' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_1126', if_condition_1126)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'stypy_return_type', int_1127)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 203)
    r_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 7), 'r')
    int_1129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 12), 'int')
    # Applying the binary operator '==' (line 203)
    result_eq_1130 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 7), '==', r_1128, int_1129)
    
    # Testing if the type of an if condition is none (line 203)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 4), result_eq_1130):
        pass
    else:
        
        # Testing the type of an if condition (line 203)
        if_condition_1131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 4), result_eq_1130)
        # Assigning a type to the variable 'if_condition_1131' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'if_condition_1131', if_condition_1131)
        # SSA begins for if statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 203):
        # Getting the type of 'type' (line 203)
        type_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'type')
        int_1133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 30), 'int')
        # Applying the binary operator '+' (line 203)
        result_add_1134 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 23), '+', type_1132, int_1133)
        
        # Assigning a type to the variable 'type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'type', result_add_1134)
        # SSA join for if statement (line 203)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 205):
    
    # Call to examine(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'U' (line 205)
    U_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'U', False)
    int_1137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 19), 'int')
    int_1138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 23), 'int')
    # Processing the call keyword arguments (line 205)
    kwargs_1139 = {}
    # Getting the type of 'examine' (line 205)
    examine_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'examine', False)
    # Calling examine(args, kwargs) (line 205)
    examine_call_result_1140 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), examine_1135, *[U_1136, int_1137, int_1138], **kwargs_1139)
    
    # Assigning a type to the variable 'r' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'r', examine_call_result_1140)
    
    # Getting the type of 'r' (line 206)
    r_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'r')
    int_1142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 12), 'int')
    # Applying the binary operator '==' (line 206)
    result_eq_1143 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 7), '==', r_1141, int_1142)
    
    # Testing if the type of an if condition is none (line 206)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 206, 4), result_eq_1143):
        pass
    else:
        
        # Testing the type of an if condition (line 206)
        if_condition_1144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 4), result_eq_1143)
        # Assigning a type to the variable 'if_condition_1144' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'if_condition_1144', if_condition_1144)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'stypy_return_type', int_1145)
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 207)
    r_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 7), 'r')
    int_1147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 12), 'int')
    # Applying the binary operator '==' (line 207)
    result_eq_1148 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 7), '==', r_1146, int_1147)
    
    # Testing if the type of an if condition is none (line 207)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 207, 4), result_eq_1148):
        pass
    else:
        
        # Testing the type of an if condition (line 207)
        if_condition_1149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 4), result_eq_1148)
        # Assigning a type to the variable 'if_condition_1149' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'if_condition_1149', if_condition_1149)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 207):
        # Getting the type of 'type' (line 207)
        type_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'type')
        int_1151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 30), 'int')
        # Applying the binary operator '+' (line 207)
        result_add_1152 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 23), '+', type_1150, int_1151)
        
        # Assigning a type to the variable 'type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'type', result_add_1152)
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 209):
    
    # Call to examine(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'U' (line 209)
    U_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'U', False)
    int_1155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 19), 'int')
    int_1156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'int')
    # Processing the call keyword arguments (line 209)
    kwargs_1157 = {}
    # Getting the type of 'examine' (line 209)
    examine_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'examine', False)
    # Calling examine(args, kwargs) (line 209)
    examine_call_result_1158 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), examine_1153, *[U_1154, int_1155, int_1156], **kwargs_1157)
    
    # Assigning a type to the variable 'r' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'r', examine_call_result_1158)
    
    # Getting the type of 'r' (line 210)
    r_1159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 7), 'r')
    int_1160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'int')
    # Applying the binary operator '==' (line 210)
    result_eq_1161 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 7), '==', r_1159, int_1160)
    
    # Testing if the type of an if condition is none (line 210)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 210, 4), result_eq_1161):
        pass
    else:
        
        # Testing the type of an if condition (line 210)
        if_condition_1162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 4), result_eq_1161)
        # Assigning a type to the variable 'if_condition_1162' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'if_condition_1162', if_condition_1162)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'stypy_return_type', int_1163)
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 211)
    r_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 7), 'r')
    int_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
    # Applying the binary operator '==' (line 211)
    result_eq_1166 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 7), '==', r_1164, int_1165)
    
    # Testing if the type of an if condition is none (line 211)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 211, 4), result_eq_1166):
        pass
    else:
        
        # Testing the type of an if condition (line 211)
        if_condition_1167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 4), result_eq_1166)
        # Assigning a type to the variable 'if_condition_1167' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'if_condition_1167', if_condition_1167)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 211):
        # Getting the type of 'type' (line 211)
        type_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'type')
        int_1169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 30), 'int')
        # Applying the binary operator '+' (line 211)
        result_add_1170 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 23), '+', type_1168, int_1169)
        
        # Assigning a type to the variable 'type' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'type', result_add_1170)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 213):
    
    # Call to examine(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'U' (line 213)
    U_1172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'U', False)
    int_1173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 19), 'int')
    int_1174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 23), 'int')
    # Processing the call keyword arguments (line 213)
    kwargs_1175 = {}
    # Getting the type of 'examine' (line 213)
    examine_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'examine', False)
    # Calling examine(args, kwargs) (line 213)
    examine_call_result_1176 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), examine_1171, *[U_1172, int_1173, int_1174], **kwargs_1175)
    
    # Assigning a type to the variable 'r' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'r', examine_call_result_1176)
    
    # Getting the type of 'r' (line 214)
    r_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 7), 'r')
    int_1178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 12), 'int')
    # Applying the binary operator '==' (line 214)
    result_eq_1179 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 7), '==', r_1177, int_1178)
    
    # Testing if the type of an if condition is none (line 214)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 214, 4), result_eq_1179):
        pass
    else:
        
        # Testing the type of an if condition (line 214)
        if_condition_1180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 4), result_eq_1179)
        # Assigning a type to the variable 'if_condition_1180' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'if_condition_1180', if_condition_1180)
        # SSA begins for if statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'stypy_return_type', int_1181)
        # SSA join for if statement (line 214)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 215)
    r_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 7), 'r')
    int_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 12), 'int')
    # Applying the binary operator '==' (line 215)
    result_eq_1184 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 7), '==', r_1182, int_1183)
    
    # Testing if the type of an if condition is none (line 215)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 215, 4), result_eq_1184):
        pass
    else:
        
        # Testing the type of an if condition (line 215)
        if_condition_1185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), result_eq_1184)
        # Assigning a type to the variable 'if_condition_1185' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_1185', if_condition_1185)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 215):
        # Getting the type of 'type' (line 215)
        type_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'type')
        int_1187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 30), 'int')
        # Applying the binary operator '+' (line 215)
        result_add_1188 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 23), '+', type_1186, int_1187)
        
        # Assigning a type to the variable 'type' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'type', result_add_1188)
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 217):
    
    # Call to examine(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'U' (line 217)
    U_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'U', False)
    int_1191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 19), 'int')
    int_1192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'int')
    # Processing the call keyword arguments (line 217)
    kwargs_1193 = {}
    # Getting the type of 'examine' (line 217)
    examine_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'examine', False)
    # Calling examine(args, kwargs) (line 217)
    examine_call_result_1194 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), examine_1189, *[U_1190, int_1191, int_1192], **kwargs_1193)
    
    # Assigning a type to the variable 'r' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'r', examine_call_result_1194)
    
    # Getting the type of 'r' (line 218)
    r_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 7), 'r')
    int_1196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 12), 'int')
    # Applying the binary operator '==' (line 218)
    result_eq_1197 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 7), '==', r_1195, int_1196)
    
    # Testing if the type of an if condition is none (line 218)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 218, 4), result_eq_1197):
        pass
    else:
        
        # Testing the type of an if condition (line 218)
        if_condition_1198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 4), result_eq_1197)
        # Assigning a type to the variable 'if_condition_1198' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'if_condition_1198', if_condition_1198)
        # SSA begins for if statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'stypy_return_type', int_1199)
        # SSA join for if statement (line 218)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 219)
    r_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 7), 'r')
    int_1201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 12), 'int')
    # Applying the binary operator '==' (line 219)
    result_eq_1202 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 7), '==', r_1200, int_1201)
    
    # Testing if the type of an if condition is none (line 219)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 219, 4), result_eq_1202):
        pass
    else:
        
        # Testing the type of an if condition (line 219)
        if_condition_1203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 4), result_eq_1202)
        # Assigning a type to the variable 'if_condition_1203' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'if_condition_1203', if_condition_1203)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 219):
        # Getting the type of 'type' (line 219)
        type_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'type')
        int_1205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 30), 'int')
        # Applying the binary operator '+' (line 219)
        result_add_1206 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 23), '+', type_1204, int_1205)
        
        # Assigning a type to the variable 'type' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'type', result_add_1206)
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 221):
    
    # Call to examine(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'U' (line 221)
    U_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'U', False)
    int_1209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 19), 'int')
    int_1210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 23), 'int')
    # Processing the call keyword arguments (line 221)
    kwargs_1211 = {}
    # Getting the type of 'examine' (line 221)
    examine_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'examine', False)
    # Calling examine(args, kwargs) (line 221)
    examine_call_result_1212 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), examine_1207, *[U_1208, int_1209, int_1210], **kwargs_1211)
    
    # Assigning a type to the variable 'r' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'r', examine_call_result_1212)
    
    # Getting the type of 'r' (line 222)
    r_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 7), 'r')
    int_1214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 12), 'int')
    # Applying the binary operator '==' (line 222)
    result_eq_1215 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 7), '==', r_1213, int_1214)
    
    # Testing if the type of an if condition is none (line 222)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 222, 4), result_eq_1215):
        pass
    else:
        
        # Testing the type of an if condition (line 222)
        if_condition_1216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 4), result_eq_1215)
        # Assigning a type to the variable 'if_condition_1216' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'if_condition_1216', if_condition_1216)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'stypy_return_type', int_1217)
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 223)
    r_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 7), 'r')
    int_1219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 12), 'int')
    # Applying the binary operator '==' (line 223)
    result_eq_1220 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 7), '==', r_1218, int_1219)
    
    # Testing if the type of an if condition is none (line 223)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 223, 4), result_eq_1220):
        pass
    else:
        
        # Testing the type of an if condition (line 223)
        if_condition_1221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 4), result_eq_1220)
        # Assigning a type to the variable 'if_condition_1221' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'if_condition_1221', if_condition_1221)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 223):
        # Getting the type of 'type' (line 223)
        type_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'type')
        int_1223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 30), 'int')
        # Applying the binary operator '+' (line 223)
        result_add_1224 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 23), '+', type_1222, int_1223)
        
        # Assigning a type to the variable 'type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'type', result_add_1224)
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 225):
    
    # Call to examine3(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'U' (line 225)
    U_1226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'U', False)
    int_1227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 20), 'int')
    int_1228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 23), 'int')
    int_1229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 26), 'int')
    # Processing the call keyword arguments (line 225)
    kwargs_1230 = {}
    # Getting the type of 'examine3' (line 225)
    examine3_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'examine3', False)
    # Calling examine3(args, kwargs) (line 225)
    examine3_call_result_1231 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), examine3_1225, *[U_1226, int_1227, int_1228, int_1229], **kwargs_1230)
    
    # Assigning a type to the variable 'r' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'r', examine3_call_result_1231)
    
    # Getting the type of 'r' (line 226)
    r_1232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'r')
    int_1233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 12), 'int')
    # Applying the binary operator '==' (line 226)
    result_eq_1234 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 7), '==', r_1232, int_1233)
    
    # Testing if the type of an if condition is none (line 226)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 4), result_eq_1234):
        pass
    else:
        
        # Testing the type of an if condition (line 226)
        if_condition_1235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 4), result_eq_1234)
        # Assigning a type to the variable 'if_condition_1235' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'if_condition_1235', if_condition_1235)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'stypy_return_type', int_1236)
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 227)
    r_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'r')
    int_1238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 12), 'int')
    # Applying the binary operator '==' (line 227)
    result_eq_1239 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 7), '==', r_1237, int_1238)
    
    # Testing if the type of an if condition is none (line 227)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 227, 4), result_eq_1239):
        pass
    else:
        
        # Testing the type of an if condition (line 227)
        if_condition_1240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), result_eq_1239)
        # Assigning a type to the variable 'if_condition_1240' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_1240', if_condition_1240)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 227):
        # Getting the type of 'type' (line 227)
        type_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'type')
        int_1242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 30), 'int')
        # Applying the binary operator '+' (line 227)
        result_add_1243 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 23), '+', type_1241, int_1242)
        
        # Assigning a type to the variable 'type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'type', result_add_1243)
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 229):
    
    # Call to examine3(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'U' (line 229)
    U_1245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), 'U', False)
    int_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 20), 'int')
    int_1247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'int')
    int_1248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 28), 'int')
    # Processing the call keyword arguments (line 229)
    kwargs_1249 = {}
    # Getting the type of 'examine3' (line 229)
    examine3_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'examine3', False)
    # Calling examine3(args, kwargs) (line 229)
    examine3_call_result_1250 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), examine3_1244, *[U_1245, int_1246, int_1247, int_1248], **kwargs_1249)
    
    # Assigning a type to the variable 'r' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'r', examine3_call_result_1250)
    
    # Getting the type of 'r' (line 230)
    r_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 7), 'r')
    int_1252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
    # Applying the binary operator '==' (line 230)
    result_eq_1253 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 7), '==', r_1251, int_1252)
    
    # Testing if the type of an if condition is none (line 230)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 4), result_eq_1253):
        pass
    else:
        
        # Testing the type of an if condition (line 230)
        if_condition_1254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 4), result_eq_1253)
        # Assigning a type to the variable 'if_condition_1254' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'if_condition_1254', if_condition_1254)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'stypy_return_type', int_1255)
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 231)
    r_1256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), 'r')
    int_1257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
    # Applying the binary operator '==' (line 231)
    result_eq_1258 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 7), '==', r_1256, int_1257)
    
    # Testing if the type of an if condition is none (line 231)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 231, 4), result_eq_1258):
        pass
    else:
        
        # Testing the type of an if condition (line 231)
        if_condition_1259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 4), result_eq_1258)
        # Assigning a type to the variable 'if_condition_1259' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'if_condition_1259', if_condition_1259)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 231):
        # Getting the type of 'type' (line 231)
        type_1260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'type')
        int_1261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 30), 'int')
        # Applying the binary operator '+' (line 231)
        result_add_1262 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 23), '+', type_1260, int_1261)
        
        # Assigning a type to the variable 'type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'type', result_add_1262)
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 233):
    
    # Call to examine3(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'U' (line 233)
    U_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 17), 'U', False)
    int_1265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'int')
    int_1266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 24), 'int')
    int_1267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 27), 'int')
    # Processing the call keyword arguments (line 233)
    kwargs_1268 = {}
    # Getting the type of 'examine3' (line 233)
    examine3_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'examine3', False)
    # Calling examine3(args, kwargs) (line 233)
    examine3_call_result_1269 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), examine3_1263, *[U_1264, int_1265, int_1266, int_1267], **kwargs_1268)
    
    # Assigning a type to the variable 'r' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'r', examine3_call_result_1269)
    
    # Getting the type of 'r' (line 234)
    r_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 7), 'r')
    int_1271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'int')
    # Applying the binary operator '==' (line 234)
    result_eq_1272 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 7), '==', r_1270, int_1271)
    
    # Testing if the type of an if condition is none (line 234)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 234, 4), result_eq_1272):
        pass
    else:
        
        # Testing the type of an if condition (line 234)
        if_condition_1273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 4), result_eq_1272)
        # Assigning a type to the variable 'if_condition_1273' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'if_condition_1273', if_condition_1273)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'stypy_return_type', int_1274)
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 235)
    r_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'r')
    int_1276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 12), 'int')
    # Applying the binary operator '==' (line 235)
    result_eq_1277 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 7), '==', r_1275, int_1276)
    
    # Testing if the type of an if condition is none (line 235)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 235, 4), result_eq_1277):
        pass
    else:
        
        # Testing the type of an if condition (line 235)
        if_condition_1278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), result_eq_1277)
        # Assigning a type to the variable 'if_condition_1278' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_1278', if_condition_1278)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 235):
        # Getting the type of 'type' (line 235)
        type_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'type')
        int_1280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 30), 'int')
        # Applying the binary operator '+' (line 235)
        result_add_1281 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 23), '+', type_1279, int_1280)
        
        # Assigning a type to the variable 'type' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'type', result_add_1281)
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 237):
    
    # Call to examine3(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'U' (line 237)
    U_1283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'U', False)
    int_1284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 20), 'int')
    int_1285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 23), 'int')
    int_1286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 27), 'int')
    # Processing the call keyword arguments (line 237)
    kwargs_1287 = {}
    # Getting the type of 'examine3' (line 237)
    examine3_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'examine3', False)
    # Calling examine3(args, kwargs) (line 237)
    examine3_call_result_1288 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), examine3_1282, *[U_1283, int_1284, int_1285, int_1286], **kwargs_1287)
    
    # Assigning a type to the variable 'r' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'r', examine3_call_result_1288)
    
    # Getting the type of 'r' (line 238)
    r_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 7), 'r')
    int_1290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'int')
    # Applying the binary operator '==' (line 238)
    result_eq_1291 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 7), '==', r_1289, int_1290)
    
    # Testing if the type of an if condition is none (line 238)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 238, 4), result_eq_1291):
        pass
    else:
        
        # Testing the type of an if condition (line 238)
        if_condition_1292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 4), result_eq_1291)
        # Assigning a type to the variable 'if_condition_1292' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'if_condition_1292', if_condition_1292)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_1293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 22), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'stypy_return_type', int_1293)
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 239)
    r_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 7), 'r')
    int_1295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 12), 'int')
    # Applying the binary operator '==' (line 239)
    result_eq_1296 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 7), '==', r_1294, int_1295)
    
    # Testing if the type of an if condition is none (line 239)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 239, 4), result_eq_1296):
        pass
    else:
        
        # Testing the type of an if condition (line 239)
        if_condition_1297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 4), result_eq_1296)
        # Assigning a type to the variable 'if_condition_1297' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'if_condition_1297', if_condition_1297)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 239):
        # Getting the type of 'type' (line 239)
        type_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'type')
        int_1299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'int')
        # Applying the binary operator '+' (line 239)
        result_add_1300 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 23), '+', type_1298, int_1299)
        
        # Assigning a type to the variable 'type' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'type', result_add_1300)
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'type' (line 241)
    type_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'type')
    # Assigning a type to the variable 'stypy_return_type' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type', type_1301)
    
    # ################# End of 'descending(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'descending' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_1302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1302)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'descending'
    return stypy_return_type_1302

# Assigning a type to the variable 'descending' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'descending', descending)

@norecursion
def main2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main2'
    module_type_store = module_type_store.open_function_context('main2', 244, 0, False)
    
    # Passed parameters checking function
    main2.stypy_localization = localization
    main2.stypy_type_of_self = None
    main2.stypy_type_store = module_type_store
    main2.stypy_function_name = 'main2'
    main2.stypy_param_names_list = []
    main2.stypy_varargs_param_name = None
    main2.stypy_kwargs_param_name = None
    main2.stypy_call_defaults = defaults
    main2.stypy_call_varargs = varargs
    main2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main2(...)' code ##################

    
    # Assigning a List to a Name (line 245):
    
    # Obtaining an instance of the builtin type 'list' (line 245)
    list_1303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 245)
    # Adding element type (line 245)
    # Getting the type of 'bt' (line 245)
    bt_1304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'bt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), list_1303, bt_1304)
    
    # Assigning a type to the variable 'list1' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'list1', list_1303)
    
    # Call to gen(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'length' (line 246)
    length_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'length', False)
    # Getting the type of 'list1' (line 246)
    list1_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'list1', False)
    # Getting the type of 'A' (line 246)
    A_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'A', False)
    # Getting the type of 'B' (line 246)
    B_1309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'B', False)
    # Getting the type of 'C' (line 246)
    C_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'C', False)
    # Getting the type of 'D' (line 246)
    D_1311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 32), 'D', False)
    # Getting the type of 'E' (line 246)
    E_1312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 35), 'E', False)
    # Getting the type of 'F' (line 246)
    F_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'F', False)
    # Processing the call keyword arguments (line 246)
    kwargs_1314 = {}
    # Getting the type of 'gen' (line 246)
    gen_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'gen', False)
    # Calling gen(args, kwargs) (line 246)
    gen_call_result_1315 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), gen_1305, *[length_1306, list1_1307, A_1308, B_1309, C_1310, D_1311, E_1312, F_1313], **kwargs_1314)
    
    
    # Assigning a ListComp to a Name (line 247):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'list1' (line 247)
    list1_1321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'list1')
    comprehension_1322 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 14), list1_1321)
    # Assigning a type to the variable 'x' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'x', comprehension_1322)
    
    # Call to inward(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'x' (line 247)
    x_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 41), 'x', False)
    # Processing the call keyword arguments (line 247)
    kwargs_1319 = {}
    # Getting the type of 'inward' (line 247)
    inward_1317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 34), 'inward', False)
    # Calling inward(args, kwargs) (line 247)
    inward_call_result_1320 = invoke(stypy.reporting.localization.Localization(__file__, 247, 34), inward_1317, *[x_1318], **kwargs_1319)
    
    # Getting the type of 'x' (line 247)
    x_1316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'x')
    list_1323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 14), list_1323, x_1316)
    # Assigning a type to the variable 'inlist' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'inlist', list_1323)
    
    # Assigning a BinOp to a Name (line 248):
    
    # Obtaining an instance of the builtin type 'list' (line 248)
    list_1324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 248)
    # Adding element type (line 248)
    int_1325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 12), list_1324, int_1325)
    
    int_1326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 18), 'int')
    # Applying the binary operator '*' (line 248)
    result_mul_1327 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 12), '*', list_1324, int_1326)
    
    # Assigning a type to the variable 'types' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'types', result_mul_1327)
    
    # Getting the type of 'inlist' (line 249)
    inlist_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 13), 'inlist')
    # Testing if the for loop is going to be iterated (line 249)
    # Testing the type of a for loop iterable (line 249)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 249, 4), inlist_1328)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 249, 4), inlist_1328):
        # Getting the type of the for loop variable (line 249)
        for_loop_var_1329 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 249, 4), inlist_1328)
        # Assigning a type to the variable 'U' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'U', for_loop_var_1329)
        # SSA begins for a for statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 250):
        
        # Call to descending(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'U' (line 250)
        U_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 23), 'U', False)
        # Processing the call keyword arguments (line 250)
        kwargs_1332 = {}
        # Getting the type of 'descending' (line 250)
        descending_1330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'descending', False)
        # Calling descending(args, kwargs) (line 250)
        descending_call_result_1333 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), descending_1330, *[U_1331], **kwargs_1332)
        
        # Assigning a type to the variable 't' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 't', descending_call_result_1333)
        
        # Getting the type of 'types' (line 251)
        types_1334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'types')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 251)
        t_1335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 't')
        # Getting the type of 'types' (line 251)
        types_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'types')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___1337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), types_1336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_1338 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), getitem___1337, t_1335)
        
        int_1339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'int')
        # Applying the binary operator '+=' (line 251)
        result_iadd_1340 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 8), '+=', subscript_call_result_1338, int_1339)
        # Getting the type of 'types' (line 251)
        types_1341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'types')
        # Getting the type of 't' (line 251)
        t_1342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 't')
        # Storing an element on a container (line 251)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 8), types_1341, (t_1342, result_iadd_1340))
        
        
        # Getting the type of 't' (line 252)
        t_1343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 11), 't')
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_1344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        int_1345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1345)
        # Adding element type (line 252)
        int_1346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1346)
        # Adding element type (line 252)
        int_1347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1347)
        # Adding element type (line 252)
        int_1348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1348)
        # Adding element type (line 252)
        int_1349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1349)
        # Adding element type (line 252)
        int_1350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1350)
        # Adding element type (line 252)
        int_1351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1351)
        # Adding element type (line 252)
        int_1352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), list_1344, int_1352)
        
        # Applying the binary operator 'in' (line 252)
        result_contains_1353 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 11), 'in', t_1343, list_1344)
        
        # Testing if the type of an if condition is none (line 252)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 252, 8), result_contains_1353):
            pass
        else:
            
            # Testing the type of an if condition (line 252)
            if_condition_1354 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 8), result_contains_1353)
            # Assigning a type to the variable 'if_condition_1354' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'if_condition_1354', if_condition_1354)
            # SSA begins for if statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to reversed(...): (line 255)
    # Processing the call arguments (line 255)
    
    # Call to range(...): (line 255)
    # Processing the call arguments (line 255)
    int_1357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 28), 'int')
    # Processing the call keyword arguments (line 255)
    kwargs_1358 = {}
    # Getting the type of 'range' (line 255)
    range_1356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'range', False)
    # Calling range(args, kwargs) (line 255)
    range_call_result_1359 = invoke(stypy.reporting.localization.Localization(__file__, 255, 22), range_1356, *[int_1357], **kwargs_1358)
    
    # Processing the call keyword arguments (line 255)
    kwargs_1360 = {}
    # Getting the type of 'reversed' (line 255)
    reversed_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'reversed', False)
    # Calling reversed(args, kwargs) (line 255)
    reversed_call_result_1361 = invoke(stypy.reporting.localization.Localization(__file__, 255, 13), reversed_1355, *[range_call_result_1359], **kwargs_1360)
    
    # Testing if the for loop is going to be iterated (line 255)
    # Testing the type of a for loop iterable (line 255)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 255, 4), reversed_call_result_1361)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 255, 4), reversed_call_result_1361):
        # Getting the type of the for loop variable (line 255)
        for_loop_var_1362 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 255, 4), reversed_call_result_1361)
        # Assigning a type to the variable 't' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 't', for_loop_var_1362)
        # SSA begins for a for statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 256)
        t_1363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 't')
        # Getting the type of 'types' (line 256)
        types_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'types')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___1365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), types_1364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_1366 = invoke(stypy.reporting.localization.Localization(__file__, 256, 11), getitem___1365, t_1363)
        
        int_1367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 22), 'int')
        # Applying the binary operator '>' (line 256)
        result_gt_1368 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), '>', subscript_call_result_1366, int_1367)
        
        # Testing if the type of an if condition is none (line 256)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 256, 8), result_gt_1368):
            pass
        else:
            
            # Testing the type of an if condition (line 256)
            if_condition_1369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), result_gt_1368)
            # Assigning a type to the variable 'if_condition_1369' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_1369', if_condition_1369)
            # SSA begins for if statement (line 256)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to binary(...): (line 257)
            # Processing the call arguments (line 257)
            # Getting the type of 't' (line 257)
            t_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 't', False)
            # Processing the call keyword arguments (line 257)
            kwargs_1372 = {}
            # Getting the type of 'binary' (line 257)
            binary_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'binary', False)
            # Calling binary(args, kwargs) (line 257)
            binary_call_result_1373 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), binary_1370, *[t_1371], **kwargs_1372)
            
            # SSA join for if statement (line 256)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'main2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main2' in the type store
    # Getting the type of 'stypy_return_type' (line 244)
    stypy_return_type_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1374)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main2'
    return stypy_return_type_1374

# Assigning a type to the variable 'main2' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'main2', main2)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 263, 0, False)
    
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

    
    
    # Call to range(...): (line 264)
    # Processing the call arguments (line 264)
    int_1376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 19), 'int')
    # Processing the call keyword arguments (line 264)
    kwargs_1377 = {}
    # Getting the type of 'range' (line 264)
    range_1375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'range', False)
    # Calling range(args, kwargs) (line 264)
    range_call_result_1378 = invoke(stypy.reporting.localization.Localization(__file__, 264, 13), range_1375, *[int_1376], **kwargs_1377)
    
    # Testing if the for loop is going to be iterated (line 264)
    # Testing the type of a for loop iterable (line 264)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 264, 4), range_call_result_1378)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 264, 4), range_call_result_1378):
        # Getting the type of the for loop variable (line 264)
        for_loop_var_1379 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 264, 4), range_call_result_1378)
        # Assigning a type to the variable 'x' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'x', for_loop_var_1379)
        # SSA begins for a for statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to main2(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_1381 = {}
        # Getting the type of 'main2' (line 265)
        main2_1380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'main2', False)
        # Calling main2(args, kwargs) (line 265)
        main2_call_result_1382 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), main2_1380, *[], **kwargs_1381)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 266)
    True_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type', True_1383)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1384)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1384

# Assigning a type to the variable 'run' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'run', run)

# Call to run(...): (line 269)
# Processing the call keyword arguments (line 269)
kwargs_1386 = {}
# Getting the type of 'run' (line 269)
run_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 0), 'run', False)
# Calling run(args, kwargs) (line 269)
run_call_result_1387 = invoke(stypy.reporting.localization.Localization(__file__, 269, 0), run_1385, *[], **kwargs_1386)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

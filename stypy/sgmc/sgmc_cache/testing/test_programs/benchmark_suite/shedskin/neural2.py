
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Back-Propagation Neural Networks
2: # 
3: # Neil Schemenauer <nas@arctrix.com>
4: # Placed in the public domain.
5: #
6: # Tweaked for Shedskin by Simon Frost <sdfrost@ucsd.edu>
7: 
8: import math
9: import random
10: import string
11: 
12: random.seed()  # (0)
13: 
14: 
15: # calculate a random number where:  a <= rand < b
16: def rand(a, b):
17:     return (b - a) * random.random() + a
18: 
19: 
20: # Make a matrix (we could use NumPy to speed this up)
21: def makeMatrix(I, J, fill=0.0):
22:     m = []
23:     for i in range(I):
24:         m.append([fill] * J)
25:     return m
26: 
27: 
28: # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
29: def sigmoid(x):
30:     return math.tanh(x)
31: 
32: 
33: # derivative of our sigmoid function
34: def dsigmoid(y):
35:     return 1.0 - y * y
36: 
37: 
38: class NN:
39:     def __init__(self, ni, nh, no):
40:         # number of input, hidden, and output nodes
41:         self.ni = ni + 1  # +1 for bias node
42:         self.nh = nh
43:         self.no = no
44: 
45:         # activations for nodes
46:         self.ai = [1.0] * self.ni
47:         self.ah = [1.0] * self.nh
48:         self.ao = [1.0] * self.no
49: 
50:         # create weights
51:         self.wi = makeMatrix(self.ni, self.nh)
52:         self.wo = makeMatrix(self.nh, self.no)
53:         # set them to random vaules
54:         for i in range(self.ni):
55:             for j in range(self.nh):
56:                 self.wi[i][j] = rand(-2.0, 2.0)
57:         for j in range(self.nh):
58:             for k in range(self.no):
59:                 self.wo[j][k] = rand(-2.0, 2.0)
60: 
61:         # last change in weights for momentum   
62:         self.ci = makeMatrix(self.ni, self.nh)
63:         self.co = makeMatrix(self.nh, self.no)
64: 
65:     def update(self, inputs):
66:         if len(inputs) != self.ni - 1:
67:             raise ValueError('wrong number of inputs')
68: 
69:         # input activations
70:         for i in range(self.ni - 1):
71:             # self.ai[i] = sigmoid(inputs[i])
72:             self.ai[i] = inputs[i]
73: 
74:         # hidden activations
75:         for j in range(self.nh):
76:             sum = 0.0
77:             for i in range(self.ni):
78:                 sum = sum + self.ai[i] * self.wi[i][j]
79:             self.ah[j] = sigmoid(sum)
80: 
81:         # output activations
82:         for k in range(self.no):
83:             sum = 0.0
84:             for j in range(self.nh):
85:                 sum = sum + self.ah[j] * self.wo[j][k]
86:             self.ao[k] = sigmoid(sum)
87: 
88:         return self.ao[:]
89: 
90:     def backPropagate(self, targets, N, M):
91:         if len(targets) != self.no:
92:             raise ValueError('wrong number of target values')
93: 
94:         # calculate error terms for output
95:         output_deltas = [0.0] * self.no
96:         for k in range(self.no):
97:             error = targets[k] - self.ao[k]
98:             output_deltas[k] = dsigmoid(self.ao[k]) * error
99: 
100:         # calculate error terms for hidden
101:         hidden_deltas = [0.0] * self.nh
102:         for j in range(self.nh):
103:             error = 0.0
104:             for k in range(self.no):
105:                 error = error + output_deltas[k] * self.wo[j][k]
106:             hidden_deltas[j] = dsigmoid(self.ah[j]) * error
107: 
108:         # update output weights
109:         for j in range(self.nh):
110:             for k in range(self.no):
111:                 change = output_deltas[k] * self.ah[j]
112:                 self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
113:                 self.co[j][k] = change
114:                 # print N*change, M*self.co[j][k]
115: 
116:         # update input weights
117:         for i in range(self.ni):
118:             for j in range(self.nh):
119:                 change = hidden_deltas[j] * self.ai[i]
120:                 self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
121:                 self.ci[i][j] = change
122: 
123:         # calculate error
124:         error = 0.0
125:         for k in range(len(targets)):
126:             error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
127:         return error
128: 
129:     def test(self, patterns):
130:         for p in patterns:
131:             ##            print p[0], '->', self.update(p[0])
132:             self.update(p[0])
133: 
134:     def weights(self):
135:         ##        print 'Input weights:'
136:         for i in range(self.ni):
137:             pass  # print self.wi[i]
138:         ##        print
139:         ##        print 'Output weights:'
140:         for j in range(self.nh):
141:             pass  # print self.wo[j]
142: 
143:     def train(self, patterns, iterations=10000, N=0.5, M=0.1):
144:         # N: learning rate
145:         # M: momentum factor
146:         for i in xrange(iterations):
147:             error = 0.0
148:             for p in patterns:
149:                 inputs = p[0]
150:                 targets = p[1]
151:                 self.update(inputs)
152:                 error = error + self.backPropagate(targets, N, M)
153:             if i % 1000 == 0:
154:                 pass  # print 'error %-14f' % error
155: 
156: 
157: def demo():
158:     # Teach network XOR function
159:     pat = [
160:         [[0, 0], [0]],
161:         [[0, 1], [1]],
162:         [[1, 0], [1]],
163:         [[1, 1], [0]]
164:     ]
165: 
166:     # create a network with two input, two hidden, and one output nodes
167:     n = NN(2, 2, 1)
168:     # train it with some patterns
169:     n.train(pat)
170:     # test it
171:     n.test(pat)
172: 
173: 
174: def run():
175:     demo()
176:     return True
177: 
178: 
179: run()
180: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import math' statement (line 8)
import math

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import random' statement (line 9)
import random

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import string' statement (line 10)
import string

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'string', string, module_type_store)


# Call to seed(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_3 = {}
# Getting the type of 'random' (line 12)
random_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 12)
seed_2 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), random_1, 'seed')
# Calling seed(args, kwargs) (line 12)
seed_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), seed_2, *[], **kwargs_3)


@norecursion
def rand(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rand'
    module_type_store = module_type_store.open_function_context('rand', 16, 0, False)
    
    # Passed parameters checking function
    rand.stypy_localization = localization
    rand.stypy_type_of_self = None
    rand.stypy_type_store = module_type_store
    rand.stypy_function_name = 'rand'
    rand.stypy_param_names_list = ['a', 'b']
    rand.stypy_varargs_param_name = None
    rand.stypy_kwargs_param_name = None
    rand.stypy_call_defaults = defaults
    rand.stypy_call_varargs = varargs
    rand.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rand', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rand', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rand(...)' code ##################

    # Getting the type of 'b' (line 17)
    b_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'b')
    # Getting the type of 'a' (line 17)
    a_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'a')
    # Applying the binary operator '-' (line 17)
    result_sub_7 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 12), '-', b_5, a_6)
    
    
    # Call to random(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_10 = {}
    # Getting the type of 'random' (line 17)
    random_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'random', False)
    # Obtaining the member 'random' of a type (line 17)
    random_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 21), random_8, 'random')
    # Calling random(args, kwargs) (line 17)
    random_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 17, 21), random_9, *[], **kwargs_10)
    
    # Applying the binary operator '*' (line 17)
    result_mul_12 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 11), '*', result_sub_7, random_call_result_11)
    
    # Getting the type of 'a' (line 17)
    a_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 39), 'a')
    # Applying the binary operator '+' (line 17)
    result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 11), '+', result_mul_12, a_13)
    
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', result_add_14)
    
    # ################# End of 'rand(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rand' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rand'
    return stypy_return_type_15

# Assigning a type to the variable 'rand' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'rand', rand)

@norecursion
def makeMatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'float')
    defaults = [float_16]
    # Create a new context for function 'makeMatrix'
    module_type_store = module_type_store.open_function_context('makeMatrix', 21, 0, False)
    
    # Passed parameters checking function
    makeMatrix.stypy_localization = localization
    makeMatrix.stypy_type_of_self = None
    makeMatrix.stypy_type_store = module_type_store
    makeMatrix.stypy_function_name = 'makeMatrix'
    makeMatrix.stypy_param_names_list = ['I', 'J', 'fill']
    makeMatrix.stypy_varargs_param_name = None
    makeMatrix.stypy_kwargs_param_name = None
    makeMatrix.stypy_call_defaults = defaults
    makeMatrix.stypy_call_varargs = varargs
    makeMatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'makeMatrix', ['I', 'J', 'fill'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'makeMatrix', localization, ['I', 'J', 'fill'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'makeMatrix(...)' code ##################

    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    
    # Assigning a type to the variable 'm' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'm', list_17)
    
    
    # Call to range(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'I' (line 23)
    I_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'I', False)
    # Processing the call keyword arguments (line 23)
    kwargs_20 = {}
    # Getting the type of 'range' (line 23)
    range_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'range', False)
    # Calling range(args, kwargs) (line 23)
    range_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), range_18, *[I_19], **kwargs_20)
    
    # Assigning a type to the variable 'range_call_result_21' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'range_call_result_21', range_call_result_21)
    # Testing if the for loop is going to be iterated (line 23)
    # Testing the type of a for loop iterable (line 23)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 4), range_call_result_21)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 23, 4), range_call_result_21):
        # Getting the type of the for loop variable (line 23)
        for_loop_var_22 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 4), range_call_result_21)
        # Assigning a type to the variable 'i' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'i', for_loop_var_22)
        # SSA begins for a for statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        # Getting the type of 'fill' (line 24)
        fill_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'fill', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), list_25, fill_26)
        
        # Getting the type of 'J' (line 24)
        J_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'J', False)
        # Applying the binary operator '*' (line 24)
        result_mul_28 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 17), '*', list_25, J_27)
        
        # Processing the call keyword arguments (line 24)
        kwargs_29 = {}
        # Getting the type of 'm' (line 24)
        m_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'm', False)
        # Obtaining the member 'append' of a type (line 24)
        append_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), m_23, 'append')
        # Calling append(args, kwargs) (line 24)
        append_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), append_24, *[result_mul_28], **kwargs_29)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'm' (line 25)
    m_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', m_31)
    
    # ################# End of 'makeMatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'makeMatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'makeMatrix'
    return stypy_return_type_32

# Assigning a type to the variable 'makeMatrix' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'makeMatrix', makeMatrix)

@norecursion
def sigmoid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sigmoid'
    module_type_store = module_type_store.open_function_context('sigmoid', 29, 0, False)
    
    # Passed parameters checking function
    sigmoid.stypy_localization = localization
    sigmoid.stypy_type_of_self = None
    sigmoid.stypy_type_store = module_type_store
    sigmoid.stypy_function_name = 'sigmoid'
    sigmoid.stypy_param_names_list = ['x']
    sigmoid.stypy_varargs_param_name = None
    sigmoid.stypy_kwargs_param_name = None
    sigmoid.stypy_call_defaults = defaults
    sigmoid.stypy_call_varargs = varargs
    sigmoid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sigmoid', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sigmoid', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sigmoid(...)' code ##################

    
    # Call to tanh(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'x' (line 30)
    x_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'x', False)
    # Processing the call keyword arguments (line 30)
    kwargs_36 = {}
    # Getting the type of 'math' (line 30)
    math_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'math', False)
    # Obtaining the member 'tanh' of a type (line 30)
    tanh_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), math_33, 'tanh')
    # Calling tanh(args, kwargs) (line 30)
    tanh_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), tanh_34, *[x_35], **kwargs_36)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', tanh_call_result_37)
    
    # ################# End of 'sigmoid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sigmoid' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sigmoid'
    return stypy_return_type_38

# Assigning a type to the variable 'sigmoid' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'sigmoid', sigmoid)

@norecursion
def dsigmoid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dsigmoid'
    module_type_store = module_type_store.open_function_context('dsigmoid', 34, 0, False)
    
    # Passed parameters checking function
    dsigmoid.stypy_localization = localization
    dsigmoid.stypy_type_of_self = None
    dsigmoid.stypy_type_store = module_type_store
    dsigmoid.stypy_function_name = 'dsigmoid'
    dsigmoid.stypy_param_names_list = ['y']
    dsigmoid.stypy_varargs_param_name = None
    dsigmoid.stypy_kwargs_param_name = None
    dsigmoid.stypy_call_defaults = defaults
    dsigmoid.stypy_call_varargs = varargs
    dsigmoid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dsigmoid', ['y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dsigmoid', localization, ['y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dsigmoid(...)' code ##################

    float_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'float')
    # Getting the type of 'y' (line 35)
    y_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'y')
    # Getting the type of 'y' (line 35)
    y_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'y')
    # Applying the binary operator '*' (line 35)
    result_mul_42 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 17), '*', y_40, y_41)
    
    # Applying the binary operator '-' (line 35)
    result_sub_43 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), '-', float_39, result_mul_42)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', result_sub_43)
    
    # ################# End of 'dsigmoid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dsigmoid' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dsigmoid'
    return stypy_return_type_44

# Assigning a type to the variable 'dsigmoid' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'dsigmoid', dsigmoid)
# Declaration of the 'NN' class

class NN:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NN.__init__', ['ni', 'nh', 'no'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ni', 'nh', 'no'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 41):
        # Getting the type of 'ni' (line 41)
        ni_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'ni')
        int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'int')
        # Applying the binary operator '+' (line 41)
        result_add_47 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 18), '+', ni_45, int_46)
        
        # Getting the type of 'self' (line 41)
        self_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'ni' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_48, 'ni', result_add_47)
        
        # Assigning a Name to a Attribute (line 42):
        # Getting the type of 'nh' (line 42)
        nh_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'nh')
        # Getting the type of 'self' (line 42)
        self_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'nh' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_50, 'nh', nh_49)
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'no' (line 43)
        no_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'no')
        # Getting the type of 'self' (line 43)
        self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'no' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_52, 'no', no_51)
        
        # Assigning a BinOp to a Attribute (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        float_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 18), list_53, float_54)
        
        # Getting the type of 'self' (line 46)
        self_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'self')
        # Obtaining the member 'ni' of a type (line 46)
        ni_56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 26), self_55, 'ni')
        # Applying the binary operator '*' (line 46)
        result_mul_57 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 18), '*', list_53, ni_56)
        
        # Getting the type of 'self' (line 46)
        self_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'ai' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_58, 'ai', result_mul_57)
        
        # Assigning a BinOp to a Attribute (line 47):
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        float_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 18), list_59, float_60)
        
        # Getting the type of 'self' (line 47)
        self_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'self')
        # Obtaining the member 'nh' of a type (line 47)
        nh_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 26), self_61, 'nh')
        # Applying the binary operator '*' (line 47)
        result_mul_63 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 18), '*', list_59, nh_62)
        
        # Getting the type of 'self' (line 47)
        self_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'ah' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_64, 'ah', result_mul_63)
        
        # Assigning a BinOp to a Attribute (line 48):
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        float_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_65, float_66)
        
        # Getting the type of 'self' (line 48)
        self_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'self')
        # Obtaining the member 'no' of a type (line 48)
        no_68 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 26), self_67, 'no')
        # Applying the binary operator '*' (line 48)
        result_mul_69 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 18), '*', list_65, no_68)
        
        # Getting the type of 'self' (line 48)
        self_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'ao' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_70, 'ao', result_mul_69)
        
        # Assigning a Call to a Attribute (line 51):
        
        # Call to makeMatrix(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'self', False)
        # Obtaining the member 'ni' of a type (line 51)
        ni_73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), self_72, 'ni')
        # Getting the type of 'self' (line 51)
        self_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'self', False)
        # Obtaining the member 'nh' of a type (line 51)
        nh_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 38), self_74, 'nh')
        # Processing the call keyword arguments (line 51)
        kwargs_76 = {}
        # Getting the type of 'makeMatrix' (line 51)
        makeMatrix_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'makeMatrix', False)
        # Calling makeMatrix(args, kwargs) (line 51)
        makeMatrix_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), makeMatrix_71, *[ni_73, nh_75], **kwargs_76)
        
        # Getting the type of 'self' (line 51)
        self_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member 'wi' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_78, 'wi', makeMatrix_call_result_77)
        
        # Assigning a Call to a Attribute (line 52):
        
        # Call to makeMatrix(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'self' (line 52)
        self_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'self', False)
        # Obtaining the member 'nh' of a type (line 52)
        nh_81 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 29), self_80, 'nh')
        # Getting the type of 'self' (line 52)
        self_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'self', False)
        # Obtaining the member 'no' of a type (line 52)
        no_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 38), self_82, 'no')
        # Processing the call keyword arguments (line 52)
        kwargs_84 = {}
        # Getting the type of 'makeMatrix' (line 52)
        makeMatrix_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'makeMatrix', False)
        # Calling makeMatrix(args, kwargs) (line 52)
        makeMatrix_call_result_85 = invoke(stypy.reporting.localization.Localization(__file__, 52, 18), makeMatrix_79, *[nh_81, no_83], **kwargs_84)
        
        # Getting the type of 'self' (line 52)
        self_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member 'wo' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_86, 'wo', makeMatrix_call_result_85)
        
        
        # Call to range(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'self', False)
        # Obtaining the member 'ni' of a type (line 54)
        ni_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 23), self_88, 'ni')
        # Processing the call keyword arguments (line 54)
        kwargs_90 = {}
        # Getting the type of 'range' (line 54)
        range_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'range', False)
        # Calling range(args, kwargs) (line 54)
        range_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), range_87, *[ni_89], **kwargs_90)
        
        # Assigning a type to the variable 'range_call_result_91' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'range_call_result_91', range_call_result_91)
        # Testing if the for loop is going to be iterated (line 54)
        # Testing the type of a for loop iterable (line 54)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 8), range_call_result_91)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 8), range_call_result_91):
            # Getting the type of the for loop variable (line 54)
            for_loop_var_92 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 8), range_call_result_91)
            # Assigning a type to the variable 'i' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'i', for_loop_var_92)
            # SSA begins for a for statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'self' (line 55)
            self_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'self', False)
            # Obtaining the member 'nh' of a type (line 55)
            nh_95 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 27), self_94, 'nh')
            # Processing the call keyword arguments (line 55)
            kwargs_96 = {}
            # Getting the type of 'range' (line 55)
            range_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'range', False)
            # Calling range(args, kwargs) (line 55)
            range_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), range_93, *[nh_95], **kwargs_96)
            
            # Assigning a type to the variable 'range_call_result_97' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'range_call_result_97', range_call_result_97)
            # Testing if the for loop is going to be iterated (line 55)
            # Testing the type of a for loop iterable (line 55)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 12), range_call_result_97)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 55, 12), range_call_result_97):
                # Getting the type of the for loop variable (line 55)
                for_loop_var_98 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 12), range_call_result_97)
                # Assigning a type to the variable 'j' (line 55)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'j', for_loop_var_98)
                # SSA begins for a for statement (line 55)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Subscript (line 56):
                
                # Call to rand(...): (line 56)
                # Processing the call arguments (line 56)
                float_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 37), 'float')
                float_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 43), 'float')
                # Processing the call keyword arguments (line 56)
                kwargs_102 = {}
                # Getting the type of 'rand' (line 56)
                rand_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'rand', False)
                # Calling rand(args, kwargs) (line 56)
                rand_call_result_103 = invoke(stypy.reporting.localization.Localization(__file__, 56, 32), rand_99, *[float_100, float_101], **kwargs_102)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 56)
                i_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'i')
                # Getting the type of 'self' (line 56)
                self_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'self')
                # Obtaining the member 'wi' of a type (line 56)
                wi_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), self_105, 'wi')
                # Obtaining the member '__getitem__' of a type (line 56)
                getitem___107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), wi_106, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 56)
                subscript_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), getitem___107, i_104)
                
                # Getting the type of 'j' (line 56)
                j_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'j')
                # Storing an element on a container (line 56)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 16), subscript_call_result_108, (j_109, rand_call_result_103))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'self', False)
        # Obtaining the member 'nh' of a type (line 57)
        nh_112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), self_111, 'nh')
        # Processing the call keyword arguments (line 57)
        kwargs_113 = {}
        # Getting the type of 'range' (line 57)
        range_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'range', False)
        # Calling range(args, kwargs) (line 57)
        range_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), range_110, *[nh_112], **kwargs_113)
        
        # Assigning a type to the variable 'range_call_result_114' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'range_call_result_114', range_call_result_114)
        # Testing if the for loop is going to be iterated (line 57)
        # Testing the type of a for loop iterable (line 57)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 8), range_call_result_114)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 57, 8), range_call_result_114):
            # Getting the type of the for loop variable (line 57)
            for_loop_var_115 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 8), range_call_result_114)
            # Assigning a type to the variable 'j' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'j', for_loop_var_115)
            # SSA begins for a for statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 58)
            # Processing the call arguments (line 58)
            # Getting the type of 'self' (line 58)
            self_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'self', False)
            # Obtaining the member 'no' of a type (line 58)
            no_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 27), self_117, 'no')
            # Processing the call keyword arguments (line 58)
            kwargs_119 = {}
            # Getting the type of 'range' (line 58)
            range_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'range', False)
            # Calling range(args, kwargs) (line 58)
            range_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 58, 21), range_116, *[no_118], **kwargs_119)
            
            # Assigning a type to the variable 'range_call_result_120' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'range_call_result_120', range_call_result_120)
            # Testing if the for loop is going to be iterated (line 58)
            # Testing the type of a for loop iterable (line 58)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 12), range_call_result_120)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 58, 12), range_call_result_120):
                # Getting the type of the for loop variable (line 58)
                for_loop_var_121 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 12), range_call_result_120)
                # Assigning a type to the variable 'k' (line 58)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'k', for_loop_var_121)
                # SSA begins for a for statement (line 58)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Subscript (line 59):
                
                # Call to rand(...): (line 59)
                # Processing the call arguments (line 59)
                float_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 37), 'float')
                float_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 43), 'float')
                # Processing the call keyword arguments (line 59)
                kwargs_125 = {}
                # Getting the type of 'rand' (line 59)
                rand_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'rand', False)
                # Calling rand(args, kwargs) (line 59)
                rand_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 59, 32), rand_122, *[float_123, float_124], **kwargs_125)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 59)
                j_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'j')
                # Getting the type of 'self' (line 59)
                self_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'self')
                # Obtaining the member 'wo' of a type (line 59)
                wo_129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), self_128, 'wo')
                # Obtaining the member '__getitem__' of a type (line 59)
                getitem___130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), wo_129, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 59)
                subscript_call_result_131 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), getitem___130, j_127)
                
                # Getting the type of 'k' (line 59)
                k_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'k')
                # Storing an element on a container (line 59)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), subscript_call_result_131, (k_132, rand_call_result_126))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Attribute (line 62):
        
        # Call to makeMatrix(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 29), 'self', False)
        # Obtaining the member 'ni' of a type (line 62)
        ni_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 29), self_134, 'ni')
        # Getting the type of 'self' (line 62)
        self_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'self', False)
        # Obtaining the member 'nh' of a type (line 62)
        nh_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 38), self_136, 'nh')
        # Processing the call keyword arguments (line 62)
        kwargs_138 = {}
        # Getting the type of 'makeMatrix' (line 62)
        makeMatrix_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'makeMatrix', False)
        # Calling makeMatrix(args, kwargs) (line 62)
        makeMatrix_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 62, 18), makeMatrix_133, *[ni_135, nh_137], **kwargs_138)
        
        # Getting the type of 'self' (line 62)
        self_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'ci' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_140, 'ci', makeMatrix_call_result_139)
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to makeMatrix(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'self', False)
        # Obtaining the member 'nh' of a type (line 63)
        nh_143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 29), self_142, 'nh')
        # Getting the type of 'self' (line 63)
        self_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 38), 'self', False)
        # Obtaining the member 'no' of a type (line 63)
        no_145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 38), self_144, 'no')
        # Processing the call keyword arguments (line 63)
        kwargs_146 = {}
        # Getting the type of 'makeMatrix' (line 63)
        makeMatrix_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'makeMatrix', False)
        # Calling makeMatrix(args, kwargs) (line 63)
        makeMatrix_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 63, 18), makeMatrix_141, *[nh_143, no_145], **kwargs_146)
        
        # Getting the type of 'self' (line 63)
        self_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'co' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_148, 'co', makeMatrix_call_result_147)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NN.update.__dict__.__setitem__('stypy_localization', localization)
        NN.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NN.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        NN.update.__dict__.__setitem__('stypy_function_name', 'NN.update')
        NN.update.__dict__.__setitem__('stypy_param_names_list', ['inputs'])
        NN.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        NN.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NN.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        NN.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        NN.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NN.update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NN.update', ['inputs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['inputs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        
        
        # Call to len(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'inputs' (line 66)
        inputs_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'inputs', False)
        # Processing the call keyword arguments (line 66)
        kwargs_151 = {}
        # Getting the type of 'len' (line 66)
        len_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'len', False)
        # Calling len(args, kwargs) (line 66)
        len_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), len_149, *[inputs_150], **kwargs_151)
        
        # Getting the type of 'self' (line 66)
        self_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'self')
        # Obtaining the member 'ni' of a type (line 66)
        ni_154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 26), self_153, 'ni')
        int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'int')
        # Applying the binary operator '-' (line 66)
        result_sub_156 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 26), '-', ni_154, int_155)
        
        # Applying the binary operator '!=' (line 66)
        result_ne_157 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), '!=', len_call_result_152, result_sub_156)
        
        # Testing if the type of an if condition is none (line 66)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 8), result_ne_157):
            pass
        else:
            
            # Testing the type of an if condition (line 66)
            if_condition_158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_ne_157)
            # Assigning a type to the variable 'if_condition_158' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_158', if_condition_158)
            # SSA begins for if statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 67)
            # Processing the call arguments (line 67)
            str_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'str', 'wrong number of inputs')
            # Processing the call keyword arguments (line 67)
            kwargs_161 = {}
            # Getting the type of 'ValueError' (line 67)
            ValueError_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 67)
            ValueError_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), ValueError_159, *[str_160], **kwargs_161)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 67, 12), ValueError_call_result_162, 'raise parameter', BaseException)
            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to range(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'self', False)
        # Obtaining the member 'ni' of a type (line 70)
        ni_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 23), self_164, 'ni')
        int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 33), 'int')
        # Applying the binary operator '-' (line 70)
        result_sub_167 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 23), '-', ni_165, int_166)
        
        # Processing the call keyword arguments (line 70)
        kwargs_168 = {}
        # Getting the type of 'range' (line 70)
        range_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'range', False)
        # Calling range(args, kwargs) (line 70)
        range_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), range_163, *[result_sub_167], **kwargs_168)
        
        # Assigning a type to the variable 'range_call_result_169' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'range_call_result_169', range_call_result_169)
        # Testing if the for loop is going to be iterated (line 70)
        # Testing the type of a for loop iterable (line 70)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 8), range_call_result_169)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 70, 8), range_call_result_169):
            # Getting the type of the for loop variable (line 70)
            for_loop_var_170 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 8), range_call_result_169)
            # Assigning a type to the variable 'i' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'i', for_loop_var_170)
            # SSA begins for a for statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Subscript (line 72):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 72)
            i_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'i')
            # Getting the type of 'inputs' (line 72)
            inputs_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'inputs')
            # Obtaining the member '__getitem__' of a type (line 72)
            getitem___173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), inputs_172, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 72)
            subscript_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 72, 25), getitem___173, i_171)
            
            # Getting the type of 'self' (line 72)
            self_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'self')
            # Obtaining the member 'ai' of a type (line 72)
            ai_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), self_175, 'ai')
            # Getting the type of 'i' (line 72)
            i_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'i')
            # Storing an element on a container (line 72)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), ai_176, (i_177, subscript_call_result_174))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'self', False)
        # Obtaining the member 'nh' of a type (line 75)
        nh_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), self_179, 'nh')
        # Processing the call keyword arguments (line 75)
        kwargs_181 = {}
        # Getting the type of 'range' (line 75)
        range_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'range', False)
        # Calling range(args, kwargs) (line 75)
        range_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), range_178, *[nh_180], **kwargs_181)
        
        # Assigning a type to the variable 'range_call_result_182' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'range_call_result_182', range_call_result_182)
        # Testing if the for loop is going to be iterated (line 75)
        # Testing the type of a for loop iterable (line 75)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 75, 8), range_call_result_182)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 75, 8), range_call_result_182):
            # Getting the type of the for loop variable (line 75)
            for_loop_var_183 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 75, 8), range_call_result_182)
            # Assigning a type to the variable 'j' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'j', for_loop_var_183)
            # SSA begins for a for statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 76):
            float_184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'float')
            # Assigning a type to the variable 'sum' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'sum', float_184)
            
            
            # Call to range(...): (line 77)
            # Processing the call arguments (line 77)
            # Getting the type of 'self' (line 77)
            self_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'self', False)
            # Obtaining the member 'ni' of a type (line 77)
            ni_187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 27), self_186, 'ni')
            # Processing the call keyword arguments (line 77)
            kwargs_188 = {}
            # Getting the type of 'range' (line 77)
            range_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'range', False)
            # Calling range(args, kwargs) (line 77)
            range_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), range_185, *[ni_187], **kwargs_188)
            
            # Assigning a type to the variable 'range_call_result_189' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'range_call_result_189', range_call_result_189)
            # Testing if the for loop is going to be iterated (line 77)
            # Testing the type of a for loop iterable (line 77)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_189)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_189):
                # Getting the type of the for loop variable (line 77)
                for_loop_var_190 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_189)
                # Assigning a type to the variable 'i' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'i', for_loop_var_190)
                # SSA begins for a for statement (line 77)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 78):
                # Getting the type of 'sum' (line 78)
                sum_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'sum')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 78)
                i_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 36), 'i')
                # Getting the type of 'self' (line 78)
                self_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'self')
                # Obtaining the member 'ai' of a type (line 78)
                ai_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 28), self_193, 'ai')
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 28), ai_194, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 78, 28), getitem___195, i_192)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 78)
                j_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 52), 'j')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 78)
                i_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 49), 'i')
                # Getting the type of 'self' (line 78)
                self_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 'self')
                # Obtaining the member 'wi' of a type (line 78)
                wi_200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 41), self_199, 'wi')
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 41), wi_200, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 78, 41), getitem___201, i_198)
                
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 41), subscript_call_result_202, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 78, 41), getitem___203, j_197)
                
                # Applying the binary operator '*' (line 78)
                result_mul_205 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 28), '*', subscript_call_result_196, subscript_call_result_204)
                
                # Applying the binary operator '+' (line 78)
                result_add_206 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 22), '+', sum_191, result_mul_205)
                
                # Assigning a type to the variable 'sum' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'sum', result_add_206)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Subscript (line 79):
            
            # Call to sigmoid(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'sum' (line 79)
            sum_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'sum', False)
            # Processing the call keyword arguments (line 79)
            kwargs_209 = {}
            # Getting the type of 'sigmoid' (line 79)
            sigmoid_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'sigmoid', False)
            # Calling sigmoid(args, kwargs) (line 79)
            sigmoid_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), sigmoid_207, *[sum_208], **kwargs_209)
            
            # Getting the type of 'self' (line 79)
            self_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self')
            # Obtaining the member 'ah' of a type (line 79)
            ah_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_211, 'ah')
            # Getting the type of 'j' (line 79)
            j_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'j')
            # Storing an element on a container (line 79)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 12), ah_212, (j_213, sigmoid_call_result_210))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'self' (line 82)
        self_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'self', False)
        # Obtaining the member 'no' of a type (line 82)
        no_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 23), self_215, 'no')
        # Processing the call keyword arguments (line 82)
        kwargs_217 = {}
        # Getting the type of 'range' (line 82)
        range_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'range', False)
        # Calling range(args, kwargs) (line 82)
        range_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), range_214, *[no_216], **kwargs_217)
        
        # Assigning a type to the variable 'range_call_result_218' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'range_call_result_218', range_call_result_218)
        # Testing if the for loop is going to be iterated (line 82)
        # Testing the type of a for loop iterable (line 82)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 8), range_call_result_218)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 82, 8), range_call_result_218):
            # Getting the type of the for loop variable (line 82)
            for_loop_var_219 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 8), range_call_result_218)
            # Assigning a type to the variable 'k' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'k', for_loop_var_219)
            # SSA begins for a for statement (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 83):
            float_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 18), 'float')
            # Assigning a type to the variable 'sum' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'sum', float_220)
            
            
            # Call to range(...): (line 84)
            # Processing the call arguments (line 84)
            # Getting the type of 'self' (line 84)
            self_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'self', False)
            # Obtaining the member 'nh' of a type (line 84)
            nh_223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), self_222, 'nh')
            # Processing the call keyword arguments (line 84)
            kwargs_224 = {}
            # Getting the type of 'range' (line 84)
            range_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'range', False)
            # Calling range(args, kwargs) (line 84)
            range_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), range_221, *[nh_223], **kwargs_224)
            
            # Assigning a type to the variable 'range_call_result_225' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'range_call_result_225', range_call_result_225)
            # Testing if the for loop is going to be iterated (line 84)
            # Testing the type of a for loop iterable (line 84)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 12), range_call_result_225)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 12), range_call_result_225):
                # Getting the type of the for loop variable (line 84)
                for_loop_var_226 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 12), range_call_result_225)
                # Assigning a type to the variable 'j' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'j', for_loop_var_226)
                # SSA begins for a for statement (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 85):
                # Getting the type of 'sum' (line 85)
                sum_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'sum')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 85)
                j_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'j')
                # Getting the type of 'self' (line 85)
                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'self')
                # Obtaining the member 'ah' of a type (line 85)
                ah_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 28), self_229, 'ah')
                # Obtaining the member '__getitem__' of a type (line 85)
                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 28), ah_230, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 85)
                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 85, 28), getitem___231, j_228)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 85)
                k_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 52), 'k')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 85)
                j_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'j')
                # Getting the type of 'self' (line 85)
                self_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'self')
                # Obtaining the member 'wo' of a type (line 85)
                wo_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), self_235, 'wo')
                # Obtaining the member '__getitem__' of a type (line 85)
                getitem___237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), wo_236, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 85)
                subscript_call_result_238 = invoke(stypy.reporting.localization.Localization(__file__, 85, 41), getitem___237, j_234)
                
                # Obtaining the member '__getitem__' of a type (line 85)
                getitem___239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), subscript_call_result_238, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 85)
                subscript_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 85, 41), getitem___239, k_233)
                
                # Applying the binary operator '*' (line 85)
                result_mul_241 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 28), '*', subscript_call_result_232, subscript_call_result_240)
                
                # Applying the binary operator '+' (line 85)
                result_add_242 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 22), '+', sum_227, result_mul_241)
                
                # Assigning a type to the variable 'sum' (line 85)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'sum', result_add_242)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Subscript (line 86):
            
            # Call to sigmoid(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'sum' (line 86)
            sum_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'sum', False)
            # Processing the call keyword arguments (line 86)
            kwargs_245 = {}
            # Getting the type of 'sigmoid' (line 86)
            sigmoid_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'sigmoid', False)
            # Calling sigmoid(args, kwargs) (line 86)
            sigmoid_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 86, 25), sigmoid_243, *[sum_244], **kwargs_245)
            
            # Getting the type of 'self' (line 86)
            self_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self')
            # Obtaining the member 'ao' of a type (line 86)
            ao_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_247, 'ao')
            # Getting the type of 'k' (line 86)
            k_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'k')
            # Storing an element on a container (line 86)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 12), ao_248, (k_249, sigmoid_call_result_246))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        slice_250 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 88, 15), None, None, None)
        # Getting the type of 'self' (line 88)
        self_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'self')
        # Obtaining the member 'ao' of a type (line 88)
        ao_252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), self_251, 'ao')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), ao_252, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), getitem___253, slice_250)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', subscript_call_result_254)
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_255)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_255


    @norecursion
    def backPropagate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'backPropagate'
        module_type_store = module_type_store.open_function_context('backPropagate', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NN.backPropagate.__dict__.__setitem__('stypy_localization', localization)
        NN.backPropagate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NN.backPropagate.__dict__.__setitem__('stypy_type_store', module_type_store)
        NN.backPropagate.__dict__.__setitem__('stypy_function_name', 'NN.backPropagate')
        NN.backPropagate.__dict__.__setitem__('stypy_param_names_list', ['targets', 'N', 'M'])
        NN.backPropagate.__dict__.__setitem__('stypy_varargs_param_name', None)
        NN.backPropagate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NN.backPropagate.__dict__.__setitem__('stypy_call_defaults', defaults)
        NN.backPropagate.__dict__.__setitem__('stypy_call_varargs', varargs)
        NN.backPropagate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NN.backPropagate.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NN.backPropagate', ['targets', 'N', 'M'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'backPropagate', localization, ['targets', 'N', 'M'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'backPropagate(...)' code ##################

        
        
        # Call to len(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'targets' (line 91)
        targets_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'targets', False)
        # Processing the call keyword arguments (line 91)
        kwargs_258 = {}
        # Getting the type of 'len' (line 91)
        len_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'len', False)
        # Calling len(args, kwargs) (line 91)
        len_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 91, 11), len_256, *[targets_257], **kwargs_258)
        
        # Getting the type of 'self' (line 91)
        self_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'self')
        # Obtaining the member 'no' of a type (line 91)
        no_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 27), self_260, 'no')
        # Applying the binary operator '!=' (line 91)
        result_ne_262 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), '!=', len_call_result_259, no_261)
        
        # Testing if the type of an if condition is none (line 91)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 8), result_ne_262):
            pass
        else:
            
            # Testing the type of an if condition (line 91)
            if_condition_263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_ne_262)
            # Assigning a type to the variable 'if_condition_263' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_263', if_condition_263)
            # SSA begins for if statement (line 91)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 92)
            # Processing the call arguments (line 92)
            str_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'str', 'wrong number of target values')
            # Processing the call keyword arguments (line 92)
            kwargs_266 = {}
            # Getting the type of 'ValueError' (line 92)
            ValueError_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 92)
            ValueError_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 92, 18), ValueError_264, *[str_265], **kwargs_266)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 92, 12), ValueError_call_result_267, 'raise parameter', BaseException)
            # SSA join for if statement (line 91)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 95):
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        float_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 24), list_268, float_269)
        
        # Getting the type of 'self' (line 95)
        self_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'self')
        # Obtaining the member 'no' of a type (line 95)
        no_271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 32), self_270, 'no')
        # Applying the binary operator '*' (line 95)
        result_mul_272 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 24), '*', list_268, no_271)
        
        # Assigning a type to the variable 'output_deltas' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'output_deltas', result_mul_272)
        
        
        # Call to range(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'self' (line 96)
        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'self', False)
        # Obtaining the member 'no' of a type (line 96)
        no_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 23), self_274, 'no')
        # Processing the call keyword arguments (line 96)
        kwargs_276 = {}
        # Getting the type of 'range' (line 96)
        range_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'range', False)
        # Calling range(args, kwargs) (line 96)
        range_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), range_273, *[no_275], **kwargs_276)
        
        # Assigning a type to the variable 'range_call_result_277' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'range_call_result_277', range_call_result_277)
        # Testing if the for loop is going to be iterated (line 96)
        # Testing the type of a for loop iterable (line 96)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 8), range_call_result_277)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 96, 8), range_call_result_277):
            # Getting the type of the for loop variable (line 96)
            for_loop_var_278 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 8), range_call_result_277)
            # Assigning a type to the variable 'k' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'k', for_loop_var_278)
            # SSA begins for a for statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 97):
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 97)
            k_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'k')
            # Getting the type of 'targets' (line 97)
            targets_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'targets')
            # Obtaining the member '__getitem__' of a type (line 97)
            getitem___281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 20), targets_280, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 97)
            subscript_call_result_282 = invoke(stypy.reporting.localization.Localization(__file__, 97, 20), getitem___281, k_279)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 97)
            k_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 41), 'k')
            # Getting the type of 'self' (line 97)
            self_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'self')
            # Obtaining the member 'ao' of a type (line 97)
            ao_285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 33), self_284, 'ao')
            # Obtaining the member '__getitem__' of a type (line 97)
            getitem___286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 33), ao_285, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 97)
            subscript_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 97, 33), getitem___286, k_283)
            
            # Applying the binary operator '-' (line 97)
            result_sub_288 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 20), '-', subscript_call_result_282, subscript_call_result_287)
            
            # Assigning a type to the variable 'error' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'error', result_sub_288)
            
            # Assigning a BinOp to a Subscript (line 98):
            
            # Call to dsigmoid(...): (line 98)
            # Processing the call arguments (line 98)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 98)
            k_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 48), 'k', False)
            # Getting the type of 'self' (line 98)
            self_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'self', False)
            # Obtaining the member 'ao' of a type (line 98)
            ao_292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 40), self_291, 'ao')
            # Obtaining the member '__getitem__' of a type (line 98)
            getitem___293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 40), ao_292, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 98)
            subscript_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 98, 40), getitem___293, k_290)
            
            # Processing the call keyword arguments (line 98)
            kwargs_295 = {}
            # Getting the type of 'dsigmoid' (line 98)
            dsigmoid_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'dsigmoid', False)
            # Calling dsigmoid(args, kwargs) (line 98)
            dsigmoid_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 98, 31), dsigmoid_289, *[subscript_call_result_294], **kwargs_295)
            
            # Getting the type of 'error' (line 98)
            error_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 54), 'error')
            # Applying the binary operator '*' (line 98)
            result_mul_298 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 31), '*', dsigmoid_call_result_296, error_297)
            
            # Getting the type of 'output_deltas' (line 98)
            output_deltas_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'output_deltas')
            # Getting the type of 'k' (line 98)
            k_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'k')
            # Storing an element on a container (line 98)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), output_deltas_299, (k_300, result_mul_298))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a BinOp to a Name (line 101):
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        float_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_301, float_302)
        
        # Getting the type of 'self' (line 101)
        self_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'self')
        # Obtaining the member 'nh' of a type (line 101)
        nh_304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 32), self_303, 'nh')
        # Applying the binary operator '*' (line 101)
        result_mul_305 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), '*', list_301, nh_304)
        
        # Assigning a type to the variable 'hidden_deltas' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'hidden_deltas', result_mul_305)
        
        
        # Call to range(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'self', False)
        # Obtaining the member 'nh' of a type (line 102)
        nh_308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), self_307, 'nh')
        # Processing the call keyword arguments (line 102)
        kwargs_309 = {}
        # Getting the type of 'range' (line 102)
        range_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'range', False)
        # Calling range(args, kwargs) (line 102)
        range_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), range_306, *[nh_308], **kwargs_309)
        
        # Assigning a type to the variable 'range_call_result_310' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'range_call_result_310', range_call_result_310)
        # Testing if the for loop is going to be iterated (line 102)
        # Testing the type of a for loop iterable (line 102)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 8), range_call_result_310)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 102, 8), range_call_result_310):
            # Getting the type of the for loop variable (line 102)
            for_loop_var_311 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 8), range_call_result_310)
            # Assigning a type to the variable 'j' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'j', for_loop_var_311)
            # SSA begins for a for statement (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 103):
            float_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'float')
            # Assigning a type to the variable 'error' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'error', float_312)
            
            
            # Call to range(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'self' (line 104)
            self_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'self', False)
            # Obtaining the member 'no' of a type (line 104)
            no_315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 27), self_314, 'no')
            # Processing the call keyword arguments (line 104)
            kwargs_316 = {}
            # Getting the type of 'range' (line 104)
            range_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'range', False)
            # Calling range(args, kwargs) (line 104)
            range_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 104, 21), range_313, *[no_315], **kwargs_316)
            
            # Assigning a type to the variable 'range_call_result_317' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'range_call_result_317', range_call_result_317)
            # Testing if the for loop is going to be iterated (line 104)
            # Testing the type of a for loop iterable (line 104)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 12), range_call_result_317)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 104, 12), range_call_result_317):
                # Getting the type of the for loop variable (line 104)
                for_loop_var_318 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 12), range_call_result_317)
                # Assigning a type to the variable 'k' (line 104)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'k', for_loop_var_318)
                # SSA begins for a for statement (line 104)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 105):
                # Getting the type of 'error' (line 105)
                error_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'error')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 105)
                k_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'k')
                # Getting the type of 'output_deltas' (line 105)
                output_deltas_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 32), 'output_deltas')
                # Obtaining the member '__getitem__' of a type (line 105)
                getitem___322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 32), output_deltas_321, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 105)
                subscript_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 105, 32), getitem___322, k_320)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 105)
                k_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 62), 'k')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 105)
                j_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 59), 'j')
                # Getting the type of 'self' (line 105)
                self_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 51), 'self')
                # Obtaining the member 'wo' of a type (line 105)
                wo_327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 51), self_326, 'wo')
                # Obtaining the member '__getitem__' of a type (line 105)
                getitem___328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 51), wo_327, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 105)
                subscript_call_result_329 = invoke(stypy.reporting.localization.Localization(__file__, 105, 51), getitem___328, j_325)
                
                # Obtaining the member '__getitem__' of a type (line 105)
                getitem___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 51), subscript_call_result_329, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 105)
                subscript_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 105, 51), getitem___330, k_324)
                
                # Applying the binary operator '*' (line 105)
                result_mul_332 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 32), '*', subscript_call_result_323, subscript_call_result_331)
                
                # Applying the binary operator '+' (line 105)
                result_add_333 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 24), '+', error_319, result_mul_332)
                
                # Assigning a type to the variable 'error' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'error', result_add_333)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Subscript (line 106):
            
            # Call to dsigmoid(...): (line 106)
            # Processing the call arguments (line 106)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 106)
            j_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), 'j', False)
            # Getting the type of 'self' (line 106)
            self_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'self', False)
            # Obtaining the member 'ah' of a type (line 106)
            ah_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), self_336, 'ah')
            # Obtaining the member '__getitem__' of a type (line 106)
            getitem___338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), ah_337, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 106)
            subscript_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 106, 40), getitem___338, j_335)
            
            # Processing the call keyword arguments (line 106)
            kwargs_340 = {}
            # Getting the type of 'dsigmoid' (line 106)
            dsigmoid_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 31), 'dsigmoid', False)
            # Calling dsigmoid(args, kwargs) (line 106)
            dsigmoid_call_result_341 = invoke(stypy.reporting.localization.Localization(__file__, 106, 31), dsigmoid_334, *[subscript_call_result_339], **kwargs_340)
            
            # Getting the type of 'error' (line 106)
            error_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 54), 'error')
            # Applying the binary operator '*' (line 106)
            result_mul_343 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 31), '*', dsigmoid_call_result_341, error_342)
            
            # Getting the type of 'hidden_deltas' (line 106)
            hidden_deltas_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'hidden_deltas')
            # Getting the type of 'j' (line 106)
            j_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'j')
            # Storing an element on a container (line 106)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 12), hidden_deltas_344, (j_345, result_mul_343))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'self', False)
        # Obtaining the member 'nh' of a type (line 109)
        nh_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 23), self_347, 'nh')
        # Processing the call keyword arguments (line 109)
        kwargs_349 = {}
        # Getting the type of 'range' (line 109)
        range_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'range', False)
        # Calling range(args, kwargs) (line 109)
        range_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 109, 17), range_346, *[nh_348], **kwargs_349)
        
        # Assigning a type to the variable 'range_call_result_350' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'range_call_result_350', range_call_result_350)
        # Testing if the for loop is going to be iterated (line 109)
        # Testing the type of a for loop iterable (line 109)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_350)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_350):
            # Getting the type of the for loop variable (line 109)
            for_loop_var_351 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_350)
            # Assigning a type to the variable 'j' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'j', for_loop_var_351)
            # SSA begins for a for statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 110)
            # Processing the call arguments (line 110)
            # Getting the type of 'self' (line 110)
            self_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'self', False)
            # Obtaining the member 'no' of a type (line 110)
            no_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 27), self_353, 'no')
            # Processing the call keyword arguments (line 110)
            kwargs_355 = {}
            # Getting the type of 'range' (line 110)
            range_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'range', False)
            # Calling range(args, kwargs) (line 110)
            range_call_result_356 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), range_352, *[no_354], **kwargs_355)
            
            # Assigning a type to the variable 'range_call_result_356' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'range_call_result_356', range_call_result_356)
            # Testing if the for loop is going to be iterated (line 110)
            # Testing the type of a for loop iterable (line 110)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 12), range_call_result_356)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 12), range_call_result_356):
                # Getting the type of the for loop variable (line 110)
                for_loop_var_357 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 12), range_call_result_356)
                # Assigning a type to the variable 'k' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'k', for_loop_var_357)
                # SSA begins for a for statement (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 111):
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 111)
                k_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 39), 'k')
                # Getting the type of 'output_deltas' (line 111)
                output_deltas_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'output_deltas')
                # Obtaining the member '__getitem__' of a type (line 111)
                getitem___360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), output_deltas_359, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                subscript_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 111, 25), getitem___360, k_358)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 111)
                j_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 52), 'j')
                # Getting the type of 'self' (line 111)
                self_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 44), 'self')
                # Obtaining the member 'ah' of a type (line 111)
                ah_364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 44), self_363, 'ah')
                # Obtaining the member '__getitem__' of a type (line 111)
                getitem___365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 44), ah_364, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                subscript_call_result_366 = invoke(stypy.reporting.localization.Localization(__file__, 111, 44), getitem___365, j_362)
                
                # Applying the binary operator '*' (line 111)
                result_mul_367 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 25), '*', subscript_call_result_361, subscript_call_result_366)
                
                # Assigning a type to the variable 'change' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'change', result_mul_367)
                
                # Assigning a BinOp to a Subscript (line 112):
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 112)
                k_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'k')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 112)
                j_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'j')
                # Getting the type of 'self' (line 112)
                self_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'self')
                # Obtaining the member 'wo' of a type (line 112)
                wo_371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), self_370, 'wo')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), wo_371, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 112, 32), getitem___372, j_369)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), subscript_call_result_373, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_375 = invoke(stypy.reporting.localization.Localization(__file__, 112, 32), getitem___374, k_368)
                
                # Getting the type of 'N' (line 112)
                N_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'N')
                # Getting the type of 'change' (line 112)
                change_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 52), 'change')
                # Applying the binary operator '*' (line 112)
                result_mul_378 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 48), '*', N_376, change_377)
                
                # Applying the binary operator '+' (line 112)
                result_add_379 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 32), '+', subscript_call_result_375, result_mul_378)
                
                # Getting the type of 'M' (line 112)
                M_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 61), 'M')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 112)
                k_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 76), 'k')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 112)
                j_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 73), 'j')
                # Getting the type of 'self' (line 112)
                self_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 65), 'self')
                # Obtaining the member 'co' of a type (line 112)
                co_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 65), self_383, 'co')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 65), co_384, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_386 = invoke(stypy.reporting.localization.Localization(__file__, 112, 65), getitem___385, j_382)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 65), subscript_call_result_386, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 112, 65), getitem___387, k_381)
                
                # Applying the binary operator '*' (line 112)
                result_mul_389 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 61), '*', M_380, subscript_call_result_388)
                
                # Applying the binary operator '+' (line 112)
                result_add_390 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 59), '+', result_add_379, result_mul_389)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 112)
                j_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'j')
                # Getting the type of 'self' (line 112)
                self_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'self')
                # Obtaining the member 'wo' of a type (line 112)
                wo_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), self_392, 'wo')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), wo_393, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_395 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), getitem___394, j_391)
                
                # Getting the type of 'k' (line 112)
                k_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'k')
                # Storing an element on a container (line 112)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 16), subscript_call_result_395, (k_396, result_add_390))
                
                # Assigning a Name to a Subscript (line 113):
                # Getting the type of 'change' (line 113)
                change_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), 'change')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 113)
                j_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'j')
                # Getting the type of 'self' (line 113)
                self_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'self')
                # Obtaining the member 'co' of a type (line 113)
                co_400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), self_399, 'co')
                # Obtaining the member '__getitem__' of a type (line 113)
                getitem___401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), co_400, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                subscript_call_result_402 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), getitem___401, j_398)
                
                # Getting the type of 'k' (line 113)
                k_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'k')
                # Storing an element on a container (line 113)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 16), subscript_call_result_402, (k_403, change_397))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'self', False)
        # Obtaining the member 'ni' of a type (line 117)
        ni_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 23), self_405, 'ni')
        # Processing the call keyword arguments (line 117)
        kwargs_407 = {}
        # Getting the type of 'range' (line 117)
        range_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'range', False)
        # Calling range(args, kwargs) (line 117)
        range_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), range_404, *[ni_406], **kwargs_407)
        
        # Assigning a type to the variable 'range_call_result_408' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'range_call_result_408', range_call_result_408)
        # Testing if the for loop is going to be iterated (line 117)
        # Testing the type of a for loop iterable (line 117)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 8), range_call_result_408)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 117, 8), range_call_result_408):
            # Getting the type of the for loop variable (line 117)
            for_loop_var_409 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 8), range_call_result_408)
            # Assigning a type to the variable 'i' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'i', for_loop_var_409)
            # SSA begins for a for statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'self' (line 118)
            self_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'self', False)
            # Obtaining the member 'nh' of a type (line 118)
            nh_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 27), self_411, 'nh')
            # Processing the call keyword arguments (line 118)
            kwargs_413 = {}
            # Getting the type of 'range' (line 118)
            range_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'range', False)
            # Calling range(args, kwargs) (line 118)
            range_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), range_410, *[nh_412], **kwargs_413)
            
            # Assigning a type to the variable 'range_call_result_414' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'range_call_result_414', range_call_result_414)
            # Testing if the for loop is going to be iterated (line 118)
            # Testing the type of a for loop iterable (line 118)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_414)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_414):
                # Getting the type of the for loop variable (line 118)
                for_loop_var_415 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_414)
                # Assigning a type to the variable 'j' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'j', for_loop_var_415)
                # SSA begins for a for statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 119):
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 119)
                j_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'j')
                # Getting the type of 'hidden_deltas' (line 119)
                hidden_deltas_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'hidden_deltas')
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), hidden_deltas_417, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 119, 25), getitem___418, j_416)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 119)
                i_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'i')
                # Getting the type of 'self' (line 119)
                self_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 44), 'self')
                # Obtaining the member 'ai' of a type (line 119)
                ai_422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 44), self_421, 'ai')
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 44), ai_422, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_424 = invoke(stypy.reporting.localization.Localization(__file__, 119, 44), getitem___423, i_420)
                
                # Applying the binary operator '*' (line 119)
                result_mul_425 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 25), '*', subscript_call_result_419, subscript_call_result_424)
                
                # Assigning a type to the variable 'change' (line 119)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'change', result_mul_425)
                
                # Assigning a BinOp to a Subscript (line 120):
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 120)
                j_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'j')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 120)
                i_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'i')
                # Getting the type of 'self' (line 120)
                self_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'self')
                # Obtaining the member 'wi' of a type (line 120)
                wi_429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), self_428, 'wi')
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), wi_429, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 120, 32), getitem___430, i_427)
                
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), subscript_call_result_431, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_433 = invoke(stypy.reporting.localization.Localization(__file__, 120, 32), getitem___432, j_426)
                
                # Getting the type of 'N' (line 120)
                N_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'N')
                # Getting the type of 'change' (line 120)
                change_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 52), 'change')
                # Applying the binary operator '*' (line 120)
                result_mul_436 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 48), '*', N_434, change_435)
                
                # Applying the binary operator '+' (line 120)
                result_add_437 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 32), '+', subscript_call_result_433, result_mul_436)
                
                # Getting the type of 'M' (line 120)
                M_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 61), 'M')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 120)
                j_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 76), 'j')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 120)
                i_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 73), 'i')
                # Getting the type of 'self' (line 120)
                self_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 65), 'self')
                # Obtaining the member 'ci' of a type (line 120)
                ci_442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 65), self_441, 'ci')
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 65), ci_442, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 120, 65), getitem___443, i_440)
                
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 65), subscript_call_result_444, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 120, 65), getitem___445, j_439)
                
                # Applying the binary operator '*' (line 120)
                result_mul_447 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 61), '*', M_438, subscript_call_result_446)
                
                # Applying the binary operator '+' (line 120)
                result_add_448 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 59), '+', result_add_437, result_mul_447)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 120)
                i_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'i')
                # Getting the type of 'self' (line 120)
                self_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'self')
                # Obtaining the member 'wi' of a type (line 120)
                wi_451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), self_450, 'wi')
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), wi_451, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), getitem___452, i_449)
                
                # Getting the type of 'j' (line 120)
                j_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'j')
                # Storing an element on a container (line 120)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 16), subscript_call_result_453, (j_454, result_add_448))
                
                # Assigning a Name to a Subscript (line 121):
                # Getting the type of 'change' (line 121)
                change_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'change')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 121)
                i_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'i')
                # Getting the type of 'self' (line 121)
                self_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'self')
                # Obtaining the member 'ci' of a type (line 121)
                ci_458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), self_457, 'ci')
                # Obtaining the member '__getitem__' of a type (line 121)
                getitem___459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), ci_458, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 121)
                subscript_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), getitem___459, i_456)
                
                # Getting the type of 'j' (line 121)
                j_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'j')
                # Storing an element on a container (line 121)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 16), subscript_call_result_460, (j_461, change_455))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Num to a Name (line 124):
        float_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'float')
        # Assigning a type to the variable 'error' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'error', float_462)
        
        
        # Call to range(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to len(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'targets' (line 125)
        targets_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'targets', False)
        # Processing the call keyword arguments (line 125)
        kwargs_466 = {}
        # Getting the type of 'len' (line 125)
        len_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'len', False)
        # Calling len(args, kwargs) (line 125)
        len_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 125, 23), len_464, *[targets_465], **kwargs_466)
        
        # Processing the call keyword arguments (line 125)
        kwargs_468 = {}
        # Getting the type of 'range' (line 125)
        range_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'range', False)
        # Calling range(args, kwargs) (line 125)
        range_call_result_469 = invoke(stypy.reporting.localization.Localization(__file__, 125, 17), range_463, *[len_call_result_467], **kwargs_468)
        
        # Assigning a type to the variable 'range_call_result_469' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'range_call_result_469', range_call_result_469)
        # Testing if the for loop is going to be iterated (line 125)
        # Testing the type of a for loop iterable (line 125)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 125, 8), range_call_result_469)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 125, 8), range_call_result_469):
            # Getting the type of the for loop variable (line 125)
            for_loop_var_470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 125, 8), range_call_result_469)
            # Assigning a type to the variable 'k' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'k', for_loop_var_470)
            # SSA begins for a for statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 126):
            # Getting the type of 'error' (line 126)
            error_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'error')
            float_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'float')
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 126)
            k_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 43), 'k')
            # Getting the type of 'targets' (line 126)
            targets_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 35), 'targets')
            # Obtaining the member '__getitem__' of a type (line 126)
            getitem___475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 35), targets_474, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 126)
            subscript_call_result_476 = invoke(stypy.reporting.localization.Localization(__file__, 126, 35), getitem___475, k_473)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 126)
            k_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 56), 'k')
            # Getting the type of 'self' (line 126)
            self_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 48), 'self')
            # Obtaining the member 'ao' of a type (line 126)
            ao_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 48), self_478, 'ao')
            # Obtaining the member '__getitem__' of a type (line 126)
            getitem___480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 48), ao_479, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 126)
            subscript_call_result_481 = invoke(stypy.reporting.localization.Localization(__file__, 126, 48), getitem___480, k_477)
            
            # Applying the binary operator '-' (line 126)
            result_sub_482 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 35), '-', subscript_call_result_476, subscript_call_result_481)
            
            int_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 63), 'int')
            # Applying the binary operator '**' (line 126)
            result_pow_484 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 34), '**', result_sub_482, int_483)
            
            # Applying the binary operator '*' (line 126)
            result_mul_485 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 28), '*', float_472, result_pow_484)
            
            # Applying the binary operator '+' (line 126)
            result_add_486 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 20), '+', error_471, result_mul_485)
            
            # Assigning a type to the variable 'error' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'error', result_add_486)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'error' (line 127)
        error_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'error')
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', error_487)
        
        # ################# End of 'backPropagate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'backPropagate' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'backPropagate'
        return stypy_return_type_488


    @norecursion
    def test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test'
        module_type_store = module_type_store.open_function_context('test', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NN.test.__dict__.__setitem__('stypy_localization', localization)
        NN.test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NN.test.__dict__.__setitem__('stypy_type_store', module_type_store)
        NN.test.__dict__.__setitem__('stypy_function_name', 'NN.test')
        NN.test.__dict__.__setitem__('stypy_param_names_list', ['patterns'])
        NN.test.__dict__.__setitem__('stypy_varargs_param_name', None)
        NN.test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NN.test.__dict__.__setitem__('stypy_call_defaults', defaults)
        NN.test.__dict__.__setitem__('stypy_call_varargs', varargs)
        NN.test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NN.test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NN.test', ['patterns'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test', localization, ['patterns'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test(...)' code ##################

        
        # Getting the type of 'patterns' (line 130)
        patterns_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'patterns')
        # Assigning a type to the variable 'patterns_489' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'patterns_489', patterns_489)
        # Testing if the for loop is going to be iterated (line 130)
        # Testing the type of a for loop iterable (line 130)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 130, 8), patterns_489)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 130, 8), patterns_489):
            # Getting the type of the for loop variable (line 130)
            for_loop_var_490 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 130, 8), patterns_489)
            # Assigning a type to the variable 'p' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'p', for_loop_var_490)
            # SSA begins for a for statement (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to update(...): (line 132)
            # Processing the call arguments (line 132)
            
            # Obtaining the type of the subscript
            int_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 26), 'int')
            # Getting the type of 'p' (line 132)
            p_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'p', False)
            # Obtaining the member '__getitem__' of a type (line 132)
            getitem___495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 24), p_494, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 132)
            subscript_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 132, 24), getitem___495, int_493)
            
            # Processing the call keyword arguments (line 132)
            kwargs_497 = {}
            # Getting the type of 'self' (line 132)
            self_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'self', False)
            # Obtaining the member 'update' of a type (line 132)
            update_492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), self_491, 'update')
            # Calling update(args, kwargs) (line 132)
            update_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), update_492, *[subscript_call_result_496], **kwargs_497)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_499)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test'
        return stypy_return_type_499


    @norecursion
    def weights(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'weights'
        module_type_store = module_type_store.open_function_context('weights', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NN.weights.__dict__.__setitem__('stypy_localization', localization)
        NN.weights.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NN.weights.__dict__.__setitem__('stypy_type_store', module_type_store)
        NN.weights.__dict__.__setitem__('stypy_function_name', 'NN.weights')
        NN.weights.__dict__.__setitem__('stypy_param_names_list', [])
        NN.weights.__dict__.__setitem__('stypy_varargs_param_name', None)
        NN.weights.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NN.weights.__dict__.__setitem__('stypy_call_defaults', defaults)
        NN.weights.__dict__.__setitem__('stypy_call_varargs', varargs)
        NN.weights.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NN.weights.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NN.weights', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'weights', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'weights(...)' code ##################

        
        
        # Call to range(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'self' (line 136)
        self_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'self', False)
        # Obtaining the member 'ni' of a type (line 136)
        ni_502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 23), self_501, 'ni')
        # Processing the call keyword arguments (line 136)
        kwargs_503 = {}
        # Getting the type of 'range' (line 136)
        range_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'range', False)
        # Calling range(args, kwargs) (line 136)
        range_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 136, 17), range_500, *[ni_502], **kwargs_503)
        
        # Assigning a type to the variable 'range_call_result_504' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'range_call_result_504', range_call_result_504)
        # Testing if the for loop is going to be iterated (line 136)
        # Testing the type of a for loop iterable (line 136)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 8), range_call_result_504)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 136, 8), range_call_result_504):
            # Getting the type of the for loop variable (line 136)
            for_loop_var_505 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 8), range_call_result_504)
            # Assigning a type to the variable 'i' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'i', for_loop_var_505)
            # SSA begins for a for statement (line 136)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            pass
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'self', False)
        # Obtaining the member 'nh' of a type (line 140)
        nh_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), self_507, 'nh')
        # Processing the call keyword arguments (line 140)
        kwargs_509 = {}
        # Getting the type of 'range' (line 140)
        range_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'range', False)
        # Calling range(args, kwargs) (line 140)
        range_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 140, 17), range_506, *[nh_508], **kwargs_509)
        
        # Assigning a type to the variable 'range_call_result_510' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'range_call_result_510', range_call_result_510)
        # Testing if the for loop is going to be iterated (line 140)
        # Testing the type of a for loop iterable (line 140)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 8), range_call_result_510)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 140, 8), range_call_result_510):
            # Getting the type of the for loop variable (line 140)
            for_loop_var_511 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 8), range_call_result_510)
            # Assigning a type to the variable 'j' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'j', for_loop_var_511)
            # SSA begins for a for statement (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            pass
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'weights(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'weights' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_512)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'weights'
        return stypy_return_type_512


    @norecursion
    def train(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 41), 'int')
        float_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 50), 'float')
        float_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 57), 'float')
        defaults = [int_513, float_514, float_515]
        # Create a new context for function 'train'
        module_type_store = module_type_store.open_function_context('train', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NN.train.__dict__.__setitem__('stypy_localization', localization)
        NN.train.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NN.train.__dict__.__setitem__('stypy_type_store', module_type_store)
        NN.train.__dict__.__setitem__('stypy_function_name', 'NN.train')
        NN.train.__dict__.__setitem__('stypy_param_names_list', ['patterns', 'iterations', 'N', 'M'])
        NN.train.__dict__.__setitem__('stypy_varargs_param_name', None)
        NN.train.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NN.train.__dict__.__setitem__('stypy_call_defaults', defaults)
        NN.train.__dict__.__setitem__('stypy_call_varargs', varargs)
        NN.train.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NN.train.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NN.train', ['patterns', 'iterations', 'N', 'M'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'train', localization, ['patterns', 'iterations', 'N', 'M'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'train(...)' code ##################

        
        
        # Call to xrange(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'iterations' (line 146)
        iterations_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'iterations', False)
        # Processing the call keyword arguments (line 146)
        kwargs_518 = {}
        # Getting the type of 'xrange' (line 146)
        xrange_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 146)
        xrange_call_result_519 = invoke(stypy.reporting.localization.Localization(__file__, 146, 17), xrange_516, *[iterations_517], **kwargs_518)
        
        # Assigning a type to the variable 'xrange_call_result_519' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'xrange_call_result_519', xrange_call_result_519)
        # Testing if the for loop is going to be iterated (line 146)
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), xrange_call_result_519)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 146, 8), xrange_call_result_519):
            # Getting the type of the for loop variable (line 146)
            for_loop_var_520 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), xrange_call_result_519)
            # Assigning a type to the variable 'i' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'i', for_loop_var_520)
            # SSA begins for a for statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 147):
            float_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'float')
            # Assigning a type to the variable 'error' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'error', float_521)
            
            # Getting the type of 'patterns' (line 148)
            patterns_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'patterns')
            # Assigning a type to the variable 'patterns_522' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'patterns_522', patterns_522)
            # Testing if the for loop is going to be iterated (line 148)
            # Testing the type of a for loop iterable (line 148)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 12), patterns_522)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 148, 12), patterns_522):
                # Getting the type of the for loop variable (line 148)
                for_loop_var_523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 12), patterns_522)
                # Assigning a type to the variable 'p' (line 148)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'p', for_loop_var_523)
                # SSA begins for a for statement (line 148)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 149):
                
                # Obtaining the type of the subscript
                int_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 27), 'int')
                # Getting the type of 'p' (line 149)
                p_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'p')
                # Obtaining the member '__getitem__' of a type (line 149)
                getitem___526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 25), p_525, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 149)
                subscript_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), getitem___526, int_524)
                
                # Assigning a type to the variable 'inputs' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'inputs', subscript_call_result_527)
                
                # Assigning a Subscript to a Name (line 150):
                
                # Obtaining the type of the subscript
                int_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 28), 'int')
                # Getting the type of 'p' (line 150)
                p_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'p')
                # Obtaining the member '__getitem__' of a type (line 150)
                getitem___530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 26), p_529, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 150)
                subscript_call_result_531 = invoke(stypy.reporting.localization.Localization(__file__, 150, 26), getitem___530, int_528)
                
                # Assigning a type to the variable 'targets' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'targets', subscript_call_result_531)
                
                # Call to update(...): (line 151)
                # Processing the call arguments (line 151)
                # Getting the type of 'inputs' (line 151)
                inputs_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'inputs', False)
                # Processing the call keyword arguments (line 151)
                kwargs_535 = {}
                # Getting the type of 'self' (line 151)
                self_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'self', False)
                # Obtaining the member 'update' of a type (line 151)
                update_533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 16), self_532, 'update')
                # Calling update(args, kwargs) (line 151)
                update_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 151, 16), update_533, *[inputs_534], **kwargs_535)
                
                
                # Assigning a BinOp to a Name (line 152):
                # Getting the type of 'error' (line 152)
                error_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'error')
                
                # Call to backPropagate(...): (line 152)
                # Processing the call arguments (line 152)
                # Getting the type of 'targets' (line 152)
                targets_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 51), 'targets', False)
                # Getting the type of 'N' (line 152)
                N_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 60), 'N', False)
                # Getting the type of 'M' (line 152)
                M_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 63), 'M', False)
                # Processing the call keyword arguments (line 152)
                kwargs_543 = {}
                # Getting the type of 'self' (line 152)
                self_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'self', False)
                # Obtaining the member 'backPropagate' of a type (line 152)
                backPropagate_539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 32), self_538, 'backPropagate')
                # Calling backPropagate(args, kwargs) (line 152)
                backPropagate_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 152, 32), backPropagate_539, *[targets_540, N_541, M_542], **kwargs_543)
                
                # Applying the binary operator '+' (line 152)
                result_add_545 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 24), '+', error_537, backPropagate_call_result_544)
                
                # Assigning a type to the variable 'error' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'error', result_add_545)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'i' (line 153)
            i_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'i')
            int_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 19), 'int')
            # Applying the binary operator '%' (line 153)
            result_mod_548 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 15), '%', i_546, int_547)
            
            int_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 27), 'int')
            # Applying the binary operator '==' (line 153)
            result_eq_550 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 15), '==', result_mod_548, int_549)
            
            # Testing if the type of an if condition is none (line 153)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 12), result_eq_550):
                pass
            else:
                
                # Testing the type of an if condition (line 153)
                if_condition_551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), result_eq_550)
                # Assigning a type to the variable 'if_condition_551' (line 153)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_551', if_condition_551)
                # SSA begins for if statement (line 153)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA join for if statement (line 153)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'train(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'train' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'train'
        return stypy_return_type_552


# Assigning a type to the variable 'NN' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'NN', NN)

@norecursion
def demo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'demo'
    module_type_store = module_type_store.open_function_context('demo', 157, 0, False)
    
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

    
    # Assigning a List to a Name (line 159):
    
    # Obtaining an instance of the builtin type 'list' (line 159)
    list_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 159)
    # Adding element type (line 159)
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    # Adding element type (line 160)
    int_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 9), list_555, int_556)
    # Adding element type (line 160)
    int_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 9), list_555, int_557)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 8), list_554, list_555)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    # Adding element type (line 160)
    int_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 17), list_558, int_559)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 8), list_554, list_558)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 10), list_553, list_554)
    # Adding element type (line 159)
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 9), list_561, int_562)
    # Adding element type (line 161)
    int_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 9), list_561, int_563)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 8), list_560, list_561)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    int_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 17), list_564, int_565)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 8), list_560, list_564)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 10), list_553, list_560)
    # Adding element type (line 159)
    
    # Obtaining an instance of the builtin type 'list' (line 162)
    list_566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 162)
    # Adding element type (line 162)
    
    # Obtaining an instance of the builtin type 'list' (line 162)
    list_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 162)
    # Adding element type (line 162)
    int_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 9), list_567, int_568)
    # Adding element type (line 162)
    int_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 9), list_567, int_569)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), list_566, list_567)
    # Adding element type (line 162)
    
    # Obtaining an instance of the builtin type 'list' (line 162)
    list_570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 162)
    # Adding element type (line 162)
    int_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 17), list_570, int_571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), list_566, list_570)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 10), list_553, list_566)
    # Adding element type (line 159)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    int_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 9), list_573, int_574)
    # Adding element type (line 163)
    int_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 9), list_573, int_575)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 8), list_572, list_573)
    # Adding element type (line 163)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    int_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 17), list_576, int_577)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 8), list_572, list_576)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 10), list_553, list_572)
    
    # Assigning a type to the variable 'pat' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'pat', list_553)
    
    # Assigning a Call to a Name (line 167):
    
    # Call to NN(...): (line 167)
    # Processing the call arguments (line 167)
    int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 11), 'int')
    int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 14), 'int')
    int_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_582 = {}
    # Getting the type of 'NN' (line 167)
    NN_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'NN', False)
    # Calling NN(args, kwargs) (line 167)
    NN_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), NN_578, *[int_579, int_580, int_581], **kwargs_582)
    
    # Assigning a type to the variable 'n' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'n', NN_call_result_583)
    
    # Call to train(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'pat' (line 169)
    pat_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'pat', False)
    # Processing the call keyword arguments (line 169)
    kwargs_587 = {}
    # Getting the type of 'n' (line 169)
    n_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'n', False)
    # Obtaining the member 'train' of a type (line 169)
    train_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 4), n_584, 'train')
    # Calling train(args, kwargs) (line 169)
    train_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 169, 4), train_585, *[pat_586], **kwargs_587)
    
    
    # Call to test(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'pat' (line 171)
    pat_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'pat', False)
    # Processing the call keyword arguments (line 171)
    kwargs_592 = {}
    # Getting the type of 'n' (line 171)
    n_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'n', False)
    # Obtaining the member 'test' of a type (line 171)
    test_590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 4), n_589, 'test')
    # Calling test(args, kwargs) (line 171)
    test_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 171, 4), test_590, *[pat_591], **kwargs_592)
    
    
    # ################# End of 'demo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'demo' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_594)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'demo'
    return stypy_return_type_594

# Assigning a type to the variable 'demo' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'demo', demo)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 174, 0, False)
    
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

    
    # Call to demo(...): (line 175)
    # Processing the call keyword arguments (line 175)
    kwargs_596 = {}
    # Getting the type of 'demo' (line 175)
    demo_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'demo', False)
    # Calling demo(args, kwargs) (line 175)
    demo_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), demo_595, *[], **kwargs_596)
    
    # Getting the type of 'True' (line 176)
    True_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type', True_598)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 174)
    stypy_return_type_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_599)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_599

# Assigning a type to the variable 'run' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'run', run)

# Call to run(...): (line 179)
# Processing the call keyword arguments (line 179)
kwargs_601 = {}
# Getting the type of 'run' (line 179)
run_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'run', False)
# Calling run(args, kwargs) (line 179)
run_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 179, 0), run_600, *[], **kwargs_601)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

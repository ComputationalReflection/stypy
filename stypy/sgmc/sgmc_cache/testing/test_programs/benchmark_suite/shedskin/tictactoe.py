
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # (c) Peter Goodspeed
2: # --- coriolinus@gmail.com
3: 
4: from math import exp
5: 
6: 
7: # functions
8: def sigmoid(x):
9:     return float(1) / (1 + exp(-x))
10: 
11: 
12: def sig(x, xshift=0, xcompress=1):
13:     return 0 + (1 * sigmoid(xcompress * (x - xshift)))
14: 
15: 
16: # exceptions
17: class SpaceNotEmpty(Exception):
18:     pass
19: 
20: 
21: class MultiVictory(Exception):
22:     def __init__(self, victorslist):
23:         self.victors = victorslist
24: 
25: 
26: # classes
27: class rectBoard(object):
28:     def __init__(self, edge=3):
29:         self.edge = edge
30:         self.__board = [edge * [0] for i in xrange(edge)]
31:         self.__empty = edge ** 2
32: 
33:     def assign(self, row, col, value):
34:         if (self.__board[row][col] == 0):
35:             self.__board[row][col] = value
36:             self.__empty -= 1
37:         else:
38:             raise SpaceNotEmpty()
39: 
40:     def isfull(self):
41:         return self.__empty == 0
42: 
43:     # def valueof(self, row, col):
44:     #        return self.__board[row][col]
45: 
46:     def isvictory(self):
47:         victors = []
48:         # examine rows
49:         for row in self.__board:
50:             if len(set(row)) == 1:
51:                 if row[0] != 0: victors.append(row[0])
52: 
53:         # examine cols
54:         for i in xrange(self.edge):
55:             col = [row[i] for row in self.__board]
56:             if len(set(col)) == 1:
57:                 if col[0] != 0: victors.append(col[0])
58: 
59:         # examine diagonals
60:         # left diagonal
61:         ld = []
62:         for i in xrange(self.edge): ld.append(self.__board[i][i])
63:         if len(set(ld)) == 1:
64:             if ld[0] != 0: victors.append(ld[0])
65: 
66:         # right diagonal
67:         rd = []
68:         for i in xrange(self.edge): rd.append(self.__board[i][self.edge - (1 + i)])
69:         if len(set(rd)) == 1:
70:             if rd[0] != 0: victors.append(rd[0])
71: 
72:         # return
73:         if len(victors) == 0:
74:             return 0
75:         if len(set(victors)) > 1:
76:             raise MultiVictory(set(victors))
77:         return victors[0]
78: 
79:     def __str__(self):
80:         ret = ""
81:         for row in xrange(self.edge):
82:             if row != 0:
83:                 ret += "\n"
84:                 for i in xrange(self.edge):
85:                     if i != 0: ret += '+'
86:                     ret += "---"
87:                 ret += "\n"
88:             ret += " "
89:             for col in xrange(self.edge):
90:                 if col != 0: ret += " | "
91:                 if self.__board[row][col] == 0:
92:                     ret += ' '
93:                 else:
94:                     ret += str(self.__board[row][col])
95:         return ret
96: 
97:     def doRow(self, fields, indices, player, scores):
98:         players = set(fields).difference(set([0]))
99: 
100:         if (len(players) == 1):
101:             if list(players)[0] == player:
102:                 for rown, coln in indices:
103:                     scores[rown][coln] += 15 * sig(fields.count(player) / float(self.edge), .5, 10)
104:             else:
105:                 for rown, coln in indices:
106:                     scores[rown][coln] += 15 * fields.count(list(players)[0]) / float(self.edge)
107: 
108:     def makeAImove(self, player):
109:         scores = [self.edge * [0] for i in xrange(self.edge)]
110: 
111:         for rown in xrange(self.edge):
112:             row = self.__board[rown]
113:             self.doRow(row, [(rown, i) for i in xrange(self.edge)], player, scores)
114: 
115:         for coln in xrange(self.edge):
116:             col = [row[coln] for row in self.__board]
117:             self.doRow(col, [(i, coln) for i in xrange(self.edge)], player, scores)
118: 
119:         indices = [(i, i) for i in xrange(self.edge)]
120:         ld = [self.__board[i][i] for i in xrange(self.edge)]
121:         self.doRow(ld, indices, player, scores)
122:         # also, because diagonals are just more useful
123:         for rown, coln in indices:
124:             scores[rown][coln] += 1
125: 
126:         # now, we do the same for right diagonals
127:         indices = [(i, (self.edge - 1) - i) for i in xrange(self.edge)]
128:         rd = [self.__board[i][(self.edge - 1) - i] for i in xrange(self.edge)]
129:         self.doRow(rd, indices, player, scores)
130:         # also, because diagonals are just more useful
131:         for rown, coln in indices:
132:             scores[rown][coln] += 1
133: 
134:         scorelist = []
135:         for rown in xrange(self.edge):
136:             for coln in xrange(self.edge):
137:                 if (self.__board[rown][coln] == 0):
138:                     scorelist.append((scores[rown][coln], (rown, coln)))
139:         scorelist.sort()
140:         scorelist.reverse()
141:         # print scorelist
142:         scorelist = [x for x in scorelist if x[0] == scorelist[0][0]]
143: 
144:         # return random.choice([(x[1], x[2]) for x in scorelist])
145: 
146:         # scorelist = [(random.random(), x[1],x[2]) for x in scorelist]
147:         # scorelist.sort()
148: 
149:         return (scorelist[0][1][0], scorelist[0][1][1])
150: 
151: 
152: def aigame(size=10, turn=1, players=2):
153:     b = rectBoard(size)
154: 
155:     while ((not b.isfull()) and (b.isvictory() == 0)):
156:         if (turn == 1):
157:             # player turn
158:             # print
159:             # print b
160:             r, c = b.makeAImove(turn)
161:             b.assign(r, c, 1)
162:             turn = 2
163:         else:
164:             # computer turn
165:             r, c = b.makeAImove(turn)
166:             b.assign(r, c, turn)
167:             if (turn == players):
168:                 turn = 1
169:             else:
170:                 turn += 1
171:     ##        print
172:     ##        print b.__str__()
173:     ##        print
174:     if (b.isvictory() == 0):
175:         pass  # print "Board is full! Draw!"
176:     else:
177:         pass  # print "Victory for player "+str(b.isvictory())+"!"
178: 
179: 
180: def run():
181:     for i in range(10):
182:         aigame()
183:     return True
184: 
185: 
186: run()
187: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from math import exp' statement (line 4)
try:
    from math import exp

except:
    exp = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'math', None, module_type_store, ['exp'], [exp])


@norecursion
def sigmoid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sigmoid'
    module_type_store = module_type_store.open_function_context('sigmoid', 8, 0, False)
    
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

    
    # Call to float(...): (line 9)
    # Processing the call arguments (line 9)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
    # Processing the call keyword arguments (line 9)
    kwargs_9 = {}
    # Getting the type of 'float' (line 9)
    float_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'float', False)
    # Calling float(args, kwargs) (line 9)
    float_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), float_7, *[int_8], **kwargs_9)
    
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'int')
    
    # Call to exp(...): (line 9)
    # Processing the call arguments (line 9)
    
    # Getting the type of 'x' (line 9)
    x_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 32), 'x', False)
    # Applying the 'usub' unary operator (line 9)
    result___neg___14 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 31), 'usub', x_13)
    
    # Processing the call keyword arguments (line 9)
    kwargs_15 = {}
    # Getting the type of 'exp' (line 9)
    exp_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 27), 'exp', False)
    # Calling exp(args, kwargs) (line 9)
    exp_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 9, 27), exp_12, *[result___neg___14], **kwargs_15)
    
    # Applying the binary operator '+' (line 9)
    result_add_17 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 23), '+', int_11, exp_call_result_16)
    
    # Applying the binary operator 'div' (line 9)
    result_div_18 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 11), 'div', float_call_result_10, result_add_17)
    
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', result_div_18)
    
    # ################# End of 'sigmoid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sigmoid' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sigmoid'
    return stypy_return_type_19

# Assigning a type to the variable 'sigmoid' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'sigmoid', sigmoid)

@norecursion
def sig(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
    defaults = [int_20, int_21]
    # Create a new context for function 'sig'
    module_type_store = module_type_store.open_function_context('sig', 12, 0, False)
    
    # Passed parameters checking function
    sig.stypy_localization = localization
    sig.stypy_type_of_self = None
    sig.stypy_type_store = module_type_store
    sig.stypy_function_name = 'sig'
    sig.stypy_param_names_list = ['x', 'xshift', 'xcompress']
    sig.stypy_varargs_param_name = None
    sig.stypy_kwargs_param_name = None
    sig.stypy_call_defaults = defaults
    sig.stypy_call_varargs = varargs
    sig.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sig', ['x', 'xshift', 'xcompress'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sig', localization, ['x', 'xshift', 'xcompress'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sig(...)' code ##################

    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'int')
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
    
    # Call to sigmoid(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'xcompress' (line 13)
    xcompress_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'xcompress', False)
    # Getting the type of 'x' (line 13)
    x_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 41), 'x', False)
    # Getting the type of 'xshift' (line 13)
    xshift_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 45), 'xshift', False)
    # Applying the binary operator '-' (line 13)
    result_sub_28 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 41), '-', x_26, xshift_27)
    
    # Applying the binary operator '*' (line 13)
    result_mul_29 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 28), '*', xcompress_25, result_sub_28)
    
    # Processing the call keyword arguments (line 13)
    kwargs_30 = {}
    # Getting the type of 'sigmoid' (line 13)
    sigmoid_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'sigmoid', False)
    # Calling sigmoid(args, kwargs) (line 13)
    sigmoid_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 13, 20), sigmoid_24, *[result_mul_29], **kwargs_30)
    
    # Applying the binary operator '*' (line 13)
    result_mul_32 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 16), '*', int_23, sigmoid_call_result_31)
    
    # Applying the binary operator '+' (line 13)
    result_add_33 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 11), '+', int_22, result_mul_32)
    
    # Assigning a type to the variable 'stypy_return_type' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type', result_add_33)
    
    # ################# End of 'sig(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sig' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sig'
    return stypy_return_type_34

# Assigning a type to the variable 'sig' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'sig', sig)
# Declaration of the 'SpaceNotEmpty' class
# Getting the type of 'Exception' (line 17)
Exception_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'Exception')

class SpaceNotEmpty(Exception_35, ):
    pass

# Assigning a type to the variable 'SpaceNotEmpty' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'SpaceNotEmpty', SpaceNotEmpty)
# Declaration of the 'MultiVictory' class
# Getting the type of 'Exception' (line 21)
Exception_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'Exception')

class MultiVictory(Exception_36, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MultiVictory.__init__', ['victorslist'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['victorslist'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 23):
        
        # Assigning a Name to a Attribute (line 23):
        # Getting the type of 'victorslist' (line 23)
        victorslist_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'victorslist')
        # Getting the type of 'self' (line 23)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'victors' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_38, 'victors', victorslist_37)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MultiVictory' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'MultiVictory', MultiVictory)
# Declaration of the 'rectBoard' class

class rectBoard(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
        defaults = [int_39]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'rectBoard.__init__', ['edge'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['edge'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 29):
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'edge' (line 29)
        edge_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'edge')
        # Getting the type of 'self' (line 29)
        self_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'edge' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_41, 'edge', edge_40)
        
        # Assigning a ListComp to a Attribute (line 30):
        
        # Assigning a ListComp to a Attribute (line 30):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'edge' (line 30)
        edge_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 51), 'edge', False)
        # Processing the call keyword arguments (line 30)
        kwargs_48 = {}
        # Getting the type of 'xrange' (line 30)
        xrange_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 44), 'xrange', False)
        # Calling xrange(args, kwargs) (line 30)
        xrange_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 30, 44), xrange_46, *[edge_47], **kwargs_48)
        
        comprehension_50 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), xrange_call_result_49)
        # Assigning a type to the variable 'i' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'i', comprehension_50)
        # Getting the type of 'edge' (line 30)
        edge_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'edge')
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_43, int_44)
        
        # Applying the binary operator '*' (line 30)
        result_mul_45 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 24), '*', edge_42, list_43)
        
        list_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), list_51, result_mul_45)
        # Getting the type of 'self' (line 30)
        self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member '__board' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_52, '__board', list_51)
        
        # Assigning a BinOp to a Attribute (line 31):
        
        # Assigning a BinOp to a Attribute (line 31):
        # Getting the type of 'edge' (line 31)
        edge_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'edge')
        int_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'int')
        # Applying the binary operator '**' (line 31)
        result_pow_55 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 23), '**', edge_53, int_54)
        
        # Getting the type of 'self' (line 31)
        self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member '__empty' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_56, '__empty', result_pow_55)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def assign(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assign'
        module_type_store = module_type_store.open_function_context('assign', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        rectBoard.assign.__dict__.__setitem__('stypy_localization', localization)
        rectBoard.assign.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        rectBoard.assign.__dict__.__setitem__('stypy_type_store', module_type_store)
        rectBoard.assign.__dict__.__setitem__('stypy_function_name', 'rectBoard.assign')
        rectBoard.assign.__dict__.__setitem__('stypy_param_names_list', ['row', 'col', 'value'])
        rectBoard.assign.__dict__.__setitem__('stypy_varargs_param_name', None)
        rectBoard.assign.__dict__.__setitem__('stypy_kwargs_param_name', None)
        rectBoard.assign.__dict__.__setitem__('stypy_call_defaults', defaults)
        rectBoard.assign.__dict__.__setitem__('stypy_call_varargs', varargs)
        rectBoard.assign.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        rectBoard.assign.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'rectBoard.assign', ['row', 'col', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assign', localization, ['row', 'col', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assign(...)' code ##################

        
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 34)
        col_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'col')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 34)
        row_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'row')
        # Getting the type of 'self' (line 34)
        self_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self')
        # Obtaining the member '__board' of a type (line 34)
        board_60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_59, '__board')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), board_60, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), getitem___61, row_58)
        
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), subscript_call_result_62, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), getitem___63, col_57)
        
        int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 38), 'int')
        # Applying the binary operator '==' (line 34)
        result_eq_66 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), '==', subscript_call_result_64, int_65)
        
        # Testing if the type of an if condition is none (line 34)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 8), result_eq_66):
            
            # Call to SpaceNotEmpty(...): (line 38)
            # Processing the call keyword arguments (line 38)
            kwargs_81 = {}
            # Getting the type of 'SpaceNotEmpty' (line 38)
            SpaceNotEmpty_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'SpaceNotEmpty', False)
            # Calling SpaceNotEmpty(args, kwargs) (line 38)
            SpaceNotEmpty_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), SpaceNotEmpty_80, *[], **kwargs_81)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 38, 12), SpaceNotEmpty_call_result_82, 'raise parameter', BaseException)
        else:
            
            # Testing the type of an if condition (line 34)
            if_condition_67 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), result_eq_66)
            # Assigning a type to the variable 'if_condition_67' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_67', if_condition_67)
            # SSA begins for if statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Subscript (line 35):
            
            # Assigning a Name to a Subscript (line 35):
            # Getting the type of 'value' (line 35)
            value_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'value')
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 35)
            row_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'row')
            # Getting the type of 'self' (line 35)
            self_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self')
            # Obtaining the member '__board' of a type (line 35)
            board_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_70, '__board')
            # Obtaining the member '__getitem__' of a type (line 35)
            getitem___72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), board_71, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 35)
            subscript_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), getitem___72, row_69)
            
            # Getting the type of 'col' (line 35)
            col_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 30), 'col')
            # Storing an element on a container (line 35)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 12), subscript_call_result_73, (col_74, value_68))
            
            # Getting the type of 'self' (line 36)
            self_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
            # Obtaining the member '__empty' of a type (line 36)
            empty_76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_75, '__empty')
            int_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'int')
            # Applying the binary operator '-=' (line 36)
            result_isub_78 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '-=', empty_76, int_77)
            # Getting the type of 'self' (line 36)
            self_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
            # Setting the type of the member '__empty' of a type (line 36)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_79, '__empty', result_isub_78)
            
            # SSA branch for the else part of an if statement (line 34)
            module_type_store.open_ssa_branch('else')
            
            # Call to SpaceNotEmpty(...): (line 38)
            # Processing the call keyword arguments (line 38)
            kwargs_81 = {}
            # Getting the type of 'SpaceNotEmpty' (line 38)
            SpaceNotEmpty_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'SpaceNotEmpty', False)
            # Calling SpaceNotEmpty(args, kwargs) (line 38)
            SpaceNotEmpty_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), SpaceNotEmpty_80, *[], **kwargs_81)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 38, 12), SpaceNotEmpty_call_result_82, 'raise parameter', BaseException)
            # SSA join for if statement (line 34)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'assign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assign' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_83)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assign'
        return stypy_return_type_83


    @norecursion
    def isfull(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isfull'
        module_type_store = module_type_store.open_function_context('isfull', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        rectBoard.isfull.__dict__.__setitem__('stypy_localization', localization)
        rectBoard.isfull.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        rectBoard.isfull.__dict__.__setitem__('stypy_type_store', module_type_store)
        rectBoard.isfull.__dict__.__setitem__('stypy_function_name', 'rectBoard.isfull')
        rectBoard.isfull.__dict__.__setitem__('stypy_param_names_list', [])
        rectBoard.isfull.__dict__.__setitem__('stypy_varargs_param_name', None)
        rectBoard.isfull.__dict__.__setitem__('stypy_kwargs_param_name', None)
        rectBoard.isfull.__dict__.__setitem__('stypy_call_defaults', defaults)
        rectBoard.isfull.__dict__.__setitem__('stypy_call_varargs', varargs)
        rectBoard.isfull.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        rectBoard.isfull.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'rectBoard.isfull', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isfull', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isfull(...)' code ##################

        
        # Getting the type of 'self' (line 41)
        self_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'self')
        # Obtaining the member '__empty' of a type (line 41)
        empty_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), self_84, '__empty')
        int_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'int')
        # Applying the binary operator '==' (line 41)
        result_eq_87 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '==', empty_85, int_86)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', result_eq_87)
        
        # ################# End of 'isfull(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isfull' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_88)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isfull'
        return stypy_return_type_88


    @norecursion
    def isvictory(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isvictory'
        module_type_store = module_type_store.open_function_context('isvictory', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        rectBoard.isvictory.__dict__.__setitem__('stypy_localization', localization)
        rectBoard.isvictory.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        rectBoard.isvictory.__dict__.__setitem__('stypy_type_store', module_type_store)
        rectBoard.isvictory.__dict__.__setitem__('stypy_function_name', 'rectBoard.isvictory')
        rectBoard.isvictory.__dict__.__setitem__('stypy_param_names_list', [])
        rectBoard.isvictory.__dict__.__setitem__('stypy_varargs_param_name', None)
        rectBoard.isvictory.__dict__.__setitem__('stypy_kwargs_param_name', None)
        rectBoard.isvictory.__dict__.__setitem__('stypy_call_defaults', defaults)
        rectBoard.isvictory.__dict__.__setitem__('stypy_call_varargs', varargs)
        rectBoard.isvictory.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        rectBoard.isvictory.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'rectBoard.isvictory', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isvictory', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isvictory(...)' code ##################

        
        # Assigning a List to a Name (line 47):
        
        # Assigning a List to a Name (line 47):
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        
        # Assigning a type to the variable 'victors' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'victors', list_89)
        
        # Getting the type of 'self' (line 49)
        self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'self')
        # Obtaining the member '__board' of a type (line 49)
        board_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), self_90, '__board')
        # Assigning a type to the variable 'board_91' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'board_91', board_91)
        # Testing if the for loop is going to be iterated (line 49)
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), board_91)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 49, 8), board_91):
            # Getting the type of the for loop variable (line 49)
            for_loop_var_92 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), board_91)
            # Assigning a type to the variable 'row' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'row', for_loop_var_92)
            # SSA begins for a for statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to len(...): (line 50)
            # Processing the call arguments (line 50)
            
            # Call to set(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'row' (line 50)
            row_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 'row', False)
            # Processing the call keyword arguments (line 50)
            kwargs_96 = {}
            # Getting the type of 'set' (line 50)
            set_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'set', False)
            # Calling set(args, kwargs) (line 50)
            set_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 50, 19), set_94, *[row_95], **kwargs_96)
            
            # Processing the call keyword arguments (line 50)
            kwargs_98 = {}
            # Getting the type of 'len' (line 50)
            len_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'len', False)
            # Calling len(args, kwargs) (line 50)
            len_call_result_99 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), len_93, *[set_call_result_97], **kwargs_98)
            
            int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 32), 'int')
            # Applying the binary operator '==' (line 50)
            result_eq_101 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '==', len_call_result_99, int_100)
            
            # Testing if the type of an if condition is none (line 50)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 50, 12), result_eq_101):
                pass
            else:
                
                # Testing the type of an if condition (line 50)
                if_condition_102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), result_eq_101)
                # Assigning a type to the variable 'if_condition_102' (line 50)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_condition_102', if_condition_102)
                # SSA begins for if statement (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Obtaining the type of the subscript
                int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                # Getting the type of 'row' (line 51)
                row_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'row')
                # Obtaining the member '__getitem__' of a type (line 51)
                getitem___105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 19), row_104, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                subscript_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), getitem___105, int_103)
                
                int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'int')
                # Applying the binary operator '!=' (line 51)
                result_ne_108 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), '!=', subscript_call_result_106, int_107)
                
                # Testing if the type of an if condition is none (line 51)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 51, 16), result_ne_108):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 51)
                    if_condition_109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 16), result_ne_108)
                    # Assigning a type to the variable 'if_condition_109' (line 51)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'if_condition_109', if_condition_109)
                    # SSA begins for if statement (line 51)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 51)
                    # Processing the call arguments (line 51)
                    
                    # Obtaining the type of the subscript
                    int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 51), 'int')
                    # Getting the type of 'row' (line 51)
                    row_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 47), 'row', False)
                    # Obtaining the member '__getitem__' of a type (line 51)
                    getitem___114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 47), row_113, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                    subscript_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 51, 47), getitem___114, int_112)
                    
                    # Processing the call keyword arguments (line 51)
                    kwargs_116 = {}
                    # Getting the type of 'victors' (line 51)
                    victors_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'victors', False)
                    # Obtaining the member 'append' of a type (line 51)
                    append_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 32), victors_110, 'append')
                    # Calling append(args, kwargs) (line 51)
                    append_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 51, 32), append_111, *[subscript_call_result_115], **kwargs_116)
                    
                    # SSA join for if statement (line 51)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 50)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to xrange(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'self', False)
        # Obtaining the member 'edge' of a type (line 54)
        edge_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), self_119, 'edge')
        # Processing the call keyword arguments (line 54)
        kwargs_121 = {}
        # Getting the type of 'xrange' (line 54)
        xrange_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 54)
        xrange_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), xrange_118, *[edge_120], **kwargs_121)
        
        # Assigning a type to the variable 'xrange_call_result_122' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'xrange_call_result_122', xrange_call_result_122)
        # Testing if the for loop is going to be iterated (line 54)
        # Testing the type of a for loop iterable (line 54)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_122)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_122):
            # Getting the type of the for loop variable (line 54)
            for_loop_var_123 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_122)
            # Assigning a type to the variable 'i' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'i', for_loop_var_123)
            # SSA begins for a for statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a ListComp to a Name (line 55):
            
            # Assigning a ListComp to a Name (line 55):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'self' (line 55)
            self_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 37), 'self')
            # Obtaining the member '__board' of a type (line 55)
            board_129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 37), self_128, '__board')
            comprehension_130 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), board_129)
            # Assigning a type to the variable 'row' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'row', comprehension_130)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 55)
            i_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'i')
            # Getting the type of 'row' (line 55)
            row_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'row')
            # Obtaining the member '__getitem__' of a type (line 55)
            getitem___126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 19), row_125, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 55)
            subscript_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), getitem___126, i_124)
            
            list_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), list_131, subscript_call_result_127)
            # Assigning a type to the variable 'col' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'col', list_131)
            
            
            # Call to len(...): (line 56)
            # Processing the call arguments (line 56)
            
            # Call to set(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'col' (line 56)
            col_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'col', False)
            # Processing the call keyword arguments (line 56)
            kwargs_135 = {}
            # Getting the type of 'set' (line 56)
            set_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'set', False)
            # Calling set(args, kwargs) (line 56)
            set_call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), set_133, *[col_134], **kwargs_135)
            
            # Processing the call keyword arguments (line 56)
            kwargs_137 = {}
            # Getting the type of 'len' (line 56)
            len_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'len', False)
            # Calling len(args, kwargs) (line 56)
            len_call_result_138 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), len_132, *[set_call_result_136], **kwargs_137)
            
            int_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 32), 'int')
            # Applying the binary operator '==' (line 56)
            result_eq_140 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), '==', len_call_result_138, int_139)
            
            # Testing if the type of an if condition is none (line 56)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 12), result_eq_140):
                pass
            else:
                
                # Testing the type of an if condition (line 56)
                if_condition_141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), result_eq_140)
                # Assigning a type to the variable 'if_condition_141' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_141', if_condition_141)
                # SSA begins for if statement (line 56)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Obtaining the type of the subscript
                int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'int')
                # Getting the type of 'col' (line 57)
                col_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'col')
                # Obtaining the member '__getitem__' of a type (line 57)
                getitem___144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 19), col_143, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 57)
                subscript_call_result_145 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), getitem___144, int_142)
                
                int_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'int')
                # Applying the binary operator '!=' (line 57)
                result_ne_147 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 19), '!=', subscript_call_result_145, int_146)
                
                # Testing if the type of an if condition is none (line 57)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 16), result_ne_147):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 57)
                    if_condition_148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 16), result_ne_147)
                    # Assigning a type to the variable 'if_condition_148' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'if_condition_148', if_condition_148)
                    # SSA begins for if statement (line 57)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 57)
                    # Processing the call arguments (line 57)
                    
                    # Obtaining the type of the subscript
                    int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 51), 'int')
                    # Getting the type of 'col' (line 57)
                    col_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 47), 'col', False)
                    # Obtaining the member '__getitem__' of a type (line 57)
                    getitem___153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 47), col_152, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
                    subscript_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 57, 47), getitem___153, int_151)
                    
                    # Processing the call keyword arguments (line 57)
                    kwargs_155 = {}
                    # Getting the type of 'victors' (line 57)
                    victors_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'victors', False)
                    # Obtaining the member 'append' of a type (line 57)
                    append_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 32), victors_149, 'append')
                    # Calling append(args, kwargs) (line 57)
                    append_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 57, 32), append_150, *[subscript_call_result_154], **kwargs_155)
                    
                    # SSA join for if statement (line 57)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 56)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a List to a Name (line 61):
        
        # Assigning a List to a Name (line 61):
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        
        # Assigning a type to the variable 'ld' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'ld', list_157)
        
        
        # Call to xrange(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'self', False)
        # Obtaining the member 'edge' of a type (line 62)
        edge_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 24), self_159, 'edge')
        # Processing the call keyword arguments (line 62)
        kwargs_161 = {}
        # Getting the type of 'xrange' (line 62)
        xrange_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 62)
        xrange_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), xrange_158, *[edge_160], **kwargs_161)
        
        # Assigning a type to the variable 'xrange_call_result_162' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'xrange_call_result_162', xrange_call_result_162)
        # Testing if the for loop is going to be iterated (line 62)
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 8), xrange_call_result_162)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 62, 8), xrange_call_result_162):
            # Getting the type of the for loop variable (line 62)
            for_loop_var_163 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 8), xrange_call_result_162)
            # Assigning a type to the variable 'i' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'i', for_loop_var_163)
            # SSA begins for a for statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 62)
            # Processing the call arguments (line 62)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 62)
            i_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 'i', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 62)
            i_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 59), 'i', False)
            # Getting the type of 'self' (line 62)
            self_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'self', False)
            # Obtaining the member '__board' of a type (line 62)
            board_169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 46), self_168, '__board')
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 46), board_169, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 62, 46), getitem___170, i_167)
            
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 46), subscript_call_result_171, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 62, 46), getitem___172, i_166)
            
            # Processing the call keyword arguments (line 62)
            kwargs_174 = {}
            # Getting the type of 'ld' (line 62)
            ld_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 36), 'ld', False)
            # Obtaining the member 'append' of a type (line 62)
            append_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 36), ld_164, 'append')
            # Calling append(args, kwargs) (line 62)
            append_call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 62, 36), append_165, *[subscript_call_result_173], **kwargs_174)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to set(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'ld' (line 63)
        ld_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'ld', False)
        # Processing the call keyword arguments (line 63)
        kwargs_179 = {}
        # Getting the type of 'set' (line 63)
        set_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'set', False)
        # Calling set(args, kwargs) (line 63)
        set_call_result_180 = invoke(stypy.reporting.localization.Localization(__file__, 63, 15), set_177, *[ld_178], **kwargs_179)
        
        # Processing the call keyword arguments (line 63)
        kwargs_181 = {}
        # Getting the type of 'len' (line 63)
        len_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'len', False)
        # Calling len(args, kwargs) (line 63)
        len_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), len_176, *[set_call_result_180], **kwargs_181)
        
        int_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
        # Applying the binary operator '==' (line 63)
        result_eq_184 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 11), '==', len_call_result_182, int_183)
        
        # Testing if the type of an if condition is none (line 63)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 63, 8), result_eq_184):
            pass
        else:
            
            # Testing the type of an if condition (line 63)
            if_condition_185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), result_eq_184)
            # Assigning a type to the variable 'if_condition_185' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_185', if_condition_185)
            # SSA begins for if statement (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining the type of the subscript
            int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'int')
            # Getting the type of 'ld' (line 64)
            ld_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'ld')
            # Obtaining the member '__getitem__' of a type (line 64)
            getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), ld_187, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 64)
            subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), getitem___188, int_186)
            
            int_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 24), 'int')
            # Applying the binary operator '!=' (line 64)
            result_ne_191 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 15), '!=', subscript_call_result_189, int_190)
            
            # Testing if the type of an if condition is none (line 64)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 12), result_ne_191):
                pass
            else:
                
                # Testing the type of an if condition (line 64)
                if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 12), result_ne_191)
                # Assigning a type to the variable 'if_condition_192' (line 64)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'if_condition_192', if_condition_192)
                # SSA begins for if statement (line 64)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 64)
                # Processing the call arguments (line 64)
                
                # Obtaining the type of the subscript
                int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'int')
                # Getting the type of 'ld' (line 64)
                ld_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'ld', False)
                # Obtaining the member '__getitem__' of a type (line 64)
                getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 42), ld_196, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 64)
                subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 64, 42), getitem___197, int_195)
                
                # Processing the call keyword arguments (line 64)
                kwargs_199 = {}
                # Getting the type of 'victors' (line 64)
                victors_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'victors', False)
                # Obtaining the member 'append' of a type (line 64)
                append_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 27), victors_193, 'append')
                # Calling append(args, kwargs) (line 64)
                append_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 64, 27), append_194, *[subscript_call_result_198], **kwargs_199)
                
                # SSA join for if statement (line 64)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 67):
        
        # Assigning a List to a Name (line 67):
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        
        # Assigning a type to the variable 'rd' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'rd', list_201)
        
        
        # Call to xrange(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'self' (line 68)
        self_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'self', False)
        # Obtaining the member 'edge' of a type (line 68)
        edge_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 24), self_203, 'edge')
        # Processing the call keyword arguments (line 68)
        kwargs_205 = {}
        # Getting the type of 'xrange' (line 68)
        xrange_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 68)
        xrange_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 68, 17), xrange_202, *[edge_204], **kwargs_205)
        
        # Assigning a type to the variable 'xrange_call_result_206' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'xrange_call_result_206', xrange_call_result_206)
        # Testing if the for loop is going to be iterated (line 68)
        # Testing the type of a for loop iterable (line 68)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 8), xrange_call_result_206)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 68, 8), xrange_call_result_206):
            # Getting the type of the for loop variable (line 68)
            for_loop_var_207 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 8), xrange_call_result_206)
            # Assigning a type to the variable 'i' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'i', for_loop_var_207)
            # SSA begins for a for statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 68)
            # Processing the call arguments (line 68)
            
            # Obtaining the type of the subscript
            # Getting the type of 'self' (line 68)
            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 62), 'self', False)
            # Obtaining the member 'edge' of a type (line 68)
            edge_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 62), self_210, 'edge')
            int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 75), 'int')
            # Getting the type of 'i' (line 68)
            i_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 79), 'i', False)
            # Applying the binary operator '+' (line 68)
            result_add_214 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 75), '+', int_212, i_213)
            
            # Applying the binary operator '-' (line 68)
            result_sub_215 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 62), '-', edge_211, result_add_214)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 68)
            i_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 59), 'i', False)
            # Getting the type of 'self' (line 68)
            self_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 46), 'self', False)
            # Obtaining the member '__board' of a type (line 68)
            board_218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 46), self_217, '__board')
            # Obtaining the member '__getitem__' of a type (line 68)
            getitem___219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 46), board_218, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 68)
            subscript_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 68, 46), getitem___219, i_216)
            
            # Obtaining the member '__getitem__' of a type (line 68)
            getitem___221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 46), subscript_call_result_220, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 68)
            subscript_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 68, 46), getitem___221, result_sub_215)
            
            # Processing the call keyword arguments (line 68)
            kwargs_223 = {}
            # Getting the type of 'rd' (line 68)
            rd_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'rd', False)
            # Obtaining the member 'append' of a type (line 68)
            append_209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 36), rd_208, 'append')
            # Calling append(args, kwargs) (line 68)
            append_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 68, 36), append_209, *[subscript_call_result_222], **kwargs_223)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to set(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'rd' (line 69)
        rd_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'rd', False)
        # Processing the call keyword arguments (line 69)
        kwargs_228 = {}
        # Getting the type of 'set' (line 69)
        set_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'set', False)
        # Calling set(args, kwargs) (line 69)
        set_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), set_226, *[rd_227], **kwargs_228)
        
        # Processing the call keyword arguments (line 69)
        kwargs_230 = {}
        # Getting the type of 'len' (line 69)
        len_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'len', False)
        # Calling len(args, kwargs) (line 69)
        len_call_result_231 = invoke(stypy.reporting.localization.Localization(__file__, 69, 11), len_225, *[set_call_result_229], **kwargs_230)
        
        int_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 27), 'int')
        # Applying the binary operator '==' (line 69)
        result_eq_233 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '==', len_call_result_231, int_232)
        
        # Testing if the type of an if condition is none (line 69)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 8), result_eq_233):
            pass
        else:
            
            # Testing the type of an if condition (line 69)
            if_condition_234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_eq_233)
            # Assigning a type to the variable 'if_condition_234' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_234', if_condition_234)
            # SSA begins for if statement (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining the type of the subscript
            int_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'int')
            # Getting the type of 'rd' (line 70)
            rd_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'rd')
            # Obtaining the member '__getitem__' of a type (line 70)
            getitem___237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), rd_236, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 70)
            subscript_call_result_238 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), getitem___237, int_235)
            
            int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'int')
            # Applying the binary operator '!=' (line 70)
            result_ne_240 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), '!=', subscript_call_result_238, int_239)
            
            # Testing if the type of an if condition is none (line 70)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 12), result_ne_240):
                pass
            else:
                
                # Testing the type of an if condition (line 70)
                if_condition_241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), result_ne_240)
                # Assigning a type to the variable 'if_condition_241' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_241', if_condition_241)
                # SSA begins for if statement (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 70)
                # Processing the call arguments (line 70)
                
                # Obtaining the type of the subscript
                int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 45), 'int')
                # Getting the type of 'rd' (line 70)
                rd_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 42), 'rd', False)
                # Obtaining the member '__getitem__' of a type (line 70)
                getitem___246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 42), rd_245, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                subscript_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 70, 42), getitem___246, int_244)
                
                # Processing the call keyword arguments (line 70)
                kwargs_248 = {}
                # Getting the type of 'victors' (line 70)
                victors_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'victors', False)
                # Obtaining the member 'append' of a type (line 70)
                append_243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 27), victors_242, 'append')
                # Calling append(args, kwargs) (line 70)
                append_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 70, 27), append_243, *[subscript_call_result_247], **kwargs_248)
                
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'victors' (line 73)
        victors_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'victors', False)
        # Processing the call keyword arguments (line 73)
        kwargs_252 = {}
        # Getting the type of 'len' (line 73)
        len_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'len', False)
        # Calling len(args, kwargs) (line 73)
        len_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 73, 11), len_250, *[victors_251], **kwargs_252)
        
        int_254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 27), 'int')
        # Applying the binary operator '==' (line 73)
        result_eq_255 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '==', len_call_result_253, int_254)
        
        # Testing if the type of an if condition is none (line 73)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_255):
            pass
        else:
            
            # Testing the type of an if condition (line 73)
            if_condition_256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_255)
            # Assigning a type to the variable 'if_condition_256' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_256', if_condition_256)
            # SSA begins for if statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', int_257)
            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to set(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'victors' (line 75)
        victors_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'victors', False)
        # Processing the call keyword arguments (line 75)
        kwargs_261 = {}
        # Getting the type of 'set' (line 75)
        set_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'set', False)
        # Calling set(args, kwargs) (line 75)
        set_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), set_259, *[victors_260], **kwargs_261)
        
        # Processing the call keyword arguments (line 75)
        kwargs_263 = {}
        # Getting the type of 'len' (line 75)
        len_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'len', False)
        # Calling len(args, kwargs) (line 75)
        len_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), len_258, *[set_call_result_262], **kwargs_263)
        
        int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 31), 'int')
        # Applying the binary operator '>' (line 75)
        result_gt_266 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), '>', len_call_result_264, int_265)
        
        # Testing if the type of an if condition is none (line 75)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 8), result_gt_266):
            pass
        else:
            
            # Testing the type of an if condition (line 75)
            if_condition_267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_gt_266)
            # Assigning a type to the variable 'if_condition_267' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_267', if_condition_267)
            # SSA begins for if statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to MultiVictory(...): (line 76)
            # Processing the call arguments (line 76)
            
            # Call to set(...): (line 76)
            # Processing the call arguments (line 76)
            # Getting the type of 'victors' (line 76)
            victors_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'victors', False)
            # Processing the call keyword arguments (line 76)
            kwargs_271 = {}
            # Getting the type of 'set' (line 76)
            set_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'set', False)
            # Calling set(args, kwargs) (line 76)
            set_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), set_269, *[victors_270], **kwargs_271)
            
            # Processing the call keyword arguments (line 76)
            kwargs_273 = {}
            # Getting the type of 'MultiVictory' (line 76)
            MultiVictory_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'MultiVictory', False)
            # Calling MultiVictory(args, kwargs) (line 76)
            MultiVictory_call_result_274 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), MultiVictory_268, *[set_call_result_272], **kwargs_273)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 76, 12), MultiVictory_call_result_274, 'raise parameter', BaseException)
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining the type of the subscript
        int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 23), 'int')
        # Getting the type of 'victors' (line 77)
        victors_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'victors')
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 15), victors_276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 77, 15), getitem___277, int_275)
        
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'stypy_return_type', subscript_call_result_278)
        
        # ################# End of 'isvictory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isvictory' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isvictory'
        return stypy_return_type_279


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_function_name', 'rectBoard.stypy__str__')
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        rectBoard.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'rectBoard.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Assigning a Str to a Name (line 80):
        
        # Assigning a Str to a Name (line 80):
        str_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'str', '')
        # Assigning a type to the variable 'ret' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'ret', str_280)
        
        
        # Call to xrange(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'self' (line 81)
        self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'self', False)
        # Obtaining the member 'edge' of a type (line 81)
        edge_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 26), self_282, 'edge')
        # Processing the call keyword arguments (line 81)
        kwargs_284 = {}
        # Getting the type of 'xrange' (line 81)
        xrange_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 81)
        xrange_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 81, 19), xrange_281, *[edge_283], **kwargs_284)
        
        # Assigning a type to the variable 'xrange_call_result_285' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'xrange_call_result_285', xrange_call_result_285)
        # Testing if the for loop is going to be iterated (line 81)
        # Testing the type of a for loop iterable (line 81)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 8), xrange_call_result_285)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 8), xrange_call_result_285):
            # Getting the type of the for loop variable (line 81)
            for_loop_var_286 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 8), xrange_call_result_285)
            # Assigning a type to the variable 'row' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'row', for_loop_var_286)
            # SSA begins for a for statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'row' (line 82)
            row_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'row')
            int_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'int')
            # Applying the binary operator '!=' (line 82)
            result_ne_289 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 15), '!=', row_287, int_288)
            
            # Testing if the type of an if condition is none (line 82)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 12), result_ne_289):
                pass
            else:
                
                # Testing the type of an if condition (line 82)
                if_condition_290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), result_ne_289)
                # Assigning a type to the variable 'if_condition_290' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_290', if_condition_290)
                # SSA begins for if statement (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'ret' (line 83)
                ret_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'ret')
                str_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'str', '\n')
                # Applying the binary operator '+=' (line 83)
                result_iadd_293 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 16), '+=', ret_291, str_292)
                # Assigning a type to the variable 'ret' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'ret', result_iadd_293)
                
                
                
                # Call to xrange(...): (line 84)
                # Processing the call arguments (line 84)
                # Getting the type of 'self' (line 84)
                self_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'self', False)
                # Obtaining the member 'edge' of a type (line 84)
                edge_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), self_295, 'edge')
                # Processing the call keyword arguments (line 84)
                kwargs_297 = {}
                # Getting the type of 'xrange' (line 84)
                xrange_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'xrange', False)
                # Calling xrange(args, kwargs) (line 84)
                xrange_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 84, 25), xrange_294, *[edge_296], **kwargs_297)
                
                # Assigning a type to the variable 'xrange_call_result_298' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'xrange_call_result_298', xrange_call_result_298)
                # Testing if the for loop is going to be iterated (line 84)
                # Testing the type of a for loop iterable (line 84)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 16), xrange_call_result_298)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 16), xrange_call_result_298):
                    # Getting the type of the for loop variable (line 84)
                    for_loop_var_299 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 16), xrange_call_result_298)
                    # Assigning a type to the variable 'i' (line 84)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'i', for_loop_var_299)
                    # SSA begins for a for statement (line 84)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'i' (line 85)
                    i_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'i')
                    int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'int')
                    # Applying the binary operator '!=' (line 85)
                    result_ne_302 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 23), '!=', i_300, int_301)
                    
                    # Testing if the type of an if condition is none (line 85)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 20), result_ne_302):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 85)
                        if_condition_303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 20), result_ne_302)
                        # Assigning a type to the variable 'if_condition_303' (line 85)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'if_condition_303', if_condition_303)
                        # SSA begins for if statement (line 85)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'ret' (line 85)
                        ret_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'ret')
                        str_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 38), 'str', '+')
                        # Applying the binary operator '+=' (line 85)
                        result_iadd_306 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 31), '+=', ret_304, str_305)
                        # Assigning a type to the variable 'ret' (line 85)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'ret', result_iadd_306)
                        
                        # SSA join for if statement (line 85)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'ret' (line 86)
                    ret_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'ret')
                    str_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'str', '---')
                    # Applying the binary operator '+=' (line 86)
                    result_iadd_309 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 20), '+=', ret_307, str_308)
                    # Assigning a type to the variable 'ret' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'ret', result_iadd_309)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Getting the type of 'ret' (line 87)
                ret_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'ret')
                str_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'str', '\n')
                # Applying the binary operator '+=' (line 87)
                result_iadd_312 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 16), '+=', ret_310, str_311)
                # Assigning a type to the variable 'ret' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'ret', result_iadd_312)
                
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'ret' (line 88)
            ret_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'ret')
            str_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'str', ' ')
            # Applying the binary operator '+=' (line 88)
            result_iadd_315 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '+=', ret_313, str_314)
            # Assigning a type to the variable 'ret' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'ret', result_iadd_315)
            
            
            
            # Call to xrange(...): (line 89)
            # Processing the call arguments (line 89)
            # Getting the type of 'self' (line 89)
            self_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'self', False)
            # Obtaining the member 'edge' of a type (line 89)
            edge_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), self_317, 'edge')
            # Processing the call keyword arguments (line 89)
            kwargs_319 = {}
            # Getting the type of 'xrange' (line 89)
            xrange_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'xrange', False)
            # Calling xrange(args, kwargs) (line 89)
            xrange_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 89, 23), xrange_316, *[edge_318], **kwargs_319)
            
            # Assigning a type to the variable 'xrange_call_result_320' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'xrange_call_result_320', xrange_call_result_320)
            # Testing if the for loop is going to be iterated (line 89)
            # Testing the type of a for loop iterable (line 89)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 12), xrange_call_result_320)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 89, 12), xrange_call_result_320):
                # Getting the type of the for loop variable (line 89)
                for_loop_var_321 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 12), xrange_call_result_320)
                # Assigning a type to the variable 'col' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'col', for_loop_var_321)
                # SSA begins for a for statement (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'col' (line 90)
                col_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'col')
                int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'int')
                # Applying the binary operator '!=' (line 90)
                result_ne_324 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '!=', col_322, int_323)
                
                # Testing if the type of an if condition is none (line 90)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 16), result_ne_324):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 90)
                    if_condition_325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 16), result_ne_324)
                    # Assigning a type to the variable 'if_condition_325' (line 90)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'if_condition_325', if_condition_325)
                    # SSA begins for if statement (line 90)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'ret' (line 90)
                    ret_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'ret')
                    str_327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 36), 'str', ' | ')
                    # Applying the binary operator '+=' (line 90)
                    result_iadd_328 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 29), '+=', ret_326, str_327)
                    # Assigning a type to the variable 'ret' (line 90)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'ret', result_iadd_328)
                    
                    # SSA join for if statement (line 90)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 91)
                col_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 91)
                row_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'row')
                # Getting the type of 'self' (line 91)
                self_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'self')
                # Obtaining the member '__board' of a type (line 91)
                board_332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), self_331, '__board')
                # Obtaining the member '__getitem__' of a type (line 91)
                getitem___333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), board_332, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 91)
                subscript_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 91, 19), getitem___333, row_330)
                
                # Obtaining the member '__getitem__' of a type (line 91)
                getitem___335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), subscript_call_result_334, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 91)
                subscript_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 91, 19), getitem___335, col_329)
                
                int_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'int')
                # Applying the binary operator '==' (line 91)
                result_eq_338 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 19), '==', subscript_call_result_336, int_337)
                
                # Testing if the type of an if condition is none (line 91)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 16), result_eq_338):
                    
                    # Getting the type of 'ret' (line 94)
                    ret_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'ret')
                    
                    # Call to str(...): (line 94)
                    # Processing the call arguments (line 94)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 94)
                    col_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 94)
                    row_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'row', False)
                    # Getting the type of 'self' (line 94)
                    self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'self', False)
                    # Obtaining the member '__board' of a type (line 94)
                    board_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), self_347, '__board')
                    # Obtaining the member '__getitem__' of a type (line 94)
                    getitem___349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), board_348, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
                    subscript_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), getitem___349, row_346)
                    
                    # Obtaining the member '__getitem__' of a type (line 94)
                    getitem___351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), subscript_call_result_350, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
                    subscript_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), getitem___351, col_345)
                    
                    # Processing the call keyword arguments (line 94)
                    kwargs_353 = {}
                    # Getting the type of 'str' (line 94)
                    str_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'str', False)
                    # Calling str(args, kwargs) (line 94)
                    str_call_result_354 = invoke(stypy.reporting.localization.Localization(__file__, 94, 27), str_344, *[subscript_call_result_352], **kwargs_353)
                    
                    # Applying the binary operator '+=' (line 94)
                    result_iadd_355 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 20), '+=', ret_343, str_call_result_354)
                    # Assigning a type to the variable 'ret' (line 94)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'ret', result_iadd_355)
                    
                else:
                    
                    # Testing the type of an if condition (line 91)
                    if_condition_339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 16), result_eq_338)
                    # Assigning a type to the variable 'if_condition_339' (line 91)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'if_condition_339', if_condition_339)
                    # SSA begins for if statement (line 91)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'ret' (line 92)
                    ret_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'ret')
                    str_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 27), 'str', ' ')
                    # Applying the binary operator '+=' (line 92)
                    result_iadd_342 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 20), '+=', ret_340, str_341)
                    # Assigning a type to the variable 'ret' (line 92)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'ret', result_iadd_342)
                    
                    # SSA branch for the else part of an if statement (line 91)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'ret' (line 94)
                    ret_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'ret')
                    
                    # Call to str(...): (line 94)
                    # Processing the call arguments (line 94)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 94)
                    col_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 94)
                    row_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'row', False)
                    # Getting the type of 'self' (line 94)
                    self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'self', False)
                    # Obtaining the member '__board' of a type (line 94)
                    board_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), self_347, '__board')
                    # Obtaining the member '__getitem__' of a type (line 94)
                    getitem___349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), board_348, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
                    subscript_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), getitem___349, row_346)
                    
                    # Obtaining the member '__getitem__' of a type (line 94)
                    getitem___351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), subscript_call_result_350, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
                    subscript_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), getitem___351, col_345)
                    
                    # Processing the call keyword arguments (line 94)
                    kwargs_353 = {}
                    # Getting the type of 'str' (line 94)
                    str_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'str', False)
                    # Calling str(args, kwargs) (line 94)
                    str_call_result_354 = invoke(stypy.reporting.localization.Localization(__file__, 94, 27), str_344, *[subscript_call_result_352], **kwargs_353)
                    
                    # Applying the binary operator '+=' (line 94)
                    result_iadd_355 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 20), '+=', ret_343, str_call_result_354)
                    # Assigning a type to the variable 'ret' (line 94)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'ret', result_iadd_355)
                    
                    # SSA join for if statement (line 91)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'ret' (line 95)
        ret_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', ret_356)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_357


    @norecursion
    def doRow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'doRow'
        module_type_store = module_type_store.open_function_context('doRow', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        rectBoard.doRow.__dict__.__setitem__('stypy_localization', localization)
        rectBoard.doRow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        rectBoard.doRow.__dict__.__setitem__('stypy_type_store', module_type_store)
        rectBoard.doRow.__dict__.__setitem__('stypy_function_name', 'rectBoard.doRow')
        rectBoard.doRow.__dict__.__setitem__('stypy_param_names_list', ['fields', 'indices', 'player', 'scores'])
        rectBoard.doRow.__dict__.__setitem__('stypy_varargs_param_name', None)
        rectBoard.doRow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        rectBoard.doRow.__dict__.__setitem__('stypy_call_defaults', defaults)
        rectBoard.doRow.__dict__.__setitem__('stypy_call_varargs', varargs)
        rectBoard.doRow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        rectBoard.doRow.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'rectBoard.doRow', ['fields', 'indices', 'player', 'scores'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'doRow', localization, ['fields', 'indices', 'player', 'scores'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'doRow(...)' code ##################

        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to difference(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to set(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        int_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 45), list_364, int_365)
        
        # Processing the call keyword arguments (line 98)
        kwargs_366 = {}
        # Getting the type of 'set' (line 98)
        set_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'set', False)
        # Calling set(args, kwargs) (line 98)
        set_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 98, 41), set_363, *[list_364], **kwargs_366)
        
        # Processing the call keyword arguments (line 98)
        kwargs_368 = {}
        
        # Call to set(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'fields' (line 98)
        fields_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'fields', False)
        # Processing the call keyword arguments (line 98)
        kwargs_360 = {}
        # Getting the type of 'set' (line 98)
        set_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'set', False)
        # Calling set(args, kwargs) (line 98)
        set_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), set_358, *[fields_359], **kwargs_360)
        
        # Obtaining the member 'difference' of a type (line 98)
        difference_362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 18), set_call_result_361, 'difference')
        # Calling difference(args, kwargs) (line 98)
        difference_call_result_369 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), difference_362, *[set_call_result_367], **kwargs_368)
        
        # Assigning a type to the variable 'players' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'players', difference_call_result_369)
        
        
        # Call to len(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'players' (line 100)
        players_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'players', False)
        # Processing the call keyword arguments (line 100)
        kwargs_372 = {}
        # Getting the type of 'len' (line 100)
        len_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'len', False)
        # Calling len(args, kwargs) (line 100)
        len_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), len_370, *[players_371], **kwargs_372)
        
        int_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 28), 'int')
        # Applying the binary operator '==' (line 100)
        result_eq_375 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 12), '==', len_call_result_373, int_374)
        
        # Testing if the type of an if condition is none (line 100)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 8), result_eq_375):
            pass
        else:
            
            # Testing the type of an if condition (line 100)
            if_condition_376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_eq_375)
            # Assigning a type to the variable 'if_condition_376' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_376', if_condition_376)
            # SSA begins for if statement (line 100)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining the type of the subscript
            int_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'int')
            
            # Call to list(...): (line 101)
            # Processing the call arguments (line 101)
            # Getting the type of 'players' (line 101)
            players_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'players', False)
            # Processing the call keyword arguments (line 101)
            kwargs_380 = {}
            # Getting the type of 'list' (line 101)
            list_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'list', False)
            # Calling list(args, kwargs) (line 101)
            list_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), list_378, *[players_379], **kwargs_380)
            
            # Obtaining the member '__getitem__' of a type (line 101)
            getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), list_call_result_381, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 101)
            subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), getitem___382, int_377)
            
            # Getting the type of 'player' (line 101)
            player_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'player')
            # Applying the binary operator '==' (line 101)
            result_eq_385 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '==', subscript_call_result_383, player_384)
            
            # Testing if the type of an if condition is none (line 101)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 12), result_eq_385):
                
                # Getting the type of 'indices' (line 105)
                indices_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'indices')
                # Assigning a type to the variable 'indices_424' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'indices_424', indices_424)
                # Testing if the for loop is going to be iterated (line 105)
                # Testing the type of a for loop iterable (line 105)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 16), indices_424)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 16), indices_424):
                    # Getting the type of the for loop variable (line 105)
                    for_loop_var_425 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 16), indices_424)
                    # Assigning a type to the variable 'rown' (line 105)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'rown', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), for_loop_var_425, 2, 0))
                    # Assigning a type to the variable 'coln' (line 105)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'coln', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), for_loop_var_425, 2, 1))
                    # SSA begins for a for statement (line 105)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 106)
                    rown_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'rown')
                    # Getting the type of 'scores' (line 106)
                    scores_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), scores_427, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___428, rown_426)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'coln' (line 106)
                    coln_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'coln')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 106)
                    rown_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'rown')
                    # Getting the type of 'scores' (line 106)
                    scores_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), scores_432, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___433, rown_431)
                    
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), subscript_call_result_434, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_436 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___435, coln_430)
                    
                    int_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 42), 'int')
                    
                    # Call to count(...): (line 106)
                    # Processing the call arguments (line 106)
                    
                    # Obtaining the type of the subscript
                    int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 74), 'int')
                    
                    # Call to list(...): (line 106)
                    # Processing the call arguments (line 106)
                    # Getting the type of 'players' (line 106)
                    players_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 65), 'players', False)
                    # Processing the call keyword arguments (line 106)
                    kwargs_443 = {}
                    # Getting the type of 'list' (line 106)
                    list_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'list', False)
                    # Calling list(args, kwargs) (line 106)
                    list_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 106, 60), list_441, *[players_442], **kwargs_443)
                    
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), list_call_result_444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 106, 60), getitem___445, int_440)
                    
                    # Processing the call keyword arguments (line 106)
                    kwargs_447 = {}
                    # Getting the type of 'fields' (line 106)
                    fields_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'fields', False)
                    # Obtaining the member 'count' of a type (line 106)
                    count_439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 47), fields_438, 'count')
                    # Calling count(args, kwargs) (line 106)
                    count_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 106, 47), count_439, *[subscript_call_result_446], **kwargs_447)
                    
                    # Applying the binary operator '*' (line 106)
                    result_mul_449 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 42), '*', int_437, count_call_result_448)
                    
                    
                    # Call to float(...): (line 106)
                    # Processing the call arguments (line 106)
                    # Getting the type of 'self' (line 106)
                    self_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 86), 'self', False)
                    # Obtaining the member 'edge' of a type (line 106)
                    edge_452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 86), self_451, 'edge')
                    # Processing the call keyword arguments (line 106)
                    kwargs_453 = {}
                    # Getting the type of 'float' (line 106)
                    float_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 80), 'float', False)
                    # Calling float(args, kwargs) (line 106)
                    float_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 106, 80), float_450, *[edge_452], **kwargs_453)
                    
                    # Applying the binary operator 'div' (line 106)
                    result_div_455 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 78), 'div', result_mul_449, float_call_result_454)
                    
                    # Applying the binary operator '+=' (line 106)
                    result_iadd_456 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 20), '+=', subscript_call_result_436, result_div_455)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 106)
                    rown_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'rown')
                    # Getting the type of 'scores' (line 106)
                    scores_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), scores_458, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___459, rown_457)
                    
                    # Getting the type of 'coln' (line 106)
                    coln_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'coln')
                    # Storing an element on a container (line 106)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 20), subscript_call_result_460, (coln_461, result_iadd_456))
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
            else:
                
                # Testing the type of an if condition (line 101)
                if_condition_386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 12), result_eq_385)
                # Assigning a type to the variable 'if_condition_386' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'if_condition_386', if_condition_386)
                # SSA begins for if statement (line 101)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'indices' (line 102)
                indices_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'indices')
                # Assigning a type to the variable 'indices_387' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'indices_387', indices_387)
                # Testing if the for loop is going to be iterated (line 102)
                # Testing the type of a for loop iterable (line 102)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 16), indices_387)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 102, 16), indices_387):
                    # Getting the type of the for loop variable (line 102)
                    for_loop_var_388 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 16), indices_387)
                    # Assigning a type to the variable 'rown' (line 102)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'rown', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), for_loop_var_388, 2, 0))
                    # Assigning a type to the variable 'coln' (line 102)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'coln', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), for_loop_var_388, 2, 1))
                    # SSA begins for a for statement (line 102)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 103)
                    rown_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'rown')
                    # Getting the type of 'scores' (line 103)
                    scores_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 103)
                    getitem___391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), scores_390, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
                    subscript_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), getitem___391, rown_389)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'coln' (line 103)
                    coln_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'coln')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 103)
                    rown_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'rown')
                    # Getting the type of 'scores' (line 103)
                    scores_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 103)
                    getitem___396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), scores_395, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
                    subscript_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), getitem___396, rown_394)
                    
                    # Obtaining the member '__getitem__' of a type (line 103)
                    getitem___398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), subscript_call_result_397, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
                    subscript_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), getitem___398, coln_393)
                    
                    int_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 42), 'int')
                    
                    # Call to sig(...): (line 103)
                    # Processing the call arguments (line 103)
                    
                    # Call to count(...): (line 103)
                    # Processing the call arguments (line 103)
                    # Getting the type of 'player' (line 103)
                    player_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 64), 'player', False)
                    # Processing the call keyword arguments (line 103)
                    kwargs_405 = {}
                    # Getting the type of 'fields' (line 103)
                    fields_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 51), 'fields', False)
                    # Obtaining the member 'count' of a type (line 103)
                    count_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 51), fields_402, 'count')
                    # Calling count(args, kwargs) (line 103)
                    count_call_result_406 = invoke(stypy.reporting.localization.Localization(__file__, 103, 51), count_403, *[player_404], **kwargs_405)
                    
                    
                    # Call to float(...): (line 103)
                    # Processing the call arguments (line 103)
                    # Getting the type of 'self' (line 103)
                    self_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 80), 'self', False)
                    # Obtaining the member 'edge' of a type (line 103)
                    edge_409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 80), self_408, 'edge')
                    # Processing the call keyword arguments (line 103)
                    kwargs_410 = {}
                    # Getting the type of 'float' (line 103)
                    float_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 74), 'float', False)
                    # Calling float(args, kwargs) (line 103)
                    float_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 103, 74), float_407, *[edge_409], **kwargs_410)
                    
                    # Applying the binary operator 'div' (line 103)
                    result_div_412 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 51), 'div', count_call_result_406, float_call_result_411)
                    
                    float_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 92), 'float')
                    int_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 96), 'int')
                    # Processing the call keyword arguments (line 103)
                    kwargs_415 = {}
                    # Getting the type of 'sig' (line 103)
                    sig_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 47), 'sig', False)
                    # Calling sig(args, kwargs) (line 103)
                    sig_call_result_416 = invoke(stypy.reporting.localization.Localization(__file__, 103, 47), sig_401, *[result_div_412, float_413, int_414], **kwargs_415)
                    
                    # Applying the binary operator '*' (line 103)
                    result_mul_417 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 42), '*', int_400, sig_call_result_416)
                    
                    # Applying the binary operator '+=' (line 103)
                    result_iadd_418 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 20), '+=', subscript_call_result_399, result_mul_417)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 103)
                    rown_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'rown')
                    # Getting the type of 'scores' (line 103)
                    scores_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 103)
                    getitem___421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), scores_420, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
                    subscript_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), getitem___421, rown_419)
                    
                    # Getting the type of 'coln' (line 103)
                    coln_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'coln')
                    # Storing an element on a container (line 103)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), subscript_call_result_422, (coln_423, result_iadd_418))
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA branch for the else part of an if statement (line 101)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'indices' (line 105)
                indices_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'indices')
                # Assigning a type to the variable 'indices_424' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'indices_424', indices_424)
                # Testing if the for loop is going to be iterated (line 105)
                # Testing the type of a for loop iterable (line 105)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 16), indices_424)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 16), indices_424):
                    # Getting the type of the for loop variable (line 105)
                    for_loop_var_425 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 16), indices_424)
                    # Assigning a type to the variable 'rown' (line 105)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'rown', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), for_loop_var_425, 2, 0))
                    # Assigning a type to the variable 'coln' (line 105)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'coln', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), for_loop_var_425, 2, 1))
                    # SSA begins for a for statement (line 105)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 106)
                    rown_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'rown')
                    # Getting the type of 'scores' (line 106)
                    scores_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), scores_427, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___428, rown_426)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'coln' (line 106)
                    coln_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'coln')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 106)
                    rown_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'rown')
                    # Getting the type of 'scores' (line 106)
                    scores_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), scores_432, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___433, rown_431)
                    
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), subscript_call_result_434, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_436 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___435, coln_430)
                    
                    int_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 42), 'int')
                    
                    # Call to count(...): (line 106)
                    # Processing the call arguments (line 106)
                    
                    # Obtaining the type of the subscript
                    int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 74), 'int')
                    
                    # Call to list(...): (line 106)
                    # Processing the call arguments (line 106)
                    # Getting the type of 'players' (line 106)
                    players_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 65), 'players', False)
                    # Processing the call keyword arguments (line 106)
                    kwargs_443 = {}
                    # Getting the type of 'list' (line 106)
                    list_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'list', False)
                    # Calling list(args, kwargs) (line 106)
                    list_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 106, 60), list_441, *[players_442], **kwargs_443)
                    
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 60), list_call_result_444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 106, 60), getitem___445, int_440)
                    
                    # Processing the call keyword arguments (line 106)
                    kwargs_447 = {}
                    # Getting the type of 'fields' (line 106)
                    fields_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'fields', False)
                    # Obtaining the member 'count' of a type (line 106)
                    count_439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 47), fields_438, 'count')
                    # Calling count(args, kwargs) (line 106)
                    count_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 106, 47), count_439, *[subscript_call_result_446], **kwargs_447)
                    
                    # Applying the binary operator '*' (line 106)
                    result_mul_449 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 42), '*', int_437, count_call_result_448)
                    
                    
                    # Call to float(...): (line 106)
                    # Processing the call arguments (line 106)
                    # Getting the type of 'self' (line 106)
                    self_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 86), 'self', False)
                    # Obtaining the member 'edge' of a type (line 106)
                    edge_452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 86), self_451, 'edge')
                    # Processing the call keyword arguments (line 106)
                    kwargs_453 = {}
                    # Getting the type of 'float' (line 106)
                    float_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 80), 'float', False)
                    # Calling float(args, kwargs) (line 106)
                    float_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 106, 80), float_450, *[edge_452], **kwargs_453)
                    
                    # Applying the binary operator 'div' (line 106)
                    result_div_455 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 78), 'div', result_mul_449, float_call_result_454)
                    
                    # Applying the binary operator '+=' (line 106)
                    result_iadd_456 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 20), '+=', subscript_call_result_436, result_div_455)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 106)
                    rown_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'rown')
                    # Getting the type of 'scores' (line 106)
                    scores_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'scores')
                    # Obtaining the member '__getitem__' of a type (line 106)
                    getitem___459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), scores_458, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                    subscript_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___459, rown_457)
                    
                    # Getting the type of 'coln' (line 106)
                    coln_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'coln')
                    # Storing an element on a container (line 106)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 20), subscript_call_result_460, (coln_461, result_iadd_456))
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for if statement (line 101)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 100)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'doRow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'doRow' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'doRow'
        return stypy_return_type_462


    @norecursion
    def makeAImove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'makeAImove'
        module_type_store = module_type_store.open_function_context('makeAImove', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        rectBoard.makeAImove.__dict__.__setitem__('stypy_localization', localization)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_type_store', module_type_store)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_function_name', 'rectBoard.makeAImove')
        rectBoard.makeAImove.__dict__.__setitem__('stypy_param_names_list', ['player'])
        rectBoard.makeAImove.__dict__.__setitem__('stypy_varargs_param_name', None)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_kwargs_param_name', None)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_call_defaults', defaults)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_call_varargs', varargs)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        rectBoard.makeAImove.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'rectBoard.makeAImove', ['player'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'makeAImove', localization, ['player'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'makeAImove(...)' code ##################

        
        # Assigning a ListComp to a Name (line 109):
        
        # Assigning a ListComp to a Name (line 109):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 50), 'self', False)
        # Obtaining the member 'edge' of a type (line 109)
        edge_470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 50), self_469, 'edge')
        # Processing the call keyword arguments (line 109)
        kwargs_471 = {}
        # Getting the type of 'xrange' (line 109)
        xrange_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 43), 'xrange', False)
        # Calling xrange(args, kwargs) (line 109)
        xrange_call_result_472 = invoke(stypy.reporting.localization.Localization(__file__, 109, 43), xrange_468, *[edge_470], **kwargs_471)
        
        comprehension_473 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), xrange_call_result_472)
        # Assigning a type to the variable 'i' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'i', comprehension_473)
        # Getting the type of 'self' (line 109)
        self_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'self')
        # Obtaining the member 'edge' of a type (line 109)
        edge_464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 18), self_463, 'edge')
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 30), list_465, int_466)
        
        # Applying the binary operator '*' (line 109)
        result_mul_467 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 18), '*', edge_464, list_465)
        
        list_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), list_474, result_mul_467)
        # Assigning a type to the variable 'scores' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'scores', list_474)
        
        
        # Call to xrange(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'self', False)
        # Obtaining the member 'edge' of a type (line 111)
        edge_477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 27), self_476, 'edge')
        # Processing the call keyword arguments (line 111)
        kwargs_478 = {}
        # Getting the type of 'xrange' (line 111)
        xrange_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'xrange', False)
        # Calling xrange(args, kwargs) (line 111)
        xrange_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), xrange_475, *[edge_477], **kwargs_478)
        
        # Assigning a type to the variable 'xrange_call_result_479' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'xrange_call_result_479', xrange_call_result_479)
        # Testing if the for loop is going to be iterated (line 111)
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 8), xrange_call_result_479)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 8), xrange_call_result_479):
            # Getting the type of the for loop variable (line 111)
            for_loop_var_480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 8), xrange_call_result_479)
            # Assigning a type to the variable 'rown' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'rown', for_loop_var_480)
            # SSA begins for a for statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 112):
            
            # Assigning a Subscript to a Name (line 112):
            
            # Obtaining the type of the subscript
            # Getting the type of 'rown' (line 112)
            rown_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'rown')
            # Getting the type of 'self' (line 112)
            self_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'self')
            # Obtaining the member '__board' of a type (line 112)
            board_483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 18), self_482, '__board')
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 18), board_483, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 112, 18), getitem___484, rown_481)
            
            # Assigning a type to the variable 'row' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'row', subscript_call_result_485)
            
            # Call to doRow(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'row' (line 113)
            row_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'row', False)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to xrange(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'self' (line 113)
            self_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 55), 'self', False)
            # Obtaining the member 'edge' of a type (line 113)
            edge_494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 55), self_493, 'edge')
            # Processing the call keyword arguments (line 113)
            kwargs_495 = {}
            # Getting the type of 'xrange' (line 113)
            xrange_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'xrange', False)
            # Calling xrange(args, kwargs) (line 113)
            xrange_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 113, 48), xrange_492, *[edge_494], **kwargs_495)
            
            comprehension_497 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 29), xrange_call_result_496)
            # Assigning a type to the variable 'i' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'i', comprehension_497)
            
            # Obtaining an instance of the builtin type 'tuple' (line 113)
            tuple_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 113)
            # Adding element type (line 113)
            # Getting the type of 'rown' (line 113)
            rown_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'rown', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 30), tuple_489, rown_490)
            # Adding element type (line 113)
            # Getting the type of 'i' (line 113)
            i_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 36), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 30), tuple_489, i_491)
            
            list_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 29), list_498, tuple_489)
            # Getting the type of 'player' (line 113)
            player_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 68), 'player', False)
            # Getting the type of 'scores' (line 113)
            scores_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 76), 'scores', False)
            # Processing the call keyword arguments (line 113)
            kwargs_501 = {}
            # Getting the type of 'self' (line 113)
            self_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self', False)
            # Obtaining the member 'doRow' of a type (line 113)
            doRow_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_486, 'doRow')
            # Calling doRow(args, kwargs) (line 113)
            doRow_call_result_502 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), doRow_487, *[row_488, list_498, player_499, scores_500], **kwargs_501)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to xrange(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'self', False)
        # Obtaining the member 'edge' of a type (line 115)
        edge_505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), self_504, 'edge')
        # Processing the call keyword arguments (line 115)
        kwargs_506 = {}
        # Getting the type of 'xrange' (line 115)
        xrange_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'xrange', False)
        # Calling xrange(args, kwargs) (line 115)
        xrange_call_result_507 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), xrange_503, *[edge_505], **kwargs_506)
        
        # Assigning a type to the variable 'xrange_call_result_507' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'xrange_call_result_507', xrange_call_result_507)
        # Testing if the for loop is going to be iterated (line 115)
        # Testing the type of a for loop iterable (line 115)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), xrange_call_result_507)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 115, 8), xrange_call_result_507):
            # Getting the type of the for loop variable (line 115)
            for_loop_var_508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), xrange_call_result_507)
            # Assigning a type to the variable 'coln' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'coln', for_loop_var_508)
            # SSA begins for a for statement (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a ListComp to a Name (line 116):
            
            # Assigning a ListComp to a Name (line 116):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'self' (line 116)
            self_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 40), 'self')
            # Obtaining the member '__board' of a type (line 116)
            board_514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 40), self_513, '__board')
            comprehension_515 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), board_514)
            # Assigning a type to the variable 'row' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'row', comprehension_515)
            
            # Obtaining the type of the subscript
            # Getting the type of 'coln' (line 116)
            coln_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'coln')
            # Getting the type of 'row' (line 116)
            row_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'row')
            # Obtaining the member '__getitem__' of a type (line 116)
            getitem___511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), row_510, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 116)
            subscript_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 116, 19), getitem___511, coln_509)
            
            list_516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_516, subscript_call_result_512)
            # Assigning a type to the variable 'col' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'col', list_516)
            
            # Call to doRow(...): (line 117)
            # Processing the call arguments (line 117)
            # Getting the type of 'col' (line 117)
            col_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'col', False)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to xrange(...): (line 117)
            # Processing the call arguments (line 117)
            # Getting the type of 'self' (line 117)
            self_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 55), 'self', False)
            # Obtaining the member 'edge' of a type (line 117)
            edge_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 55), self_524, 'edge')
            # Processing the call keyword arguments (line 117)
            kwargs_526 = {}
            # Getting the type of 'xrange' (line 117)
            xrange_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 48), 'xrange', False)
            # Calling xrange(args, kwargs) (line 117)
            xrange_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 117, 48), xrange_523, *[edge_525], **kwargs_526)
            
            comprehension_528 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), xrange_call_result_527)
            # Assigning a type to the variable 'i' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'i', comprehension_528)
            
            # Obtaining an instance of the builtin type 'tuple' (line 117)
            tuple_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 117)
            # Adding element type (line 117)
            # Getting the type of 'i' (line 117)
            i_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 30), tuple_520, i_521)
            # Adding element type (line 117)
            # Getting the type of 'coln' (line 117)
            coln_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'coln', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 30), tuple_520, coln_522)
            
            list_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), list_529, tuple_520)
            # Getting the type of 'player' (line 117)
            player_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 68), 'player', False)
            # Getting the type of 'scores' (line 117)
            scores_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 76), 'scores', False)
            # Processing the call keyword arguments (line 117)
            kwargs_532 = {}
            # Getting the type of 'self' (line 117)
            self_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'self', False)
            # Obtaining the member 'doRow' of a type (line 117)
            doRow_518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), self_517, 'doRow')
            # Calling doRow(args, kwargs) (line 117)
            doRow_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), doRow_518, *[col_519, list_529, player_530, scores_531], **kwargs_532)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a ListComp to a Name (line 119):
        
        # Assigning a ListComp to a Name (line 119):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'self' (line 119)
        self_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'self', False)
        # Obtaining the member 'edge' of a type (line 119)
        edge_539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 42), self_538, 'edge')
        # Processing the call keyword arguments (line 119)
        kwargs_540 = {}
        # Getting the type of 'xrange' (line 119)
        xrange_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'xrange', False)
        # Calling xrange(args, kwargs) (line 119)
        xrange_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 119, 35), xrange_537, *[edge_539], **kwargs_540)
        
        comprehension_542 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 19), xrange_call_result_541)
        # Assigning a type to the variable 'i' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'i', comprehension_542)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        # Getting the type of 'i' (line 119)
        i_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), tuple_534, i_535)
        # Adding element type (line 119)
        # Getting the type of 'i' (line 119)
        i_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), tuple_534, i_536)
        
        list_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 19), list_543, tuple_534)
        # Assigning a type to the variable 'indices' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'indices', list_543)
        
        # Assigning a ListComp to a Name (line 120):
        
        # Assigning a ListComp to a Name (line 120):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 49), 'self', False)
        # Obtaining the member 'edge' of a type (line 120)
        edge_554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 49), self_553, 'edge')
        # Processing the call keyword arguments (line 120)
        kwargs_555 = {}
        # Getting the type of 'xrange' (line 120)
        xrange_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 42), 'xrange', False)
        # Calling xrange(args, kwargs) (line 120)
        xrange_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 120, 42), xrange_552, *[edge_554], **kwargs_555)
        
        comprehension_557 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 14), xrange_call_result_556)
        # Assigning a type to the variable 'i' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'i', comprehension_557)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 120)
        i_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'i')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 120)
        i_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'i')
        # Getting the type of 'self' (line 120)
        self_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'self')
        # Obtaining the member '__board' of a type (line 120)
        board_547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 14), self_546, '__board')
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 14), board_547, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_549 = invoke(stypy.reporting.localization.Localization(__file__, 120, 14), getitem___548, i_545)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 14), subscript_call_result_549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 120, 14), getitem___550, i_544)
        
        list_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 14), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 14), list_558, subscript_call_result_551)
        # Assigning a type to the variable 'ld' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'ld', list_558)
        
        # Call to doRow(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'ld' (line 121)
        ld_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'ld', False)
        # Getting the type of 'indices' (line 121)
        indices_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'indices', False)
        # Getting the type of 'player' (line 121)
        player_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'player', False)
        # Getting the type of 'scores' (line 121)
        scores_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'scores', False)
        # Processing the call keyword arguments (line 121)
        kwargs_565 = {}
        # Getting the type of 'self' (line 121)
        self_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self', False)
        # Obtaining the member 'doRow' of a type (line 121)
        doRow_560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_559, 'doRow')
        # Calling doRow(args, kwargs) (line 121)
        doRow_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), doRow_560, *[ld_561, indices_562, player_563, scores_564], **kwargs_565)
        
        
        # Getting the type of 'indices' (line 123)
        indices_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'indices')
        # Assigning a type to the variable 'indices_567' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'indices_567', indices_567)
        # Testing if the for loop is going to be iterated (line 123)
        # Testing the type of a for loop iterable (line 123)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 8), indices_567)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 123, 8), indices_567):
            # Getting the type of the for loop variable (line 123)
            for_loop_var_568 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 8), indices_567)
            # Assigning a type to the variable 'rown' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'rown', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), for_loop_var_568, 2, 0))
            # Assigning a type to the variable 'coln' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'coln', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), for_loop_var_568, 2, 1))
            # SSA begins for a for statement (line 123)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'rown' (line 124)
            rown_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'rown')
            # Getting the type of 'scores' (line 124)
            scores_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'scores')
            # Obtaining the member '__getitem__' of a type (line 124)
            getitem___571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), scores_570, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 124)
            subscript_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), getitem___571, rown_569)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'coln' (line 124)
            coln_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'coln')
            
            # Obtaining the type of the subscript
            # Getting the type of 'rown' (line 124)
            rown_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'rown')
            # Getting the type of 'scores' (line 124)
            scores_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'scores')
            # Obtaining the member '__getitem__' of a type (line 124)
            getitem___576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), scores_575, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 124)
            subscript_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), getitem___576, rown_574)
            
            # Obtaining the member '__getitem__' of a type (line 124)
            getitem___578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), subscript_call_result_577, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 124)
            subscript_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), getitem___578, coln_573)
            
            int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 34), 'int')
            # Applying the binary operator '+=' (line 124)
            result_iadd_581 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 12), '+=', subscript_call_result_579, int_580)
            
            # Obtaining the type of the subscript
            # Getting the type of 'rown' (line 124)
            rown_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'rown')
            # Getting the type of 'scores' (line 124)
            scores_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'scores')
            # Obtaining the member '__getitem__' of a type (line 124)
            getitem___584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), scores_583, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 124)
            subscript_call_result_585 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), getitem___584, rown_582)
            
            # Getting the type of 'coln' (line 124)
            coln_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'coln')
            # Storing an element on a container (line 124)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 12), subscript_call_result_585, (coln_586, result_iadd_581))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a ListComp to a Name (line 127):
        
        # Assigning a ListComp to a Name (line 127):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'self' (line 127)
        self_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 60), 'self', False)
        # Obtaining the member 'edge' of a type (line 127)
        edge_597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 60), self_596, 'edge')
        # Processing the call keyword arguments (line 127)
        kwargs_598 = {}
        # Getting the type of 'xrange' (line 127)
        xrange_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 53), 'xrange', False)
        # Calling xrange(args, kwargs) (line 127)
        xrange_call_result_599 = invoke(stypy.reporting.localization.Localization(__file__, 127, 53), xrange_595, *[edge_597], **kwargs_598)
        
        comprehension_600 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 19), xrange_call_result_599)
        # Assigning a type to the variable 'i' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'i', comprehension_600)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        # Getting the type of 'i' (line 127)
        i_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), tuple_587, i_588)
        # Adding element type (line 127)
        # Getting the type of 'self' (line 127)
        self_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'self')
        # Obtaining the member 'edge' of a type (line 127)
        edge_590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), self_589, 'edge')
        int_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 36), 'int')
        # Applying the binary operator '-' (line 127)
        result_sub_592 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 24), '-', edge_590, int_591)
        
        # Getting the type of 'i' (line 127)
        i_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'i')
        # Applying the binary operator '-' (line 127)
        result_sub_594 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 23), '-', result_sub_592, i_593)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), tuple_587, result_sub_594)
        
        list_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 19), list_601, tuple_587)
        # Assigning a type to the variable 'indices' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'indices', list_601)
        
        # Assigning a ListComp to a Name (line 128):
        
        # Assigning a ListComp to a Name (line 128):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'self' (line 128)
        self_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 67), 'self', False)
        # Obtaining the member 'edge' of a type (line 128)
        edge_617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 67), self_616, 'edge')
        # Processing the call keyword arguments (line 128)
        kwargs_618 = {}
        # Getting the type of 'xrange' (line 128)
        xrange_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 60), 'xrange', False)
        # Calling xrange(args, kwargs) (line 128)
        xrange_call_result_619 = invoke(stypy.reporting.localization.Localization(__file__, 128, 60), xrange_615, *[edge_617], **kwargs_618)
        
        comprehension_620 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 14), xrange_call_result_619)
        # Assigning a type to the variable 'i' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'i', comprehension_620)
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 128)
        self_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'self')
        # Obtaining the member 'edge' of a type (line 128)
        edge_603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), self_602, 'edge')
        int_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 43), 'int')
        # Applying the binary operator '-' (line 128)
        result_sub_605 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 31), '-', edge_603, int_604)
        
        # Getting the type of 'i' (line 128)
        i_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 48), 'i')
        # Applying the binary operator '-' (line 128)
        result_sub_607 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 30), '-', result_sub_605, i_606)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 128)
        i_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'i')
        # Getting the type of 'self' (line 128)
        self_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'self')
        # Obtaining the member '__board' of a type (line 128)
        board_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 14), self_609, '__board')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 14), board_610, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 128, 14), getitem___611, i_608)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 14), subscript_call_result_612, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 128, 14), getitem___613, result_sub_607)
        
        list_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 14), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 14), list_621, subscript_call_result_614)
        # Assigning a type to the variable 'rd' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'rd', list_621)
        
        # Call to doRow(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'rd' (line 129)
        rd_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'rd', False)
        # Getting the type of 'indices' (line 129)
        indices_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'indices', False)
        # Getting the type of 'player' (line 129)
        player_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'player', False)
        # Getting the type of 'scores' (line 129)
        scores_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'scores', False)
        # Processing the call keyword arguments (line 129)
        kwargs_628 = {}
        # Getting the type of 'self' (line 129)
        self_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'doRow' of a type (line 129)
        doRow_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_622, 'doRow')
        # Calling doRow(args, kwargs) (line 129)
        doRow_call_result_629 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), doRow_623, *[rd_624, indices_625, player_626, scores_627], **kwargs_628)
        
        
        # Getting the type of 'indices' (line 131)
        indices_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'indices')
        # Assigning a type to the variable 'indices_630' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'indices_630', indices_630)
        # Testing if the for loop is going to be iterated (line 131)
        # Testing the type of a for loop iterable (line 131)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 131, 8), indices_630)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 131, 8), indices_630):
            # Getting the type of the for loop variable (line 131)
            for_loop_var_631 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 131, 8), indices_630)
            # Assigning a type to the variable 'rown' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'rown', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 8), for_loop_var_631, 2, 0))
            # Assigning a type to the variable 'coln' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'coln', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 8), for_loop_var_631, 2, 1))
            # SSA begins for a for statement (line 131)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'rown' (line 132)
            rown_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'rown')
            # Getting the type of 'scores' (line 132)
            scores_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'scores')
            # Obtaining the member '__getitem__' of a type (line 132)
            getitem___634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), scores_633, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 132)
            subscript_call_result_635 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), getitem___634, rown_632)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'coln' (line 132)
            coln_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'coln')
            
            # Obtaining the type of the subscript
            # Getting the type of 'rown' (line 132)
            rown_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'rown')
            # Getting the type of 'scores' (line 132)
            scores_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'scores')
            # Obtaining the member '__getitem__' of a type (line 132)
            getitem___639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), scores_638, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 132)
            subscript_call_result_640 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), getitem___639, rown_637)
            
            # Obtaining the member '__getitem__' of a type (line 132)
            getitem___641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), subscript_call_result_640, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 132)
            subscript_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), getitem___641, coln_636)
            
            int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'int')
            # Applying the binary operator '+=' (line 132)
            result_iadd_644 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 12), '+=', subscript_call_result_642, int_643)
            
            # Obtaining the type of the subscript
            # Getting the type of 'rown' (line 132)
            rown_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'rown')
            # Getting the type of 'scores' (line 132)
            scores_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'scores')
            # Obtaining the member '__getitem__' of a type (line 132)
            getitem___647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), scores_646, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 132)
            subscript_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), getitem___647, rown_645)
            
            # Getting the type of 'coln' (line 132)
            coln_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'coln')
            # Storing an element on a container (line 132)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), subscript_call_result_648, (coln_649, result_iadd_644))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a List to a Name (line 134):
        
        # Assigning a List to a Name (line 134):
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        
        # Assigning a type to the variable 'scorelist' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'scorelist', list_650)
        
        
        # Call to xrange(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'self' (line 135)
        self_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'self', False)
        # Obtaining the member 'edge' of a type (line 135)
        edge_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 27), self_652, 'edge')
        # Processing the call keyword arguments (line 135)
        kwargs_654 = {}
        # Getting the type of 'xrange' (line 135)
        xrange_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'xrange', False)
        # Calling xrange(args, kwargs) (line 135)
        xrange_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 135, 20), xrange_651, *[edge_653], **kwargs_654)
        
        # Assigning a type to the variable 'xrange_call_result_655' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'xrange_call_result_655', xrange_call_result_655)
        # Testing if the for loop is going to be iterated (line 135)
        # Testing the type of a for loop iterable (line 135)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 8), xrange_call_result_655)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 135, 8), xrange_call_result_655):
            # Getting the type of the for loop variable (line 135)
            for_loop_var_656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 8), xrange_call_result_655)
            # Assigning a type to the variable 'rown' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'rown', for_loop_var_656)
            # SSA begins for a for statement (line 135)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to xrange(...): (line 136)
            # Processing the call arguments (line 136)
            # Getting the type of 'self' (line 136)
            self_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'self', False)
            # Obtaining the member 'edge' of a type (line 136)
            edge_659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 31), self_658, 'edge')
            # Processing the call keyword arguments (line 136)
            kwargs_660 = {}
            # Getting the type of 'xrange' (line 136)
            xrange_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'xrange', False)
            # Calling xrange(args, kwargs) (line 136)
            xrange_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), xrange_657, *[edge_659], **kwargs_660)
            
            # Assigning a type to the variable 'xrange_call_result_661' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'xrange_call_result_661', xrange_call_result_661)
            # Testing if the for loop is going to be iterated (line 136)
            # Testing the type of a for loop iterable (line 136)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 12), xrange_call_result_661)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 136, 12), xrange_call_result_661):
                # Getting the type of the for loop variable (line 136)
                for_loop_var_662 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 12), xrange_call_result_661)
                # Assigning a type to the variable 'coln' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'coln', for_loop_var_662)
                # SSA begins for a for statement (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'coln' (line 137)
                coln_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'coln')
                
                # Obtaining the type of the subscript
                # Getting the type of 'rown' (line 137)
                rown_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'rown')
                # Getting the type of 'self' (line 137)
                self_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'self')
                # Obtaining the member '__board' of a type (line 137)
                board_666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 20), self_665, '__board')
                # Obtaining the member '__getitem__' of a type (line 137)
                getitem___667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 20), board_666, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 137)
                subscript_call_result_668 = invoke(stypy.reporting.localization.Localization(__file__, 137, 20), getitem___667, rown_664)
                
                # Obtaining the member '__getitem__' of a type (line 137)
                getitem___669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 20), subscript_call_result_668, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 137)
                subscript_call_result_670 = invoke(stypy.reporting.localization.Localization(__file__, 137, 20), getitem___669, coln_663)
                
                int_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 48), 'int')
                # Applying the binary operator '==' (line 137)
                result_eq_672 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 20), '==', subscript_call_result_670, int_671)
                
                # Testing if the type of an if condition is none (line 137)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 137, 16), result_eq_672):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 137)
                    if_condition_673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 16), result_eq_672)
                    # Assigning a type to the variable 'if_condition_673' (line 137)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'if_condition_673', if_condition_673)
                    # SSA begins for if statement (line 137)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 138)
                    # Processing the call arguments (line 138)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 138)
                    tuple_676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 38), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 138)
                    # Adding element type (line 138)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'coln' (line 138)
                    coln_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 51), 'coln', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'rown' (line 138)
                    rown_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 45), 'rown', False)
                    # Getting the type of 'scores' (line 138)
                    scores_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 38), 'scores', False)
                    # Obtaining the member '__getitem__' of a type (line 138)
                    getitem___680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 38), scores_679, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
                    subscript_call_result_681 = invoke(stypy.reporting.localization.Localization(__file__, 138, 38), getitem___680, rown_678)
                    
                    # Obtaining the member '__getitem__' of a type (line 138)
                    getitem___682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 38), subscript_call_result_681, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
                    subscript_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 138, 38), getitem___682, coln_677)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 38), tuple_676, subscript_call_result_683)
                    # Adding element type (line 138)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 138)
                    tuple_684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 59), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 138)
                    # Adding element type (line 138)
                    # Getting the type of 'rown' (line 138)
                    rown_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 59), 'rown', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 59), tuple_684, rown_685)
                    # Adding element type (line 138)
                    # Getting the type of 'coln' (line 138)
                    coln_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 65), 'coln', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 59), tuple_684, coln_686)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 38), tuple_676, tuple_684)
                    
                    # Processing the call keyword arguments (line 138)
                    kwargs_687 = {}
                    # Getting the type of 'scorelist' (line 138)
                    scorelist_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'scorelist', False)
                    # Obtaining the member 'append' of a type (line 138)
                    append_675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 20), scorelist_674, 'append')
                    # Calling append(args, kwargs) (line 138)
                    append_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 138, 20), append_675, *[tuple_676], **kwargs_687)
                    
                    # SSA join for if statement (line 137)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to sort(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_691 = {}
        # Getting the type of 'scorelist' (line 139)
        scorelist_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'scorelist', False)
        # Obtaining the member 'sort' of a type (line 139)
        sort_690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), scorelist_689, 'sort')
        # Calling sort(args, kwargs) (line 139)
        sort_call_result_692 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), sort_690, *[], **kwargs_691)
        
        
        # Call to reverse(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_695 = {}
        # Getting the type of 'scorelist' (line 140)
        scorelist_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'scorelist', False)
        # Obtaining the member 'reverse' of a type (line 140)
        reverse_694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), scorelist_693, 'reverse')
        # Calling reverse(args, kwargs) (line 140)
        reverse_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), reverse_694, *[], **kwargs_695)
        
        
        # Assigning a ListComp to a Name (line 142):
        
        # Assigning a ListComp to a Name (line 142):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'scorelist' (line 142)
        scorelist_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'scorelist')
        comprehension_711 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), scorelist_710)
        # Assigning a type to the variable 'x' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'x', comprehension_711)
        
        
        # Obtaining the type of the subscript
        int_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 47), 'int')
        # Getting the type of 'x' (line 142)
        x_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'x')
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 45), x_699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 142, 45), getitem___700, int_698)
        
        
        # Obtaining the type of the subscript
        int_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 66), 'int')
        
        # Obtaining the type of the subscript
        int_703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 63), 'int')
        # Getting the type of 'scorelist' (line 142)
        scorelist_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 53), 'scorelist')
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 53), scorelist_704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 142, 53), getitem___705, int_703)
        
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 53), subscript_call_result_706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_708 = invoke(stypy.reporting.localization.Localization(__file__, 142, 53), getitem___707, int_702)
        
        # Applying the binary operator '==' (line 142)
        result_eq_709 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 45), '==', subscript_call_result_701, subscript_call_result_708)
        
        # Getting the type of 'x' (line 142)
        x_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'x')
        list_712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), list_712, x_697)
        # Assigning a type to the variable 'scorelist' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'scorelist', list_712)
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        
        # Obtaining the type of the subscript
        int_714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 32), 'int')
        
        # Obtaining the type of the subscript
        int_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'int')
        
        # Obtaining the type of the subscript
        int_716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 26), 'int')
        # Getting the type of 'scorelist' (line 149)
        scorelist_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'scorelist')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), scorelist_717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), getitem___718, int_716)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), subscript_call_result_719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_721 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), getitem___720, int_715)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), subscript_call_result_721, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_723 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), getitem___722, int_714)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 16), tuple_713, subscript_call_result_723)
        # Adding element type (line 149)
        
        # Obtaining the type of the subscript
        int_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 52), 'int')
        
        # Obtaining the type of the subscript
        int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 49), 'int')
        
        # Obtaining the type of the subscript
        int_726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 46), 'int')
        # Getting the type of 'scorelist' (line 149)
        scorelist_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'scorelist')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 36), scorelist_727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_729 = invoke(stypy.reporting.localization.Localization(__file__, 149, 36), getitem___728, int_726)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 36), subscript_call_result_729, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 149, 36), getitem___730, int_725)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 36), subscript_call_result_731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_733 = invoke(stypy.reporting.localization.Localization(__file__, 149, 36), getitem___732, int_724)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 16), tuple_713, subscript_call_result_733)
        
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', tuple_713)
        
        # ################# End of 'makeAImove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'makeAImove' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'makeAImove'
        return stypy_return_type_734


# Assigning a type to the variable 'rectBoard' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'rectBoard', rectBoard)

@norecursion
def aigame(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
    int_736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'int')
    int_737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 36), 'int')
    defaults = [int_735, int_736, int_737]
    # Create a new context for function 'aigame'
    module_type_store = module_type_store.open_function_context('aigame', 152, 0, False)
    
    # Passed parameters checking function
    aigame.stypy_localization = localization
    aigame.stypy_type_of_self = None
    aigame.stypy_type_store = module_type_store
    aigame.stypy_function_name = 'aigame'
    aigame.stypy_param_names_list = ['size', 'turn', 'players']
    aigame.stypy_varargs_param_name = None
    aigame.stypy_kwargs_param_name = None
    aigame.stypy_call_defaults = defaults
    aigame.stypy_call_varargs = varargs
    aigame.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'aigame', ['size', 'turn', 'players'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'aigame', localization, ['size', 'turn', 'players'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'aigame(...)' code ##################

    
    # Assigning a Call to a Name (line 153):
    
    # Assigning a Call to a Name (line 153):
    
    # Call to rectBoard(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'size' (line 153)
    size_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'size', False)
    # Processing the call keyword arguments (line 153)
    kwargs_740 = {}
    # Getting the type of 'rectBoard' (line 153)
    rectBoard_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'rectBoard', False)
    # Calling rectBoard(args, kwargs) (line 153)
    rectBoard_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), rectBoard_738, *[size_739], **kwargs_740)
    
    # Assigning a type to the variable 'b' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'b', rectBoard_call_result_741)
    
    
    # Evaluating a boolean operation
    
    
    # Call to isfull(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_744 = {}
    # Getting the type of 'b' (line 155)
    b_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'b', False)
    # Obtaining the member 'isfull' of a type (line 155)
    isfull_743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), b_742, 'isfull')
    # Calling isfull(args, kwargs) (line 155)
    isfull_call_result_745 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), isfull_743, *[], **kwargs_744)
    
    # Applying the 'not' unary operator (line 155)
    result_not__746 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 12), 'not', isfull_call_result_745)
    
    
    
    # Call to isvictory(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_749 = {}
    # Getting the type of 'b' (line 155)
    b_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'b', False)
    # Obtaining the member 'isvictory' of a type (line 155)
    isvictory_748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 33), b_747, 'isvictory')
    # Calling isvictory(args, kwargs) (line 155)
    isvictory_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 155, 33), isvictory_748, *[], **kwargs_749)
    
    int_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 50), 'int')
    # Applying the binary operator '==' (line 155)
    result_eq_752 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 33), '==', isvictory_call_result_750, int_751)
    
    # Applying the binary operator 'and' (line 155)
    result_and_keyword_753 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), 'and', result_not__746, result_eq_752)
    
    # Assigning a type to the variable 'result_and_keyword_753' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'result_and_keyword_753', result_and_keyword_753)
    # Testing if the while is going to be iterated (line 155)
    # Testing the type of an if condition (line 155)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 4), result_and_keyword_753)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 155, 4), result_and_keyword_753):
        # SSA begins for while statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'turn' (line 156)
        turn_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'turn')
        int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'int')
        # Applying the binary operator '==' (line 156)
        result_eq_756 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 12), '==', turn_754, int_755)
        
        # Testing if the type of an if condition is none (line 156)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 156, 8), result_eq_756):
            
            # Assigning a Call to a Tuple (line 165):
            
            # Assigning a Call to a Name:
            
            # Call to makeAImove(...): (line 165)
            # Processing the call arguments (line 165)
            # Getting the type of 'turn' (line 165)
            turn_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'turn', False)
            # Processing the call keyword arguments (line 165)
            kwargs_780 = {}
            # Getting the type of 'b' (line 165)
            b_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'b', False)
            # Obtaining the member 'makeAImove' of a type (line 165)
            makeAImove_778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 19), b_777, 'makeAImove')
            # Calling makeAImove(args, kwargs) (line 165)
            makeAImove_call_result_781 = invoke(stypy.reporting.localization.Localization(__file__, 165, 19), makeAImove_778, *[turn_779], **kwargs_780)
            
            # Assigning a type to the variable 'call_assignment_4' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_4', makeAImove_call_result_781)
            
            # Assigning a Call to a Name (line 165):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 165)
            call_assignment_4_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_783 = stypy_get_value_from_tuple(call_assignment_4_782, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_5' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_5', stypy_get_value_from_tuple_call_result_783)
            
            # Assigning a Name to a Name (line 165):
            # Getting the type of 'call_assignment_5' (line 165)
            call_assignment_5_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_5')
            # Assigning a type to the variable 'r' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'r', call_assignment_5_784)
            
            # Assigning a Call to a Name (line 165):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 165)
            call_assignment_4_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_786 = stypy_get_value_from_tuple(call_assignment_4_785, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_6', stypy_get_value_from_tuple_call_result_786)
            
            # Assigning a Name to a Name (line 165):
            # Getting the type of 'call_assignment_6' (line 165)
            call_assignment_6_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_6')
            # Assigning a type to the variable 'c' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'c', call_assignment_6_787)
            
            # Call to assign(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'r' (line 166)
            r_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'r', False)
            # Getting the type of 'c' (line 166)
            c_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'c', False)
            # Getting the type of 'turn' (line 166)
            turn_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'turn', False)
            # Processing the call keyword arguments (line 166)
            kwargs_793 = {}
            # Getting the type of 'b' (line 166)
            b_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'b', False)
            # Obtaining the member 'assign' of a type (line 166)
            assign_789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), b_788, 'assign')
            # Calling assign(args, kwargs) (line 166)
            assign_call_result_794 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), assign_789, *[r_790, c_791, turn_792], **kwargs_793)
            
            
            # Getting the type of 'turn' (line 167)
            turn_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'turn')
            # Getting the type of 'players' (line 167)
            players_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'players')
            # Applying the binary operator '==' (line 167)
            result_eq_797 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 16), '==', turn_795, players_796)
            
            # Testing if the type of an if condition is none (line 167)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 12), result_eq_797):
                
                # Getting the type of 'turn' (line 170)
                turn_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn')
                int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 24), 'int')
                # Applying the binary operator '+=' (line 170)
                result_iadd_802 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), '+=', turn_800, int_801)
                # Assigning a type to the variable 'turn' (line 170)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn', result_iadd_802)
                
            else:
                
                # Testing the type of an if condition (line 167)
                if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 12), result_eq_797)
                # Assigning a type to the variable 'if_condition_798' (line 167)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'if_condition_798', if_condition_798)
                # SSA begins for if statement (line 167)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Name (line 168):
                
                # Assigning a Num to a Name (line 168):
                int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'int')
                # Assigning a type to the variable 'turn' (line 168)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'turn', int_799)
                # SSA branch for the else part of an if statement (line 167)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'turn' (line 170)
                turn_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn')
                int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 24), 'int')
                # Applying the binary operator '+=' (line 170)
                result_iadd_802 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), '+=', turn_800, int_801)
                # Assigning a type to the variable 'turn' (line 170)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn', result_iadd_802)
                
                # SSA join for if statement (line 167)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 156)
            if_condition_757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_eq_756)
            # Assigning a type to the variable 'if_condition_757' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_757', if_condition_757)
            # SSA begins for if statement (line 156)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 160):
            
            # Assigning a Call to a Name:
            
            # Call to makeAImove(...): (line 160)
            # Processing the call arguments (line 160)
            # Getting the type of 'turn' (line 160)
            turn_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'turn', False)
            # Processing the call keyword arguments (line 160)
            kwargs_761 = {}
            # Getting the type of 'b' (line 160)
            b_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'b', False)
            # Obtaining the member 'makeAImove' of a type (line 160)
            makeAImove_759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), b_758, 'makeAImove')
            # Calling makeAImove(args, kwargs) (line 160)
            makeAImove_call_result_762 = invoke(stypy.reporting.localization.Localization(__file__, 160, 19), makeAImove_759, *[turn_760], **kwargs_761)
            
            # Assigning a type to the variable 'call_assignment_1' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'call_assignment_1', makeAImove_call_result_762)
            
            # Assigning a Call to a Name (line 160):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 160)
            call_assignment_1_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_764 = stypy_get_value_from_tuple(call_assignment_1_763, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_2' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'call_assignment_2', stypy_get_value_from_tuple_call_result_764)
            
            # Assigning a Name to a Name (line 160):
            # Getting the type of 'call_assignment_2' (line 160)
            call_assignment_2_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'call_assignment_2')
            # Assigning a type to the variable 'r' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'r', call_assignment_2_765)
            
            # Assigning a Call to a Name (line 160):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 160)
            call_assignment_1_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_767 = stypy_get_value_from_tuple(call_assignment_1_766, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_3' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'call_assignment_3', stypy_get_value_from_tuple_call_result_767)
            
            # Assigning a Name to a Name (line 160):
            # Getting the type of 'call_assignment_3' (line 160)
            call_assignment_3_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'call_assignment_3')
            # Assigning a type to the variable 'c' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'c', call_assignment_3_768)
            
            # Call to assign(...): (line 161)
            # Processing the call arguments (line 161)
            # Getting the type of 'r' (line 161)
            r_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'r', False)
            # Getting the type of 'c' (line 161)
            c_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'c', False)
            int_773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 27), 'int')
            # Processing the call keyword arguments (line 161)
            kwargs_774 = {}
            # Getting the type of 'b' (line 161)
            b_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'b', False)
            # Obtaining the member 'assign' of a type (line 161)
            assign_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), b_769, 'assign')
            # Calling assign(args, kwargs) (line 161)
            assign_call_result_775 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), assign_770, *[r_771, c_772, int_773], **kwargs_774)
            
            
            # Assigning a Num to a Name (line 162):
            
            # Assigning a Num to a Name (line 162):
            int_776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 19), 'int')
            # Assigning a type to the variable 'turn' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'turn', int_776)
            # SSA branch for the else part of an if statement (line 156)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 165):
            
            # Assigning a Call to a Name:
            
            # Call to makeAImove(...): (line 165)
            # Processing the call arguments (line 165)
            # Getting the type of 'turn' (line 165)
            turn_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'turn', False)
            # Processing the call keyword arguments (line 165)
            kwargs_780 = {}
            # Getting the type of 'b' (line 165)
            b_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'b', False)
            # Obtaining the member 'makeAImove' of a type (line 165)
            makeAImove_778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 19), b_777, 'makeAImove')
            # Calling makeAImove(args, kwargs) (line 165)
            makeAImove_call_result_781 = invoke(stypy.reporting.localization.Localization(__file__, 165, 19), makeAImove_778, *[turn_779], **kwargs_780)
            
            # Assigning a type to the variable 'call_assignment_4' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_4', makeAImove_call_result_781)
            
            # Assigning a Call to a Name (line 165):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 165)
            call_assignment_4_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_783 = stypy_get_value_from_tuple(call_assignment_4_782, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_5' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_5', stypy_get_value_from_tuple_call_result_783)
            
            # Assigning a Name to a Name (line 165):
            # Getting the type of 'call_assignment_5' (line 165)
            call_assignment_5_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_5')
            # Assigning a type to the variable 'r' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'r', call_assignment_5_784)
            
            # Assigning a Call to a Name (line 165):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 165)
            call_assignment_4_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_786 = stypy_get_value_from_tuple(call_assignment_4_785, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_6', stypy_get_value_from_tuple_call_result_786)
            
            # Assigning a Name to a Name (line 165):
            # Getting the type of 'call_assignment_6' (line 165)
            call_assignment_6_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'call_assignment_6')
            # Assigning a type to the variable 'c' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'c', call_assignment_6_787)
            
            # Call to assign(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'r' (line 166)
            r_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'r', False)
            # Getting the type of 'c' (line 166)
            c_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'c', False)
            # Getting the type of 'turn' (line 166)
            turn_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'turn', False)
            # Processing the call keyword arguments (line 166)
            kwargs_793 = {}
            # Getting the type of 'b' (line 166)
            b_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'b', False)
            # Obtaining the member 'assign' of a type (line 166)
            assign_789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), b_788, 'assign')
            # Calling assign(args, kwargs) (line 166)
            assign_call_result_794 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), assign_789, *[r_790, c_791, turn_792], **kwargs_793)
            
            
            # Getting the type of 'turn' (line 167)
            turn_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'turn')
            # Getting the type of 'players' (line 167)
            players_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'players')
            # Applying the binary operator '==' (line 167)
            result_eq_797 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 16), '==', turn_795, players_796)
            
            # Testing if the type of an if condition is none (line 167)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 12), result_eq_797):
                
                # Getting the type of 'turn' (line 170)
                turn_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn')
                int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 24), 'int')
                # Applying the binary operator '+=' (line 170)
                result_iadd_802 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), '+=', turn_800, int_801)
                # Assigning a type to the variable 'turn' (line 170)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn', result_iadd_802)
                
            else:
                
                # Testing the type of an if condition (line 167)
                if_condition_798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 12), result_eq_797)
                # Assigning a type to the variable 'if_condition_798' (line 167)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'if_condition_798', if_condition_798)
                # SSA begins for if statement (line 167)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Name (line 168):
                
                # Assigning a Num to a Name (line 168):
                int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'int')
                # Assigning a type to the variable 'turn' (line 168)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'turn', int_799)
                # SSA branch for the else part of an if statement (line 167)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'turn' (line 170)
                turn_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn')
                int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 24), 'int')
                # Applying the binary operator '+=' (line 170)
                result_iadd_802 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), '+=', turn_800, int_801)
                # Assigning a type to the variable 'turn' (line 170)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'turn', result_iadd_802)
                
                # SSA join for if statement (line 167)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 156)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for while statement (line 155)
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to isvictory(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_805 = {}
    # Getting the type of 'b' (line 174)
    b_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'b', False)
    # Obtaining the member 'isvictory' of a type (line 174)
    isvictory_804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), b_803, 'isvictory')
    # Calling isvictory(args, kwargs) (line 174)
    isvictory_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), isvictory_804, *[], **kwargs_805)
    
    int_807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 25), 'int')
    # Applying the binary operator '==' (line 174)
    result_eq_808 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 8), '==', isvictory_call_result_806, int_807)
    
    # Testing if the type of an if condition is none (line 174)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 174, 4), result_eq_808):
        pass
    else:
        
        # Testing the type of an if condition (line 174)
        if_condition_809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_eq_808)
        # Assigning a type to the variable 'if_condition_809' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_809', if_condition_809)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 174)
        module_type_store.open_ssa_branch('else')
        pass
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'aigame(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'aigame' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_810)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'aigame'
    return stypy_return_type_810

# Assigning a type to the variable 'aigame' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'aigame', aigame)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 180, 0, False)
    
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

    
    
    # Call to range(...): (line 181)
    # Processing the call arguments (line 181)
    int_812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 19), 'int')
    # Processing the call keyword arguments (line 181)
    kwargs_813 = {}
    # Getting the type of 'range' (line 181)
    range_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'range', False)
    # Calling range(args, kwargs) (line 181)
    range_call_result_814 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), range_811, *[int_812], **kwargs_813)
    
    # Assigning a type to the variable 'range_call_result_814' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'range_call_result_814', range_call_result_814)
    # Testing if the for loop is going to be iterated (line 181)
    # Testing the type of a for loop iterable (line 181)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 181, 4), range_call_result_814)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 181, 4), range_call_result_814):
        # Getting the type of the for loop variable (line 181)
        for_loop_var_815 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 181, 4), range_call_result_814)
        # Assigning a type to the variable 'i' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'i', for_loop_var_815)
        # SSA begins for a for statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to aigame(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_817 = {}
        # Getting the type of 'aigame' (line 182)
        aigame_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'aigame', False)
        # Calling aigame(args, kwargs) (line 182)
        aigame_call_result_818 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), aigame_816, *[], **kwargs_817)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 183)
    True_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', True_819)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_820)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_820

# Assigning a type to the variable 'run' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'run', run)

# Call to run(...): (line 186)
# Processing the call keyword arguments (line 186)
kwargs_822 = {}
# Getting the type of 'run' (line 186)
run_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'run', False)
# Calling run(args, kwargs) (line 186)
run_call_result_823 = invoke(stypy.reporting.localization.Localization(__file__, 186, 0), run_821, *[], **kwargs_822)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

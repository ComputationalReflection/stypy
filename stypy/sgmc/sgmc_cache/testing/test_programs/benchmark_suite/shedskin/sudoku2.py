
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # (c) Peter Goodspeed
2: # --- coriolinus@gmail.com
3: #
4: # sudoku solver
5: 
6: from math import ceil
7: from time import time
8: import sys
9: import os
10: 
11: 
12: def Relative(path):
13:     return os.path.join(os.path.dirname(__file__), path)
14: 
15: 
16: class bmp(object):
17:     def __init__(self, vals=9 * [True], n=-1):
18:         self.v = vals[0:9]
19:         if n >= 0: self.v[n] = not self.v[n]
20: 
21:     def __and__(self, other):
22:         return bmp([self.v[i] and other.v[i] for i in xrange(9)])
23: 
24:     def cnt(self):
25:         return len([i for i in self.v if i])
26: 
27: 
28: class boardRep(object):
29:     def __init__(self, board):
30:         self.__fields = list(board.final)
31: 
32:     def fields(self):
33:         return self.__fields
34: 
35:     def __eq__(self, other):
36:         return self.__fields == other.fields()
37: 
38:     def __ne__(self, other):
39:         return self.__fields != other.fields()
40: 
41:     def __hash__(self):
42:         rep = ""
43:         for i in xrange(9):
44:             for j in xrange(9):
45:                 rep += str(self.__fields[i][j])
46:         return hash(rep)
47: 
48: 
49: class board(object):
50:     notifyOnCompletion = True  # let the user know when you're done computing a game
51:     completeSearch = False  # search past the first solution
52: 
53:     def __init__(self):
54:         # final numbers: a 9 by 9 grid
55:         self.final = [9 * [0] for i in xrange(9)]
56:         self.rows = 9 * [bmp()]
57:         self.cols = 9 * [bmp()]
58:         self.cels = [3 * [bmp()] for i in xrange(3)]
59: 
60:         # statistics
61:         self.__turns = 0
62:         self.__backtracks = 0
63:         self.__starttime = 0
64:         self.__endtime = 0
65:         self.__status = 0
66:         self.__maxdepth = 0
67:         self.__openspaces = 81
68: 
69:         # a set of all solved boards discovered so far
70:         self.solutions = set()
71:         # a set of all boards examined--should help reduce the amount of search duplication
72:         self.examined = set()
73: 
74:     def fread(self, fn=''):
75:         # self.__init__()
76:         if fn == '':
77:             fn = raw_input("filename: ")
78:         f = file(fn, 'r')
79:         lines = f.readlines()
80:         for row in xrange(9):
81:             for digit in xrange(1, 10):
82:                 try:
83:                     self.setval(row, lines[row].index(str(digit)), digit)
84:                 except ValueError:
85:                     pass
86:         f.close()
87: 
88:     def setval(self, row, col, val):
89:         # add the number to the grid
90:         self.final[row][col] = val
91:         self.__openspaces -= 1
92: 
93:         # remove the number from the potential masks
94:         mask = bmp(n=val - 1)
95:         # rows and cols
96:         self.rows[row] = self.rows[row] & mask
97:         self.cols[col] = self.cols[col] & mask
98:         # cels
99:         cr = self.cell(row)
100:         cc = self.cell(col)
101:         self.cels[cr][cc] = self.cels[cr][cc] & mask
102: 
103:     def cell(self, num):
104:         return int(ceil((num + 1) / 3.0)) - 1
105: 
106:     def __str__(self):
107:         ret = ""
108:         for row in xrange(9):
109:             if row == 3 or row == 6: ret += (((3 * "---") + "+") * 3)[:-1] + "\n"
110:             for col in xrange(9):
111:                 if col == 3 or col == 6: ret += "|"
112:                 if self.final[row][col]:
113:                     c = str(self.final[row][col])
114:                 else:
115:                     c = " "
116:                 ret += " " + c + " "
117:             ret += "\n"
118:         return ret
119: 
120:     def solve(self, notify=True, completeSearch=False):
121:         if self.__status == 0:
122:             self.__status = 1
123:             self.__starttime = time()
124:             board.notifyOnCompletion = notify
125:             board.completeSearch = completeSearch
126:             self.__solve(self, 0)
127: 
128:     def openspaces(self):
129:         return self.__openspaces
130: 
131:     def __solve(self, _board, depth):
132:         if boardRep(_board) not in self.examined:
133:             self.examined.add(boardRep(_board))
134: 
135:             # check for solution condition:
136:             if _board.openspaces() <= 0:
137:                 self.solutions.add(boardRep(_board))
138:                 ##                                print 'solution:'
139:                 ##                                print _board
140:                 if depth == 0: self.onexit()
141:                 if not board.completeSearch:
142:                     self.onexit()
143: 
144:             else:
145:                 # update the statistics
146:                 self.__turns += 1
147:                 if depth > self.__maxdepth: self.__maxdepth = depth
148: 
149:                 # figure out the mincount
150:                 mincnt, coords = _board.findmincounts()
151:                 if mincnt <= 0:
152:                     self.__backtracks += 1
153:                     if depth == 0: self.onexit()
154:                 else:
155:                     # coords is a list of tuples of coordinates with equal, minimal counts
156:                     # of possible values. Try each of them in turn.
157:                     for row, col in coords:
158:                         # now we iterate through possible values to put in there
159:                         broken = False
160:                         for val in [i for i in xrange(9) if _board.mergemask(row, col).v[i] == True]:
161:                             if not board.completeSearch and self.__status == 2:
162:                                 broken = True
163:                                 break
164:                             val += 1
165:                             t = _board.clone()
166:                             t.setval(row, col, val)
167:                             self.__solve(t, depth + 1)
168:                         # if we broke out of the previous loop, we also want to break out of
169:                         # this one. unfortunately, "break 2" seems to be invalid syntax.
170:                         if broken: break
171:                         # else: didntBreak = True
172:                         # if not didntBreak: break
173: 
174:     def clone(self):
175:         ret = board()
176:         for row in xrange(9):
177:             for col in xrange(9):
178:                 if self.final[row][col]:
179:                     ret.setval(row, col, self.final[row][col])
180:         return ret
181: 
182:     def mergemask(self, row, col):
183:         return self.rows[row] & self.cols[col] & self.cels[self.cell(row)][self.cell(col)]
184: 
185:     def findmincounts(self):
186:         # compute the list of lenghths of merged masks
187:         masks = []
188:         for row in xrange(9):
189:             for col in xrange(9):
190:                 if self.final[row][col] == 0:
191:                     numallowed = self.mergemask(row, col).cnt()
192:                     masks.append((numallowed, row, col))
193:         # return the minimum number of allowed moves, and a list of cells which are
194:         # not currently occupied and which have that number of allowed moves
195:         return min(masks)[0], [(i[1], i[2]) for i in masks if i[0] == min(masks)[0]]
196: 
197:     def onexit(self):
198:         self.__endtime = time()
199:         self.__status = 2
200: 
201:         if board.notifyOnCompletion: pass  # print self.stats()['turns']
202: 
203:     def stats(self):
204:         if self.__status == 1:
205:             t = time() - self.__starttime
206:         else:
207:             t = self.__endtime - self.__starttime
208:         return {'max depth': self.__maxdepth, 'turns': self.__turns, 'backtracks': self.__backtracks,
209:                 'elapsed time': int(t), 'boards examined': len(self.examined),
210:                 'number of solutions': len(self.solutions)}
211: 
212: 
213: def main():
214:     puzzle = board()
215:     puzzle.fread(Relative('testdata/b6.pz'))
216:     ##    print puzzle
217:     puzzle.solve()
218: 
219: 
220: def run():
221:     main()
222:     return True
223: 
224: 
225: run()
226: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from math import ceil' statement (line 6)
try:
    from math import ceil

except:
    ceil = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'math', None, module_type_store, ['ceil'], [ceil])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from time import time' statement (line 7)
try:
    from time import time

except:
    time = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'time', None, module_type_store, ['time'], [time])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 12, 0, False)
    
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

    
    # Call to join(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to dirname(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of '__file__' (line 13)
    file___10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 40), '__file__', False)
    # Processing the call keyword arguments (line 13)
    kwargs_11 = {}
    # Getting the type of 'os' (line 13)
    os_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 13)
    path_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 24), os_7, 'path')
    # Obtaining the member 'dirname' of a type (line 13)
    dirname_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 24), path_8, 'dirname')
    # Calling dirname(args, kwargs) (line 13)
    dirname_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 13, 24), dirname_9, *[file___10], **kwargs_11)
    
    # Getting the type of 'path' (line 13)
    path_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 51), 'path', False)
    # Processing the call keyword arguments (line 13)
    kwargs_14 = {}
    # Getting the type of 'os' (line 13)
    os_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 13)
    path_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), os_4, 'path')
    # Obtaining the member 'join' of a type (line 13)
    join_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), path_5, 'join')
    # Calling join(args, kwargs) (line 13)
    join_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), join_6, *[dirname_call_result_12, path_13], **kwargs_14)
    
    # Assigning a type to the variable 'stypy_return_type' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type', join_call_result_15)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_16

# Assigning a type to the variable 'Relative' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'Relative', Relative)
# Declaration of the 'bmp' class

class bmp(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        # Getting the type of 'True' (line 17)
        True_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 33), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 32), list_18, True_19)
        
        # Applying the binary operator '*' (line 17)
        result_mul_20 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 28), '*', int_17, list_18)
        
        int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'int')
        defaults = [result_mul_20, int_21]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bmp.__init__', ['vals', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['vals', 'n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Subscript to a Attribute (line 18):
        
        # Assigning a Subscript to a Attribute (line 18):
        
        # Obtaining the type of the subscript
        int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
        int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'int')
        slice_24 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 18, 17), int_22, int_23, None)
        # Getting the type of 'vals' (line 18)
        vals_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'vals')
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), vals_25, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), getitem___26, slice_24)
        
        # Getting the type of 'self' (line 18)
        self_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'v' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_28, 'v', subscript_call_result_27)
        
        # Getting the type of 'n' (line 19)
        n_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'n')
        int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'int')
        # Applying the binary operator '>=' (line 19)
        result_ge_31 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 11), '>=', n_29, int_30)
        
        # Testing if the type of an if condition is none (line 19)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 19, 8), result_ge_31):
            pass
        else:
            
            # Testing the type of an if condition (line 19)
            if_condition_32 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 8), result_ge_31)
            # Assigning a type to the variable 'if_condition_32' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'if_condition_32', if_condition_32)
            # SSA begins for if statement (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a UnaryOp to a Subscript (line 19):
            
            # Assigning a UnaryOp to a Subscript (line 19):
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 19)
            n_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 42), 'n')
            # Getting the type of 'self' (line 19)
            self_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'self')
            # Obtaining the member 'v' of a type (line 19)
            v_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 35), self_34, 'v')
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 35), v_35, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 19, 35), getitem___36, n_33)
            
            # Applying the 'not' unary operator (line 19)
            result_not__38 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 31), 'not', subscript_call_result_37)
            
            # Getting the type of 'self' (line 19)
            self_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'self')
            # Obtaining the member 'v' of a type (line 19)
            v_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), self_39, 'v')
            # Getting the type of 'n' (line 19)
            n_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'n')
            # Storing an element on a container (line 19)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 19), v_40, (n_41, result_not__38))
            # SSA join for if statement (line 19)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __and__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__and__'
        module_type_store = module_type_store.open_function_context('__and__', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bmp.__and__.__dict__.__setitem__('stypy_localization', localization)
        bmp.__and__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bmp.__and__.__dict__.__setitem__('stypy_type_store', module_type_store)
        bmp.__and__.__dict__.__setitem__('stypy_function_name', 'bmp.__and__')
        bmp.__and__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        bmp.__and__.__dict__.__setitem__('stypy_varargs_param_name', None)
        bmp.__and__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bmp.__and__.__dict__.__setitem__('stypy_call_defaults', defaults)
        bmp.__and__.__dict__.__setitem__('stypy_call_varargs', varargs)
        bmp.__and__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bmp.__and__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bmp.__and__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__and__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__and__(...)' code ##################

        
        # Call to bmp(...): (line 22)
        # Processing the call arguments (line 22)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 22)
        # Processing the call arguments (line 22)
        int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 61), 'int')
        # Processing the call keyword arguments (line 22)
        kwargs_56 = {}
        # Getting the type of 'xrange' (line 22)
        xrange_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 54), 'xrange', False)
        # Calling xrange(args, kwargs) (line 22)
        xrange_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 22, 54), xrange_54, *[int_55], **kwargs_56)
        
        comprehension_58 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), xrange_call_result_57)
        # Assigning a type to the variable 'i' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'i', comprehension_58)
        
        # Evaluating a boolean operation
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 22)
        i_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'i', False)
        # Getting the type of 'self' (line 22)
        self_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'self', False)
        # Obtaining the member 'v' of a type (line 22)
        v_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 20), self_44, 'v')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___46 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 20), v_45, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 22, 20), getitem___46, i_43)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 22)
        i_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 42), 'i', False)
        # Getting the type of 'other' (line 22)
        other_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'other', False)
        # Obtaining the member 'v' of a type (line 22)
        v_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 34), other_49, 'v')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 34), v_50, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 22, 34), getitem___51, i_48)
        
        # Applying the binary operator 'and' (line 22)
        result_and_keyword_53 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 20), 'and', subscript_call_result_47, subscript_call_result_52)
        
        list_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), list_59, result_and_keyword_53)
        # Processing the call keyword arguments (line 22)
        kwargs_60 = {}
        # Getting the type of 'bmp' (line 22)
        bmp_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'bmp', False)
        # Calling bmp(args, kwargs) (line 22)
        bmp_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), bmp_42, *[list_59], **kwargs_60)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', bmp_call_result_61)
        
        # ################# End of '__and__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__and__' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__and__'
        return stypy_return_type_62


    @norecursion
    def cnt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cnt'
        module_type_store = module_type_store.open_function_context('cnt', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bmp.cnt.__dict__.__setitem__('stypy_localization', localization)
        bmp.cnt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bmp.cnt.__dict__.__setitem__('stypy_type_store', module_type_store)
        bmp.cnt.__dict__.__setitem__('stypy_function_name', 'bmp.cnt')
        bmp.cnt.__dict__.__setitem__('stypy_param_names_list', [])
        bmp.cnt.__dict__.__setitem__('stypy_varargs_param_name', None)
        bmp.cnt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bmp.cnt.__dict__.__setitem__('stypy_call_defaults', defaults)
        bmp.cnt.__dict__.__setitem__('stypy_call_varargs', varargs)
        bmp.cnt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bmp.cnt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bmp.cnt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cnt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cnt(...)' code ##################

        
        # Call to len(...): (line 25)
        # Processing the call arguments (line 25)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 25)
        self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'self', False)
        # Obtaining the member 'v' of a type (line 25)
        v_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 31), self_66, 'v')
        comprehension_68 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), v_67)
        # Assigning a type to the variable 'i' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'i', comprehension_68)
        # Getting the type of 'i' (line 25)
        i_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 41), 'i', False)
        # Getting the type of 'i' (line 25)
        i_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'i', False)
        list_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), list_69, i_64)
        # Processing the call keyword arguments (line 25)
        kwargs_70 = {}
        # Getting the type of 'len' (line 25)
        len_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'len', False)
        # Calling len(args, kwargs) (line 25)
        len_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), len_63, *[list_69], **kwargs_70)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', len_call_result_71)
        
        # ################# End of 'cnt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cnt' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cnt'
        return stypy_return_type_72


# Assigning a type to the variable 'bmp' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'bmp', bmp)
# Declaration of the 'boardRep' class

class boardRep(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boardRep.__init__', ['board'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['board'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 30):
        
        # Assigning a Call to a Attribute (line 30):
        
        # Call to list(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'board' (line 30)
        board_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'board', False)
        # Obtaining the member 'final' of a type (line 30)
        final_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 29), board_74, 'final')
        # Processing the call keyword arguments (line 30)
        kwargs_76 = {}
        # Getting the type of 'list' (line 30)
        list_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'list', False)
        # Calling list(args, kwargs) (line 30)
        list_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), list_73, *[final_75], **kwargs_76)
        
        # Getting the type of 'self' (line 30)
        self_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member '__fields' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_78, '__fields', list_call_result_77)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fields(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fields'
        module_type_store = module_type_store.open_function_context('fields', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boardRep.fields.__dict__.__setitem__('stypy_localization', localization)
        boardRep.fields.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boardRep.fields.__dict__.__setitem__('stypy_type_store', module_type_store)
        boardRep.fields.__dict__.__setitem__('stypy_function_name', 'boardRep.fields')
        boardRep.fields.__dict__.__setitem__('stypy_param_names_list', [])
        boardRep.fields.__dict__.__setitem__('stypy_varargs_param_name', None)
        boardRep.fields.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boardRep.fields.__dict__.__setitem__('stypy_call_defaults', defaults)
        boardRep.fields.__dict__.__setitem__('stypy_call_varargs', varargs)
        boardRep.fields.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boardRep.fields.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boardRep.fields', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fields', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fields(...)' code ##################

        # Getting the type of 'self' (line 33)
        self_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'self')
        # Obtaining the member '__fields' of a type (line 33)
        fields_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), self_79, '__fields')
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', fields_80)
        
        # ################# End of 'fields(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fields' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_81)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fields'
        return stypy_return_type_81


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'boardRep.stypy__eq__')
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boardRep.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boardRep.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Getting the type of 'self' (line 36)
        self_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'self')
        # Obtaining the member '__fields' of a type (line 36)
        fields_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), self_82, '__fields')
        
        # Call to fields(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_86 = {}
        # Getting the type of 'other' (line 36)
        other_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'other', False)
        # Obtaining the member 'fields' of a type (line 36)
        fields_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 32), other_84, 'fields')
        # Calling fields(args, kwargs) (line 36)
        fields_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 36, 32), fields_85, *[], **kwargs_86)
        
        # Applying the binary operator '==' (line 36)
        result_eq_88 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 15), '==', fields_83, fields_call_result_87)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', result_eq_88)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_89)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_89


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boardRep.__ne__.__dict__.__setitem__('stypy_localization', localization)
        boardRep.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boardRep.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        boardRep.__ne__.__dict__.__setitem__('stypy_function_name', 'boardRep.__ne__')
        boardRep.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        boardRep.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        boardRep.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boardRep.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        boardRep.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        boardRep.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boardRep.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boardRep.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        # Getting the type of 'self' (line 39)
        self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'self')
        # Obtaining the member '__fields' of a type (line 39)
        fields_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), self_90, '__fields')
        
        # Call to fields(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_94 = {}
        # Getting the type of 'other' (line 39)
        other_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 32), 'other', False)
        # Obtaining the member 'fields' of a type (line 39)
        fields_93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 32), other_92, 'fields')
        # Calling fields(args, kwargs) (line 39)
        fields_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 39, 32), fields_93, *[], **kwargs_94)
        
        # Applying the binary operator '!=' (line 39)
        result_ne_96 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '!=', fields_91, fields_call_result_95)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', result_ne_96)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_97


    @norecursion
    def stypy__hash__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hash__'
        module_type_store = module_type_store.open_function_context('__hash__', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_localization', localization)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_type_store', module_type_store)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_function_name', 'boardRep.stypy__hash__')
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_param_names_list', [])
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_varargs_param_name', None)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_call_defaults', defaults)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_call_varargs', varargs)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boardRep.stypy__hash__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boardRep.stypy__hash__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hash__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hash__(...)' code ##################

        
        # Assigning a Str to a Name (line 42):
        
        # Assigning a Str to a Name (line 42):
        str_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'str', '')
        # Assigning a type to the variable 'rep' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'rep', str_98)
        
        
        # Call to xrange(...): (line 43)
        # Processing the call arguments (line 43)
        int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'int')
        # Processing the call keyword arguments (line 43)
        kwargs_101 = {}
        # Getting the type of 'xrange' (line 43)
        xrange_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 43)
        xrange_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), xrange_99, *[int_100], **kwargs_101)
        
        # Assigning a type to the variable 'xrange_call_result_102' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'xrange_call_result_102', xrange_call_result_102)
        # Testing if the for loop is going to be iterated (line 43)
        # Testing the type of a for loop iterable (line 43)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 8), xrange_call_result_102)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 43, 8), xrange_call_result_102):
            # Getting the type of the for loop variable (line 43)
            for_loop_var_103 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 8), xrange_call_result_102)
            # Assigning a type to the variable 'i' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'i', for_loop_var_103)
            # SSA begins for a for statement (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to xrange(...): (line 44)
            # Processing the call arguments (line 44)
            int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'int')
            # Processing the call keyword arguments (line 44)
            kwargs_106 = {}
            # Getting the type of 'xrange' (line 44)
            xrange_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 44)
            xrange_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 44, 21), xrange_104, *[int_105], **kwargs_106)
            
            # Assigning a type to the variable 'xrange_call_result_107' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'xrange_call_result_107', xrange_call_result_107)
            # Testing if the for loop is going to be iterated (line 44)
            # Testing the type of a for loop iterable (line 44)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 12), xrange_call_result_107)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 44, 12), xrange_call_result_107):
                # Getting the type of the for loop variable (line 44)
                for_loop_var_108 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 12), xrange_call_result_107)
                # Assigning a type to the variable 'j' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'j', for_loop_var_108)
                # SSA begins for a for statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'rep' (line 45)
                rep_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'rep')
                
                # Call to str(...): (line 45)
                # Processing the call arguments (line 45)
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 45)
                j_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'j', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 45)
                i_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 41), 'i', False)
                # Getting the type of 'self' (line 45)
                self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'self', False)
                # Obtaining the member '__fields' of a type (line 45)
                fields_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), self_113, '__fields')
                # Obtaining the member '__getitem__' of a type (line 45)
                getitem___115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), fields_114, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 45)
                subscript_call_result_116 = invoke(stypy.reporting.localization.Localization(__file__, 45, 27), getitem___115, i_112)
                
                # Obtaining the member '__getitem__' of a type (line 45)
                getitem___117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), subscript_call_result_116, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 45)
                subscript_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 45, 27), getitem___117, j_111)
                
                # Processing the call keyword arguments (line 45)
                kwargs_119 = {}
                # Getting the type of 'str' (line 45)
                str_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'str', False)
                # Calling str(args, kwargs) (line 45)
                str_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 45, 23), str_110, *[subscript_call_result_118], **kwargs_119)
                
                # Applying the binary operator '+=' (line 45)
                result_iadd_121 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 16), '+=', rep_109, str_call_result_120)
                # Assigning a type to the variable 'rep' (line 45)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'rep', result_iadd_121)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to hash(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'rep' (line 46)
        rep_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'rep', False)
        # Processing the call keyword arguments (line 46)
        kwargs_124 = {}
        # Getting the type of 'hash' (line 46)
        hash_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'hash', False)
        # Calling hash(args, kwargs) (line 46)
        hash_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), hash_122, *[rep_123], **kwargs_124)
        
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', hash_call_result_125)
        
        # ################# End of '__hash__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hash__' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hash__'
        return stypy_return_type_126


# Assigning a type to the variable 'boardRep' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'boardRep', boardRep)
# Declaration of the 'board' class

class board(object, ):
    
    # Assigning a Name to a Name (line 50):
    
    # Assigning a Name to a Name (line 51):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a ListComp to a Attribute (line 55):
        
        # Assigning a ListComp to a Attribute (line 55):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 55)
        # Processing the call arguments (line 55)
        int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 46), 'int')
        # Processing the call keyword arguments (line 55)
        kwargs_133 = {}
        # Getting the type of 'xrange' (line 55)
        xrange_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 39), 'xrange', False)
        # Calling xrange(args, kwargs) (line 55)
        xrange_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 55, 39), xrange_131, *[int_132], **kwargs_133)
        
        comprehension_135 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 22), xrange_call_result_134)
        # Assigning a type to the variable 'i' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'i', comprehension_135)
        int_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 26), list_128, int_129)
        
        # Applying the binary operator '*' (line 55)
        result_mul_130 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 22), '*', int_127, list_128)
        
        list_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 22), list_136, result_mul_130)
        # Getting the type of 'self' (line 55)
        self_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'final' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_137, 'final', list_136)
        
        # Assigning a BinOp to a Attribute (line 56):
        
        # Assigning a BinOp to a Attribute (line 56):
        int_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        
        # Call to bmp(...): (line 56)
        # Processing the call keyword arguments (line 56)
        kwargs_141 = {}
        # Getting the type of 'bmp' (line 56)
        bmp_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'bmp', False)
        # Calling bmp(args, kwargs) (line 56)
        bmp_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 56, 25), bmp_140, *[], **kwargs_141)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 24), list_139, bmp_call_result_142)
        
        # Applying the binary operator '*' (line 56)
        result_mul_143 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 20), '*', int_138, list_139)
        
        # Getting the type of 'self' (line 56)
        self_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'rows' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_144, 'rows', result_mul_143)
        
        # Assigning a BinOp to a Attribute (line 57):
        
        # Assigning a BinOp to a Attribute (line 57):
        int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 20), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        
        # Call to bmp(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_148 = {}
        # Getting the type of 'bmp' (line 57)
        bmp_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'bmp', False)
        # Calling bmp(args, kwargs) (line 57)
        bmp_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), bmp_147, *[], **kwargs_148)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 24), list_146, bmp_call_result_149)
        
        # Applying the binary operator '*' (line 57)
        result_mul_150 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 20), '*', int_145, list_146)
        
        # Getting the type of 'self' (line 57)
        self_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'cols' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_151, 'cols', result_mul_150)
        
        # Assigning a ListComp to a Attribute (line 58):
        
        # Assigning a ListComp to a Attribute (line 58):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 58)
        # Processing the call arguments (line 58)
        int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 49), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_160 = {}
        # Getting the type of 'xrange' (line 58)
        xrange_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 42), 'xrange', False)
        # Calling xrange(args, kwargs) (line 58)
        xrange_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 58, 42), xrange_158, *[int_159], **kwargs_160)
        
        comprehension_162 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), xrange_call_result_161)
        # Assigning a type to the variable 'i' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'i', comprehension_162)
        int_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        
        # Call to bmp(...): (line 58)
        # Processing the call keyword arguments (line 58)
        kwargs_155 = {}
        # Getting the type of 'bmp' (line 58)
        bmp_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'bmp', False)
        # Calling bmp(args, kwargs) (line 58)
        bmp_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 58, 26), bmp_154, *[], **kwargs_155)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 25), list_153, bmp_call_result_156)
        
        # Applying the binary operator '*' (line 58)
        result_mul_157 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 21), '*', int_152, list_153)
        
        list_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_163, result_mul_157)
        # Getting the type of 'self' (line 58)
        self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'cels' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_164, 'cels', list_163)
        
        # Assigning a Num to a Attribute (line 61):
        
        # Assigning a Num to a Attribute (line 61):
        int_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'int')
        # Getting the type of 'self' (line 61)
        self_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member '__turns' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_166, '__turns', int_165)
        
        # Assigning a Num to a Attribute (line 62):
        
        # Assigning a Num to a Attribute (line 62):
        int_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'int')
        # Getting the type of 'self' (line 62)
        self_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member '__backtracks' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_168, '__backtracks', int_167)
        
        # Assigning a Num to a Attribute (line 63):
        
        # Assigning a Num to a Attribute (line 63):
        int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
        # Getting the type of 'self' (line 63)
        self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member '__starttime' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_170, '__starttime', int_169)
        
        # Assigning a Num to a Attribute (line 64):
        
        # Assigning a Num to a Attribute (line 64):
        int_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'int')
        # Getting the type of 'self' (line 64)
        self_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member '__endtime' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_172, '__endtime', int_171)
        
        # Assigning a Num to a Attribute (line 65):
        
        # Assigning a Num to a Attribute (line 65):
        int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'int')
        # Getting the type of 'self' (line 65)
        self_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member '__status' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_174, '__status', int_173)
        
        # Assigning a Num to a Attribute (line 66):
        
        # Assigning a Num to a Attribute (line 66):
        int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'int')
        # Getting the type of 'self' (line 66)
        self_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member '__maxdepth' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_176, '__maxdepth', int_175)
        
        # Assigning a Num to a Attribute (line 67):
        
        # Assigning a Num to a Attribute (line 67):
        int_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'int')
        # Getting the type of 'self' (line 67)
        self_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member '__openspaces' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_178, '__openspaces', int_177)
        
        # Assigning a Call to a Attribute (line 70):
        
        # Assigning a Call to a Attribute (line 70):
        
        # Call to set(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_180 = {}
        # Getting the type of 'set' (line 70)
        set_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'set', False)
        # Calling set(args, kwargs) (line 70)
        set_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 70, 25), set_179, *[], **kwargs_180)
        
        # Getting the type of 'self' (line 70)
        self_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'solutions' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_182, 'solutions', set_call_result_181)
        
        # Assigning a Call to a Attribute (line 72):
        
        # Assigning a Call to a Attribute (line 72):
        
        # Call to set(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_184 = {}
        # Getting the type of 'set' (line 72)
        set_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'set', False)
        # Calling set(args, kwargs) (line 72)
        set_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), set_183, *[], **kwargs_184)
        
        # Getting the type of 'self' (line 72)
        self_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'examined' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_186, 'examined', set_call_result_185)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fread(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'str', '')
        defaults = [str_187]
        # Create a new context for function 'fread'
        module_type_store = module_type_store.open_function_context('fread', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.fread.__dict__.__setitem__('stypy_localization', localization)
        board.fread.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.fread.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.fread.__dict__.__setitem__('stypy_function_name', 'board.fread')
        board.fread.__dict__.__setitem__('stypy_param_names_list', ['fn'])
        board.fread.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.fread.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.fread.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.fread.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.fread.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.fread.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.fread', ['fn'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fread', localization, ['fn'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fread(...)' code ##################

        
        # Getting the type of 'fn' (line 76)
        fn_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'fn')
        str_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'str', '')
        # Applying the binary operator '==' (line 76)
        result_eq_190 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), '==', fn_188, str_189)
        
        # Testing if the type of an if condition is none (line 76)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 8), result_eq_190):
            pass
        else:
            
            # Testing the type of an if condition (line 76)
            if_condition_191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_eq_190)
            # Assigning a type to the variable 'if_condition_191' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_191', if_condition_191)
            # SSA begins for if statement (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 77):
            
            # Assigning a Call to a Name (line 77):
            
            # Call to raw_input(...): (line 77)
            # Processing the call arguments (line 77)
            str_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'str', 'filename: ')
            # Processing the call keyword arguments (line 77)
            kwargs_194 = {}
            # Getting the type of 'raw_input' (line 77)
            raw_input_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'raw_input', False)
            # Calling raw_input(args, kwargs) (line 77)
            raw_input_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), raw_input_192, *[str_193], **kwargs_194)
            
            # Assigning a type to the variable 'fn' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'fn', raw_input_call_result_195)
            # SSA join for if statement (line 76)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to file(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'fn' (line 78)
        fn_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'fn', False)
        str_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'str', 'r')
        # Processing the call keyword arguments (line 78)
        kwargs_199 = {}
        # Getting the type of 'file' (line 78)
        file_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'file', False)
        # Calling file(args, kwargs) (line 78)
        file_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), file_196, *[fn_197, str_198], **kwargs_199)
        
        # Assigning a type to the variable 'f' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'f', file_call_result_200)
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to readlines(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_203 = {}
        # Getting the type of 'f' (line 79)
        f_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'f', False)
        # Obtaining the member 'readlines' of a type (line 79)
        readlines_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), f_201, 'readlines')
        # Calling readlines(args, kwargs) (line 79)
        readlines_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), readlines_202, *[], **kwargs_203)
        
        # Assigning a type to the variable 'lines' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'lines', readlines_call_result_204)
        
        
        # Call to xrange(...): (line 80)
        # Processing the call arguments (line 80)
        int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 26), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_207 = {}
        # Getting the type of 'xrange' (line 80)
        xrange_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 80)
        xrange_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 80, 19), xrange_205, *[int_206], **kwargs_207)
        
        # Assigning a type to the variable 'xrange_call_result_208' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'xrange_call_result_208', xrange_call_result_208)
        # Testing if the for loop is going to be iterated (line 80)
        # Testing the type of a for loop iterable (line 80)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 8), xrange_call_result_208)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 80, 8), xrange_call_result_208):
            # Getting the type of the for loop variable (line 80)
            for_loop_var_209 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 8), xrange_call_result_208)
            # Assigning a type to the variable 'row' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'row', for_loop_var_209)
            # SSA begins for a for statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to xrange(...): (line 81)
            # Processing the call arguments (line 81)
            int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 32), 'int')
            int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 35), 'int')
            # Processing the call keyword arguments (line 81)
            kwargs_213 = {}
            # Getting the type of 'xrange' (line 81)
            xrange_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'xrange', False)
            # Calling xrange(args, kwargs) (line 81)
            xrange_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 81, 25), xrange_210, *[int_211, int_212], **kwargs_213)
            
            # Assigning a type to the variable 'xrange_call_result_214' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'xrange_call_result_214', xrange_call_result_214)
            # Testing if the for loop is going to be iterated (line 81)
            # Testing the type of a for loop iterable (line 81)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 12), xrange_call_result_214)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 12), xrange_call_result_214):
                # Getting the type of the for loop variable (line 81)
                for_loop_var_215 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 12), xrange_call_result_214)
                # Assigning a type to the variable 'digit' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'digit', for_loop_var_215)
                # SSA begins for a for statement (line 81)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # SSA begins for try-except statement (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Call to setval(...): (line 83)
                # Processing the call arguments (line 83)
                # Getting the type of 'row' (line 83)
                row_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'row', False)
                
                # Call to index(...): (line 83)
                # Processing the call arguments (line 83)
                
                # Call to str(...): (line 83)
                # Processing the call arguments (line 83)
                # Getting the type of 'digit' (line 83)
                digit_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 58), 'digit', False)
                # Processing the call keyword arguments (line 83)
                kwargs_226 = {}
                # Getting the type of 'str' (line 83)
                str_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 54), 'str', False)
                # Calling str(args, kwargs) (line 83)
                str_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 83, 54), str_224, *[digit_225], **kwargs_226)
                
                # Processing the call keyword arguments (line 83)
                kwargs_228 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 83)
                row_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 43), 'row', False)
                # Getting the type of 'lines' (line 83)
                lines_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 37), 'lines', False)
                # Obtaining the member '__getitem__' of a type (line 83)
                getitem___221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 37), lines_220, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 83)
                subscript_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 83, 37), getitem___221, row_219)
                
                # Obtaining the member 'index' of a type (line 83)
                index_223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 37), subscript_call_result_222, 'index')
                # Calling index(args, kwargs) (line 83)
                index_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 83, 37), index_223, *[str_call_result_227], **kwargs_228)
                
                # Getting the type of 'digit' (line 83)
                digit_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 67), 'digit', False)
                # Processing the call keyword arguments (line 83)
                kwargs_231 = {}
                # Getting the type of 'self' (line 83)
                self_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'self', False)
                # Obtaining the member 'setval' of a type (line 83)
                setval_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), self_216, 'setval')
                # Calling setval(args, kwargs) (line 83)
                setval_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 83, 20), setval_217, *[row_218, index_call_result_229, digit_230], **kwargs_231)
                
                # SSA branch for the except part of a try statement (line 82)
                # SSA branch for the except 'ValueError' branch of a try statement (line 82)
                module_type_store.open_ssa_branch('except')
                pass
                # SSA join for try-except statement (line 82)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to close(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_235 = {}
        # Getting the type of 'f' (line 86)
        f_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 86)
        close_234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), f_233, 'close')
        # Calling close(args, kwargs) (line 86)
        close_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), close_234, *[], **kwargs_235)
        
        
        # ################# End of 'fread(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fread' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fread'
        return stypy_return_type_237


    @norecursion
    def setval(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setval'
        module_type_store = module_type_store.open_function_context('setval', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.setval.__dict__.__setitem__('stypy_localization', localization)
        board.setval.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.setval.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.setval.__dict__.__setitem__('stypy_function_name', 'board.setval')
        board.setval.__dict__.__setitem__('stypy_param_names_list', ['row', 'col', 'val'])
        board.setval.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.setval.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.setval.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.setval.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.setval.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.setval.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.setval', ['row', 'col', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setval', localization, ['row', 'col', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setval(...)' code ##################

        
        # Assigning a Name to a Subscript (line 90):
        
        # Assigning a Name to a Subscript (line 90):
        # Getting the type of 'val' (line 90)
        val_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'val')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 90)
        row_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'row')
        # Getting the type of 'self' (line 90)
        self_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Obtaining the member 'final' of a type (line 90)
        final_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_240, 'final')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), final_241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), getitem___242, row_239)
        
        # Getting the type of 'col' (line 90)
        col_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'col')
        # Storing an element on a container (line 90)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), subscript_call_result_243, (col_244, val_238))
        
        # Getting the type of 'self' (line 91)
        self_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Obtaining the member '__openspaces' of a type (line 91)
        openspaces_246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_245, '__openspaces')
        int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'int')
        # Applying the binary operator '-=' (line 91)
        result_isub_248 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 8), '-=', openspaces_246, int_247)
        # Getting the type of 'self' (line 91)
        self_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member '__openspaces' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_249, '__openspaces', result_isub_248)
        
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to bmp(...): (line 94)
        # Processing the call keyword arguments (line 94)
        # Getting the type of 'val' (line 94)
        val_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'val', False)
        int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 27), 'int')
        # Applying the binary operator '-' (line 94)
        result_sub_253 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 21), '-', val_251, int_252)
        
        keyword_254 = result_sub_253
        kwargs_255 = {'n': keyword_254}
        # Getting the type of 'bmp' (line 94)
        bmp_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'bmp', False)
        # Calling bmp(args, kwargs) (line 94)
        bmp_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), bmp_250, *[], **kwargs_255)
        
        # Assigning a type to the variable 'mask' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'mask', bmp_call_result_256)
        
        # Assigning a BinOp to a Subscript (line 96):
        
        # Assigning a BinOp to a Subscript (line 96):
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 96)
        row_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 35), 'row')
        # Getting the type of 'self' (line 96)
        self_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'self')
        # Obtaining the member 'rows' of a type (line 96)
        rows_259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), self_258, 'rows')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), rows_259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 96, 25), getitem___260, row_257)
        
        # Getting the type of 'mask' (line 96)
        mask_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'mask')
        # Applying the binary operator '&' (line 96)
        result_and__263 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 25), '&', subscript_call_result_261, mask_262)
        
        # Getting the type of 'self' (line 96)
        self_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Obtaining the member 'rows' of a type (line 96)
        rows_265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_264, 'rows')
        # Getting the type of 'row' (line 96)
        row_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'row')
        # Storing an element on a container (line 96)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), rows_265, (row_266, result_and__263))
        
        # Assigning a BinOp to a Subscript (line 97):
        
        # Assigning a BinOp to a Subscript (line 97):
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 97)
        col_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 35), 'col')
        # Getting the type of 'self' (line 97)
        self_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'self')
        # Obtaining the member 'cols' of a type (line 97)
        cols_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 25), self_268, 'cols')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 25), cols_269, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 97, 25), getitem___270, col_267)
        
        # Getting the type of 'mask' (line 97)
        mask_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'mask')
        # Applying the binary operator '&' (line 97)
        result_and__273 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 25), '&', subscript_call_result_271, mask_272)
        
        # Getting the type of 'self' (line 97)
        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Obtaining the member 'cols' of a type (line 97)
        cols_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_274, 'cols')
        # Getting the type of 'col' (line 97)
        col_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'col')
        # Storing an element on a container (line 97)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 8), cols_275, (col_276, result_and__273))
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to cell(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'row' (line 99)
        row_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'row', False)
        # Processing the call keyword arguments (line 99)
        kwargs_280 = {}
        # Getting the type of 'self' (line 99)
        self_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'self', False)
        # Obtaining the member 'cell' of a type (line 99)
        cell_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), self_277, 'cell')
        # Calling cell(args, kwargs) (line 99)
        cell_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), cell_278, *[row_279], **kwargs_280)
        
        # Assigning a type to the variable 'cr' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'cr', cell_call_result_281)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to cell(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'col' (line 100)
        col_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'col', False)
        # Processing the call keyword arguments (line 100)
        kwargs_285 = {}
        # Getting the type of 'self' (line 100)
        self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'self', False)
        # Obtaining the member 'cell' of a type (line 100)
        cell_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), self_282, 'cell')
        # Calling cell(args, kwargs) (line 100)
        cell_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), cell_283, *[col_284], **kwargs_285)
        
        # Assigning a type to the variable 'cc' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'cc', cell_call_result_286)
        
        # Assigning a BinOp to a Subscript (line 101):
        
        # Assigning a BinOp to a Subscript (line 101):
        
        # Obtaining the type of the subscript
        # Getting the type of 'cc' (line 101)
        cc_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'cc')
        
        # Obtaining the type of the subscript
        # Getting the type of 'cr' (line 101)
        cr_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'cr')
        # Getting the type of 'self' (line 101)
        self_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'self')
        # Obtaining the member 'cels' of a type (line 101)
        cels_290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), self_289, 'cels')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), cels_290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), getitem___291, cr_288)
        
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), subscript_call_result_292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), getitem___293, cc_287)
        
        # Getting the type of 'mask' (line 101)
        mask_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 48), 'mask')
        # Applying the binary operator '&' (line 101)
        result_and__296 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 28), '&', subscript_call_result_294, mask_295)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'cr' (line 101)
        cr_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'cr')
        # Getting the type of 'self' (line 101)
        self_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Obtaining the member 'cels' of a type (line 101)
        cels_299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_298, 'cels')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), cels_299, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___300, cr_297)
        
        # Getting the type of 'cc' (line 101)
        cc_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'cc')
        # Storing an element on a container (line 101)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 8), subscript_call_result_301, (cc_302, result_and__296))
        
        # ################# End of 'setval(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setval' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setval'
        return stypy_return_type_303


    @norecursion
    def cell(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cell'
        module_type_store = module_type_store.open_function_context('cell', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.cell.__dict__.__setitem__('stypy_localization', localization)
        board.cell.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.cell.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.cell.__dict__.__setitem__('stypy_function_name', 'board.cell')
        board.cell.__dict__.__setitem__('stypy_param_names_list', ['num'])
        board.cell.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.cell.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.cell.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.cell.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.cell.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.cell.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.cell', ['num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cell', localization, ['num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cell(...)' code ##################

        
        # Call to int(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to ceil(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'num' (line 104)
        num_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'num', False)
        int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'int')
        # Applying the binary operator '+' (line 104)
        result_add_308 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 25), '+', num_306, int_307)
        
        float_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'float')
        # Applying the binary operator 'div' (line 104)
        result_div_310 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 24), 'div', result_add_308, float_309)
        
        # Processing the call keyword arguments (line 104)
        kwargs_311 = {}
        # Getting the type of 'ceil' (line 104)
        ceil_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'ceil', False)
        # Calling ceil(args, kwargs) (line 104)
        ceil_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 104, 19), ceil_305, *[result_div_310], **kwargs_311)
        
        # Processing the call keyword arguments (line 104)
        kwargs_313 = {}
        # Getting the type of 'int' (line 104)
        int_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'int', False)
        # Calling int(args, kwargs) (line 104)
        int_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), int_304, *[ceil_call_result_312], **kwargs_313)
        
        int_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'int')
        # Applying the binary operator '-' (line 104)
        result_sub_316 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '-', int_call_result_314, int_315)
        
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', result_sub_316)
        
        # ################# End of 'cell(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cell' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_317)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cell'
        return stypy_return_type_317


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        board.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.stypy__str__.__dict__.__setitem__('stypy_function_name', 'board.stypy__str__')
        board.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        board.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 107):
        
        # Assigning a Str to a Name (line 107):
        str_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 14), 'str', '')
        # Assigning a type to the variable 'ret' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'ret', str_318)
        
        
        # Call to xrange(...): (line 108)
        # Processing the call arguments (line 108)
        int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'int')
        # Processing the call keyword arguments (line 108)
        kwargs_321 = {}
        # Getting the type of 'xrange' (line 108)
        xrange_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 108)
        xrange_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), xrange_319, *[int_320], **kwargs_321)
        
        # Assigning a type to the variable 'xrange_call_result_322' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'xrange_call_result_322', xrange_call_result_322)
        # Testing if the for loop is going to be iterated (line 108)
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_322)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_322):
            # Getting the type of the for loop variable (line 108)
            for_loop_var_323 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_322)
            # Assigning a type to the variable 'row' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'row', for_loop_var_323)
            # SSA begins for a for statement (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'row' (line 109)
            row_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'row')
            int_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'int')
            # Applying the binary operator '==' (line 109)
            result_eq_326 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 15), '==', row_324, int_325)
            
            
            # Getting the type of 'row' (line 109)
            row_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'row')
            int_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 34), 'int')
            # Applying the binary operator '==' (line 109)
            result_eq_329 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 27), '==', row_327, int_328)
            
            # Applying the binary operator 'or' (line 109)
            result_or_keyword_330 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 15), 'or', result_eq_326, result_eq_329)
            
            # Testing if the type of an if condition is none (line 109)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 12), result_or_keyword_330):
                pass
            else:
                
                # Testing the type of an if condition (line 109)
                if_condition_331 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 12), result_or_keyword_330)
                # Assigning a type to the variable 'if_condition_331' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'if_condition_331', if_condition_331)
                # SSA begins for if statement (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'ret' (line 109)
                ret_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'ret')
                
                # Obtaining the type of the subscript
                int_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 71), 'int')
                slice_334 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 45), None, int_333, None)
                int_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'int')
                str_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 51), 'str', '---')
                # Applying the binary operator '*' (line 109)
                result_mul_337 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 47), '*', int_335, str_336)
                
                str_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 60), 'str', '+')
                # Applying the binary operator '+' (line 109)
                result_add_339 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 46), '+', result_mul_337, str_338)
                
                int_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 67), 'int')
                # Applying the binary operator '*' (line 109)
                result_mul_341 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 45), '*', result_add_339, int_340)
                
                # Obtaining the member '__getitem__' of a type (line 109)
                getitem___342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 45), result_mul_341, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 109)
                subscript_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 109, 45), getitem___342, slice_334)
                
                str_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 77), 'str', '\n')
                # Applying the binary operator '+' (line 109)
                result_add_345 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 44), '+', subscript_call_result_343, str_344)
                
                # Applying the binary operator '+=' (line 109)
                result_iadd_346 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 37), '+=', ret_332, result_add_345)
                # Assigning a type to the variable 'ret' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'ret', result_iadd_346)
                
                # SSA join for if statement (line 109)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to xrange(...): (line 110)
            # Processing the call arguments (line 110)
            int_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
            # Processing the call keyword arguments (line 110)
            kwargs_349 = {}
            # Getting the type of 'xrange' (line 110)
            xrange_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'xrange', False)
            # Calling xrange(args, kwargs) (line 110)
            xrange_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 110, 23), xrange_347, *[int_348], **kwargs_349)
            
            # Assigning a type to the variable 'xrange_call_result_350' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'xrange_call_result_350', xrange_call_result_350)
            # Testing if the for loop is going to be iterated (line 110)
            # Testing the type of a for loop iterable (line 110)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 12), xrange_call_result_350)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 12), xrange_call_result_350):
                # Getting the type of the for loop variable (line 110)
                for_loop_var_351 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 12), xrange_call_result_350)
                # Assigning a type to the variable 'col' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'col', for_loop_var_351)
                # SSA begins for a for statement (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'col' (line 111)
                col_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'col')
                int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 26), 'int')
                # Applying the binary operator '==' (line 111)
                result_eq_354 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 19), '==', col_352, int_353)
                
                
                # Getting the type of 'col' (line 111)
                col_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 31), 'col')
                int_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'int')
                # Applying the binary operator '==' (line 111)
                result_eq_357 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 31), '==', col_355, int_356)
                
                # Applying the binary operator 'or' (line 111)
                result_or_keyword_358 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 19), 'or', result_eq_354, result_eq_357)
                
                # Testing if the type of an if condition is none (line 111)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 111, 16), result_or_keyword_358):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 111)
                    if_condition_359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 16), result_or_keyword_358)
                    # Assigning a type to the variable 'if_condition_359' (line 111)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'if_condition_359', if_condition_359)
                    # SSA begins for if statement (line 111)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'ret' (line 111)
                    ret_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'ret')
                    str_361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 48), 'str', '|')
                    # Applying the binary operator '+=' (line 111)
                    result_iadd_362 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 41), '+=', ret_360, str_361)
                    # Assigning a type to the variable 'ret' (line 111)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'ret', result_iadd_362)
                    
                    # SSA join for if statement (line 111)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 112)
                col_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 112)
                row_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'row')
                # Getting the type of 'self' (line 112)
                self_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'self')
                # Obtaining the member 'final' of a type (line 112)
                final_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), self_365, 'final')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), final_366, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), getitem___367, row_364)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), subscript_call_result_368, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), getitem___369, col_363)
                
                # Testing if the type of an if condition is none (line 112)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 16), subscript_call_result_370):
                    
                    # Assigning a Str to a Name (line 115):
                    
                    # Assigning a Str to a Name (line 115):
                    str_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 24), 'str', ' ')
                    # Assigning a type to the variable 'c' (line 115)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'c', str_383)
                else:
                    
                    # Testing the type of an if condition (line 112)
                    if_condition_371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), subscript_call_result_370)
                    # Assigning a type to the variable 'if_condition_371' (line 112)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_371', if_condition_371)
                    # SSA begins for if statement (line 112)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 113):
                    
                    # Assigning a Call to a Name (line 113):
                    
                    # Call to str(...): (line 113)
                    # Processing the call arguments (line 113)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 113)
                    col_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 113)
                    row_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 39), 'row', False)
                    # Getting the type of 'self' (line 113)
                    self_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'self', False)
                    # Obtaining the member 'final' of a type (line 113)
                    final_376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), self_375, 'final')
                    # Obtaining the member '__getitem__' of a type (line 113)
                    getitem___377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), final_376, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                    subscript_call_result_378 = invoke(stypy.reporting.localization.Localization(__file__, 113, 28), getitem___377, row_374)
                    
                    # Obtaining the member '__getitem__' of a type (line 113)
                    getitem___379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), subscript_call_result_378, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                    subscript_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 113, 28), getitem___379, col_373)
                    
                    # Processing the call keyword arguments (line 113)
                    kwargs_381 = {}
                    # Getting the type of 'str' (line 113)
                    str_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'str', False)
                    # Calling str(args, kwargs) (line 113)
                    str_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), str_372, *[subscript_call_result_380], **kwargs_381)
                    
                    # Assigning a type to the variable 'c' (line 113)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'c', str_call_result_382)
                    # SSA branch for the else part of an if statement (line 112)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Str to a Name (line 115):
                    
                    # Assigning a Str to a Name (line 115):
                    str_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 24), 'str', ' ')
                    # Assigning a type to the variable 'c' (line 115)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'c', str_383)
                    # SSA join for if statement (line 112)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'ret' (line 116)
                ret_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'ret')
                str_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'str', ' ')
                # Getting the type of 'c' (line 116)
                c_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'c')
                # Applying the binary operator '+' (line 116)
                result_add_387 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 23), '+', str_385, c_386)
                
                str_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 33), 'str', ' ')
                # Applying the binary operator '+' (line 116)
                result_add_389 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 31), '+', result_add_387, str_388)
                
                # Applying the binary operator '+=' (line 116)
                result_iadd_390 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 16), '+=', ret_384, result_add_389)
                # Assigning a type to the variable 'ret' (line 116)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'ret', result_iadd_390)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'ret' (line 117)
            ret_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'ret')
            str_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 19), 'str', '\n')
            # Applying the binary operator '+=' (line 117)
            result_iadd_393 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '+=', ret_391, str_392)
            # Assigning a type to the variable 'ret' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'ret', result_iadd_393)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'ret' (line 118)
        ret_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', ret_394)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_395


    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 120)
        True_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'True')
        # Getting the type of 'False' (line 120)
        False_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'False')
        defaults = [True_396, False_397]
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.solve.__dict__.__setitem__('stypy_localization', localization)
        board.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.solve.__dict__.__setitem__('stypy_function_name', 'board.solve')
        board.solve.__dict__.__setitem__('stypy_param_names_list', ['notify', 'completeSearch'])
        board.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.solve.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.solve', ['notify', 'completeSearch'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, ['notify', 'completeSearch'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        
        # Getting the type of 'self' (line 121)
        self_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'self')
        # Obtaining the member '__status' of a type (line 121)
        status_399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), self_398, '__status')
        int_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'int')
        # Applying the binary operator '==' (line 121)
        result_eq_401 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 11), '==', status_399, int_400)
        
        # Testing if the type of an if condition is none (line 121)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 8), result_eq_401):
            pass
        else:
            
            # Testing the type of an if condition (line 121)
            if_condition_402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 8), result_eq_401)
            # Assigning a type to the variable 'if_condition_402' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'if_condition_402', if_condition_402)
            # SSA begins for if statement (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Attribute (line 122):
            
            # Assigning a Num to a Attribute (line 122):
            int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 28), 'int')
            # Getting the type of 'self' (line 122)
            self_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'self')
            # Setting the type of the member '__status' of a type (line 122)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), self_404, '__status', int_403)
            
            # Assigning a Call to a Attribute (line 123):
            
            # Assigning a Call to a Attribute (line 123):
            
            # Call to time(...): (line 123)
            # Processing the call keyword arguments (line 123)
            kwargs_406 = {}
            # Getting the type of 'time' (line 123)
            time_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'time', False)
            # Calling time(args, kwargs) (line 123)
            time_call_result_407 = invoke(stypy.reporting.localization.Localization(__file__, 123, 31), time_405, *[], **kwargs_406)
            
            # Getting the type of 'self' (line 123)
            self_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'self')
            # Setting the type of the member '__starttime' of a type (line 123)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), self_408, '__starttime', time_call_result_407)
            
            # Assigning a Name to a Attribute (line 124):
            
            # Assigning a Name to a Attribute (line 124):
            # Getting the type of 'notify' (line 124)
            notify_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 39), 'notify')
            # Getting the type of 'board' (line 124)
            board_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'board')
            # Setting the type of the member 'notifyOnCompletion' of a type (line 124)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), board_410, 'notifyOnCompletion', notify_409)
            
            # Assigning a Name to a Attribute (line 125):
            
            # Assigning a Name to a Attribute (line 125):
            # Getting the type of 'completeSearch' (line 125)
            completeSearch_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'completeSearch')
            # Getting the type of 'board' (line 125)
            board_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'board')
            # Setting the type of the member 'completeSearch' of a type (line 125)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), board_412, 'completeSearch', completeSearch_411)
            
            # Call to __solve(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'self' (line 126)
            self_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'self', False)
            int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 31), 'int')
            # Processing the call keyword arguments (line 126)
            kwargs_417 = {}
            # Getting the type of 'self' (line 126)
            self_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'self', False)
            # Obtaining the member '__solve' of a type (line 126)
            solve_414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), self_413, '__solve')
            # Calling __solve(args, kwargs) (line 126)
            solve_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), solve_414, *[self_415, int_416], **kwargs_417)
            
            # SSA join for if statement (line 121)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_419


    @norecursion
    def openspaces(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'openspaces'
        module_type_store = module_type_store.open_function_context('openspaces', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.openspaces.__dict__.__setitem__('stypy_localization', localization)
        board.openspaces.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.openspaces.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.openspaces.__dict__.__setitem__('stypy_function_name', 'board.openspaces')
        board.openspaces.__dict__.__setitem__('stypy_param_names_list', [])
        board.openspaces.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.openspaces.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.openspaces.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.openspaces.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.openspaces.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.openspaces.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.openspaces', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'openspaces', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'openspaces(...)' code ##################

        # Getting the type of 'self' (line 129)
        self_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'self')
        # Obtaining the member '__openspaces' of a type (line 129)
        openspaces_421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 15), self_420, '__openspaces')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', openspaces_421)
        
        # ################# End of 'openspaces(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'openspaces' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'openspaces'
        return stypy_return_type_422


    @norecursion
    def __solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__solve'
        module_type_store = module_type_store.open_function_context('__solve', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.__solve.__dict__.__setitem__('stypy_localization', localization)
        board.__solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.__solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.__solve.__dict__.__setitem__('stypy_function_name', 'board.__solve')
        board.__solve.__dict__.__setitem__('stypy_param_names_list', ['_board', 'depth'])
        board.__solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.__solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.__solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.__solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.__solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.__solve.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.__solve', ['_board', 'depth'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__solve', localization, ['_board', 'depth'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__solve(...)' code ##################

        
        
        # Call to boardRep(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of '_board' (line 132)
        _board_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), '_board', False)
        # Processing the call keyword arguments (line 132)
        kwargs_425 = {}
        # Getting the type of 'boardRep' (line 132)
        boardRep_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'boardRep', False)
        # Calling boardRep(args, kwargs) (line 132)
        boardRep_call_result_426 = invoke(stypy.reporting.localization.Localization(__file__, 132, 11), boardRep_423, *[_board_424], **kwargs_425)
        
        # Getting the type of 'self' (line 132)
        self_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'self')
        # Obtaining the member 'examined' of a type (line 132)
        examined_428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 35), self_427, 'examined')
        # Applying the binary operator 'notin' (line 132)
        result_contains_429 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 11), 'notin', boardRep_call_result_426, examined_428)
        
        # Testing if the type of an if condition is none (line 132)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 132, 8), result_contains_429):
            pass
        else:
            
            # Testing the type of an if condition (line 132)
            if_condition_430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_contains_429)
            # Assigning a type to the variable 'if_condition_430' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_430', if_condition_430)
            # SSA begins for if statement (line 132)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Call to boardRep(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of '_board' (line 133)
            _board_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 39), '_board', False)
            # Processing the call keyword arguments (line 133)
            kwargs_436 = {}
            # Getting the type of 'boardRep' (line 133)
            boardRep_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), 'boardRep', False)
            # Calling boardRep(args, kwargs) (line 133)
            boardRep_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 133, 30), boardRep_434, *[_board_435], **kwargs_436)
            
            # Processing the call keyword arguments (line 133)
            kwargs_438 = {}
            # Getting the type of 'self' (line 133)
            self_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'self', False)
            # Obtaining the member 'examined' of a type (line 133)
            examined_432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), self_431, 'examined')
            # Obtaining the member 'add' of a type (line 133)
            add_433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), examined_432, 'add')
            # Calling add(args, kwargs) (line 133)
            add_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), add_433, *[boardRep_call_result_437], **kwargs_438)
            
            
            
            # Call to openspaces(...): (line 136)
            # Processing the call keyword arguments (line 136)
            kwargs_442 = {}
            # Getting the type of '_board' (line 136)
            _board_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), '_board', False)
            # Obtaining the member 'openspaces' of a type (line 136)
            openspaces_441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 15), _board_440, 'openspaces')
            # Calling openspaces(args, kwargs) (line 136)
            openspaces_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 136, 15), openspaces_441, *[], **kwargs_442)
            
            int_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 38), 'int')
            # Applying the binary operator '<=' (line 136)
            result_le_445 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 15), '<=', openspaces_call_result_443, int_444)
            
            # Testing if the type of an if condition is none (line 136)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 136, 12), result_le_445):
                
                # Getting the type of 'self' (line 146)
                self_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'self')
                # Obtaining the member '__turns' of a type (line 146)
                turns_473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), self_472, '__turns')
                int_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 32), 'int')
                # Applying the binary operator '+=' (line 146)
                result_iadd_475 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 16), '+=', turns_473, int_474)
                # Getting the type of 'self' (line 146)
                self_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'self')
                # Setting the type of the member '__turns' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), self_476, '__turns', result_iadd_475)
                
                
                # Getting the type of 'depth' (line 147)
                depth_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'depth')
                # Getting the type of 'self' (line 147)
                self_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'self')
                # Obtaining the member '__maxdepth' of a type (line 147)
                maxdepth_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 27), self_478, '__maxdepth')
                # Applying the binary operator '>' (line 147)
                result_gt_480 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), '>', depth_477, maxdepth_479)
                
                # Testing if the type of an if condition is none (line 147)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 147, 16), result_gt_480):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 147)
                    if_condition_481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 16), result_gt_480)
                    # Assigning a type to the variable 'if_condition_481' (line 147)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'if_condition_481', if_condition_481)
                    # SSA begins for if statement (line 147)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Attribute (line 147):
                    
                    # Assigning a Name to a Attribute (line 147):
                    # Getting the type of 'depth' (line 147)
                    depth_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 62), 'depth')
                    # Getting the type of 'self' (line 147)
                    self_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 44), 'self')
                    # Setting the type of the member '__maxdepth' of a type (line 147)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 44), self_483, '__maxdepth', depth_482)
                    # SSA join for if statement (line 147)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Tuple (line 150):
                
                # Assigning a Call to a Name:
                
                # Call to findmincounts(...): (line 150)
                # Processing the call keyword arguments (line 150)
                kwargs_486 = {}
                # Getting the type of '_board' (line 150)
                _board_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 33), '_board', False)
                # Obtaining the member 'findmincounts' of a type (line 150)
                findmincounts_485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 33), _board_484, 'findmincounts')
                # Calling findmincounts(args, kwargs) (line 150)
                findmincounts_call_result_487 = invoke(stypy.reporting.localization.Localization(__file__, 150, 33), findmincounts_485, *[], **kwargs_486)
                
                # Assigning a type to the variable 'call_assignment_1' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_1', findmincounts_call_result_487)
                
                # Assigning a Call to a Name (line 150):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1' (line 150)
                call_assignment_1_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_1', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_489 = stypy_get_value_from_tuple(call_assignment_1_488, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_2' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_2', stypy_get_value_from_tuple_call_result_489)
                
                # Assigning a Name to a Name (line 150):
                # Getting the type of 'call_assignment_2' (line 150)
                call_assignment_2_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_2')
                # Assigning a type to the variable 'mincnt' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'mincnt', call_assignment_2_490)
                
                # Assigning a Call to a Name (line 150):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1' (line 150)
                call_assignment_1_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_1', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_492 = stypy_get_value_from_tuple(call_assignment_1_491, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_3' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_3', stypy_get_value_from_tuple_call_result_492)
                
                # Assigning a Name to a Name (line 150):
                # Getting the type of 'call_assignment_3' (line 150)
                call_assignment_3_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_3')
                # Assigning a type to the variable 'coords' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'coords', call_assignment_3_493)
                
                # Getting the type of 'mincnt' (line 151)
                mincnt_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'mincnt')
                int_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'int')
                # Applying the binary operator '<=' (line 151)
                result_le_496 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 19), '<=', mincnt_494, int_495)
                
                # Testing if the type of an if condition is none (line 151)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 16), result_le_496):
                    
                    # Getting the type of 'coords' (line 157)
                    coords_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'coords')
                    # Assigning a type to the variable 'coords_511' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'coords_511', coords_511)
                    # Testing if the for loop is going to be iterated (line 157)
                    # Testing the type of a for loop iterable (line 157)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511):
                        # Getting the type of the for loop variable (line 157)
                        for_loop_var_512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)
                        # Assigning a type to the variable 'row' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 0))
                        # Assigning a type to the variable 'col' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 1))
                        # SSA begins for a for statement (line 157)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Assigning a Name to a Name (line 159):
                        
                        # Assigning a Name to a Name (line 159):
                        # Getting the type of 'False' (line 159)
                        False_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'False')
                        # Assigning a type to the variable 'broken' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'broken', False_513)
                        
                        # Calculating list comprehension
                        # Calculating comprehension expression
                        
                        # Call to xrange(...): (line 160)
                        # Processing the call arguments (line 160)
                        int_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 54), 'int')
                        # Processing the call keyword arguments (line 160)
                        kwargs_529 = {}
                        # Getting the type of 'xrange' (line 160)
                        xrange_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 47), 'xrange', False)
                        # Calling xrange(args, kwargs) (line 160)
                        xrange_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 160, 47), xrange_527, *[int_528], **kwargs_529)
                        
                        comprehension_531 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), xrange_call_result_530)
                        # Assigning a type to the variable 'i' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i', comprehension_531)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 160)
                        i_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 89), 'i')
                        
                        # Call to mergemask(...): (line 160)
                        # Processing the call arguments (line 160)
                        # Getting the type of 'row' (line 160)
                        row_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 77), 'row', False)
                        # Getting the type of 'col' (line 160)
                        col_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 82), 'col', False)
                        # Processing the call keyword arguments (line 160)
                        kwargs_520 = {}
                        # Getting the type of '_board' (line 160)
                        _board_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 60), '_board', False)
                        # Obtaining the member 'mergemask' of a type (line 160)
                        mergemask_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), _board_516, 'mergemask')
                        # Calling mergemask(args, kwargs) (line 160)
                        mergemask_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_517, *[row_518, col_519], **kwargs_520)
                        
                        # Obtaining the member 'v' of a type (line 160)
                        v_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_call_result_521, 'v')
                        # Obtaining the member '__getitem__' of a type (line 160)
                        getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), v_522, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
                        subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), getitem___523, i_515)
                        
                        # Getting the type of 'True' (line 160)
                        True_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 95), 'True')
                        # Applying the binary operator '==' (line 160)
                        result_eq_526 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 60), '==', subscript_call_result_524, True_525)
                        
                        # Getting the type of 'i' (line 160)
                        i_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i')
                        list_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'list')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), list_532, i_514)
                        # Assigning a type to the variable 'list_532' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'list_532', list_532)
                        # Testing if the for loop is going to be iterated (line 160)
                        # Testing the type of a for loop iterable (line 160)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 160, 24), list_532):
                            # Getting the type of the for loop variable (line 160)
                            for_loop_var_533 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)
                            # Assigning a type to the variable 'val' (line 160)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'val', for_loop_var_533)
                            # SSA begins for a for statement (line 160)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'board' (line 161)
                            board_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'board')
                            # Obtaining the member 'completeSearch' of a type (line 161)
                            completeSearch_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 35), board_534, 'completeSearch')
                            # Applying the 'not' unary operator (line 161)
                            result_not__536 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'not', completeSearch_535)
                            
                            
                            # Getting the type of 'self' (line 161)
                            self_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 60), 'self')
                            # Obtaining the member '__status' of a type (line 161)
                            status_538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 60), self_537, '__status')
                            int_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 77), 'int')
                            # Applying the binary operator '==' (line 161)
                            result_eq_540 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 60), '==', status_538, int_539)
                            
                            # Applying the binary operator 'and' (line 161)
                            result_and_keyword_541 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'and', result_not__536, result_eq_540)
                            
                            # Testing if the type of an if condition is none (line 161)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 161)
                                if_condition_542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541)
                                # Assigning a type to the variable 'if_condition_542' (line 161)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'if_condition_542', if_condition_542)
                                # SSA begins for if statement (line 161)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Name (line 162):
                                
                                # Assigning a Name to a Name (line 162):
                                # Getting the type of 'True' (line 162)
                                True_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'True')
                                # Assigning a type to the variable 'broken' (line 162)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'broken', True_543)
                                # SSA join for if statement (line 161)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Getting the type of 'val' (line 164)
                            val_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val')
                            int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 35), 'int')
                            # Applying the binary operator '+=' (line 164)
                            result_iadd_546 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 28), '+=', val_544, int_545)
                            # Assigning a type to the variable 'val' (line 164)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val', result_iadd_546)
                            
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Call to clone(...): (line 165)
                            # Processing the call keyword arguments (line 165)
                            kwargs_549 = {}
                            # Getting the type of '_board' (line 165)
                            _board_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), '_board', False)
                            # Obtaining the member 'clone' of a type (line 165)
                            clone_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), _board_547, 'clone')
                            # Calling clone(args, kwargs) (line 165)
                            clone_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), clone_548, *[], **kwargs_549)
                            
                            # Assigning a type to the variable 't' (line 165)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 't', clone_call_result_550)
                            
                            # Call to setval(...): (line 166)
                            # Processing the call arguments (line 166)
                            # Getting the type of 'row' (line 166)
                            row_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'row', False)
                            # Getting the type of 'col' (line 166)
                            col_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 42), 'col', False)
                            # Getting the type of 'val' (line 166)
                            val_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'val', False)
                            # Processing the call keyword arguments (line 166)
                            kwargs_556 = {}
                            # Getting the type of 't' (line 166)
                            t_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 't', False)
                            # Obtaining the member 'setval' of a type (line 166)
                            setval_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), t_551, 'setval')
                            # Calling setval(args, kwargs) (line 166)
                            setval_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 166, 28), setval_552, *[row_553, col_554, val_555], **kwargs_556)
                            
                            
                            # Call to __solve(...): (line 167)
                            # Processing the call arguments (line 167)
                            # Getting the type of 't' (line 167)
                            t_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 't', False)
                            # Getting the type of 'depth' (line 167)
                            depth_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'depth', False)
                            int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 52), 'int')
                            # Applying the binary operator '+' (line 167)
                            result_add_563 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 44), '+', depth_561, int_562)
                            
                            # Processing the call keyword arguments (line 167)
                            kwargs_564 = {}
                            # Getting the type of 'self' (line 167)
                            self_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'self', False)
                            # Obtaining the member '__solve' of a type (line 167)
                            solve_559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 28), self_558, '__solve')
                            # Calling __solve(args, kwargs) (line 167)
                            solve_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 167, 28), solve_559, *[t_560, result_add_563], **kwargs_564)
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # Getting the type of 'broken' (line 170)
                        broken_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'broken')
                        # Testing if the type of an if condition is none (line 170)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 170)
                            if_condition_567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566)
                            # Assigning a type to the variable 'if_condition_567' (line 170)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'if_condition_567', if_condition_567)
                            # SSA begins for if statement (line 170)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # SSA join for if statement (line 170)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                else:
                    
                    # Testing the type of an if condition (line 151)
                    if_condition_497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 16), result_le_496)
                    # Assigning a type to the variable 'if_condition_497' (line 151)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'if_condition_497', if_condition_497)
                    # SSA begins for if statement (line 151)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'self' (line 152)
                    self_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'self')
                    # Obtaining the member '__backtracks' of a type (line 152)
                    backtracks_499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), self_498, '__backtracks')
                    int_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 41), 'int')
                    # Applying the binary operator '+=' (line 152)
                    result_iadd_501 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 20), '+=', backtracks_499, int_500)
                    # Getting the type of 'self' (line 152)
                    self_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'self')
                    # Setting the type of the member '__backtracks' of a type (line 152)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), self_502, '__backtracks', result_iadd_501)
                    
                    
                    # Getting the type of 'depth' (line 153)
                    depth_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'depth')
                    int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'int')
                    # Applying the binary operator '==' (line 153)
                    result_eq_505 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 23), '==', depth_503, int_504)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 20), result_eq_505):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 20), result_eq_505)
                        # Assigning a type to the variable 'if_condition_506' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'if_condition_506', if_condition_506)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to onexit(...): (line 153)
                        # Processing the call keyword arguments (line 153)
                        kwargs_509 = {}
                        # Getting the type of 'self' (line 153)
                        self_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 35), 'self', False)
                        # Obtaining the member 'onexit' of a type (line 153)
                        onexit_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 35), self_507, 'onexit')
                        # Calling onexit(args, kwargs) (line 153)
                        onexit_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 153, 35), onexit_508, *[], **kwargs_509)
                        
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 151)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'coords' (line 157)
                    coords_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'coords')
                    # Assigning a type to the variable 'coords_511' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'coords_511', coords_511)
                    # Testing if the for loop is going to be iterated (line 157)
                    # Testing the type of a for loop iterable (line 157)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511):
                        # Getting the type of the for loop variable (line 157)
                        for_loop_var_512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)
                        # Assigning a type to the variable 'row' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 0))
                        # Assigning a type to the variable 'col' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 1))
                        # SSA begins for a for statement (line 157)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Assigning a Name to a Name (line 159):
                        
                        # Assigning a Name to a Name (line 159):
                        # Getting the type of 'False' (line 159)
                        False_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'False')
                        # Assigning a type to the variable 'broken' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'broken', False_513)
                        
                        # Calculating list comprehension
                        # Calculating comprehension expression
                        
                        # Call to xrange(...): (line 160)
                        # Processing the call arguments (line 160)
                        int_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 54), 'int')
                        # Processing the call keyword arguments (line 160)
                        kwargs_529 = {}
                        # Getting the type of 'xrange' (line 160)
                        xrange_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 47), 'xrange', False)
                        # Calling xrange(args, kwargs) (line 160)
                        xrange_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 160, 47), xrange_527, *[int_528], **kwargs_529)
                        
                        comprehension_531 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), xrange_call_result_530)
                        # Assigning a type to the variable 'i' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i', comprehension_531)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 160)
                        i_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 89), 'i')
                        
                        # Call to mergemask(...): (line 160)
                        # Processing the call arguments (line 160)
                        # Getting the type of 'row' (line 160)
                        row_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 77), 'row', False)
                        # Getting the type of 'col' (line 160)
                        col_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 82), 'col', False)
                        # Processing the call keyword arguments (line 160)
                        kwargs_520 = {}
                        # Getting the type of '_board' (line 160)
                        _board_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 60), '_board', False)
                        # Obtaining the member 'mergemask' of a type (line 160)
                        mergemask_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), _board_516, 'mergemask')
                        # Calling mergemask(args, kwargs) (line 160)
                        mergemask_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_517, *[row_518, col_519], **kwargs_520)
                        
                        # Obtaining the member 'v' of a type (line 160)
                        v_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_call_result_521, 'v')
                        # Obtaining the member '__getitem__' of a type (line 160)
                        getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), v_522, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
                        subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), getitem___523, i_515)
                        
                        # Getting the type of 'True' (line 160)
                        True_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 95), 'True')
                        # Applying the binary operator '==' (line 160)
                        result_eq_526 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 60), '==', subscript_call_result_524, True_525)
                        
                        # Getting the type of 'i' (line 160)
                        i_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i')
                        list_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'list')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), list_532, i_514)
                        # Assigning a type to the variable 'list_532' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'list_532', list_532)
                        # Testing if the for loop is going to be iterated (line 160)
                        # Testing the type of a for loop iterable (line 160)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 160, 24), list_532):
                            # Getting the type of the for loop variable (line 160)
                            for_loop_var_533 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)
                            # Assigning a type to the variable 'val' (line 160)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'val', for_loop_var_533)
                            # SSA begins for a for statement (line 160)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'board' (line 161)
                            board_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'board')
                            # Obtaining the member 'completeSearch' of a type (line 161)
                            completeSearch_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 35), board_534, 'completeSearch')
                            # Applying the 'not' unary operator (line 161)
                            result_not__536 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'not', completeSearch_535)
                            
                            
                            # Getting the type of 'self' (line 161)
                            self_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 60), 'self')
                            # Obtaining the member '__status' of a type (line 161)
                            status_538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 60), self_537, '__status')
                            int_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 77), 'int')
                            # Applying the binary operator '==' (line 161)
                            result_eq_540 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 60), '==', status_538, int_539)
                            
                            # Applying the binary operator 'and' (line 161)
                            result_and_keyword_541 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'and', result_not__536, result_eq_540)
                            
                            # Testing if the type of an if condition is none (line 161)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 161)
                                if_condition_542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541)
                                # Assigning a type to the variable 'if_condition_542' (line 161)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'if_condition_542', if_condition_542)
                                # SSA begins for if statement (line 161)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Name (line 162):
                                
                                # Assigning a Name to a Name (line 162):
                                # Getting the type of 'True' (line 162)
                                True_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'True')
                                # Assigning a type to the variable 'broken' (line 162)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'broken', True_543)
                                # SSA join for if statement (line 161)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Getting the type of 'val' (line 164)
                            val_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val')
                            int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 35), 'int')
                            # Applying the binary operator '+=' (line 164)
                            result_iadd_546 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 28), '+=', val_544, int_545)
                            # Assigning a type to the variable 'val' (line 164)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val', result_iadd_546)
                            
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Call to clone(...): (line 165)
                            # Processing the call keyword arguments (line 165)
                            kwargs_549 = {}
                            # Getting the type of '_board' (line 165)
                            _board_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), '_board', False)
                            # Obtaining the member 'clone' of a type (line 165)
                            clone_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), _board_547, 'clone')
                            # Calling clone(args, kwargs) (line 165)
                            clone_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), clone_548, *[], **kwargs_549)
                            
                            # Assigning a type to the variable 't' (line 165)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 't', clone_call_result_550)
                            
                            # Call to setval(...): (line 166)
                            # Processing the call arguments (line 166)
                            # Getting the type of 'row' (line 166)
                            row_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'row', False)
                            # Getting the type of 'col' (line 166)
                            col_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 42), 'col', False)
                            # Getting the type of 'val' (line 166)
                            val_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'val', False)
                            # Processing the call keyword arguments (line 166)
                            kwargs_556 = {}
                            # Getting the type of 't' (line 166)
                            t_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 't', False)
                            # Obtaining the member 'setval' of a type (line 166)
                            setval_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), t_551, 'setval')
                            # Calling setval(args, kwargs) (line 166)
                            setval_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 166, 28), setval_552, *[row_553, col_554, val_555], **kwargs_556)
                            
                            
                            # Call to __solve(...): (line 167)
                            # Processing the call arguments (line 167)
                            # Getting the type of 't' (line 167)
                            t_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 't', False)
                            # Getting the type of 'depth' (line 167)
                            depth_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'depth', False)
                            int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 52), 'int')
                            # Applying the binary operator '+' (line 167)
                            result_add_563 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 44), '+', depth_561, int_562)
                            
                            # Processing the call keyword arguments (line 167)
                            kwargs_564 = {}
                            # Getting the type of 'self' (line 167)
                            self_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'self', False)
                            # Obtaining the member '__solve' of a type (line 167)
                            solve_559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 28), self_558, '__solve')
                            # Calling __solve(args, kwargs) (line 167)
                            solve_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 167, 28), solve_559, *[t_560, result_add_563], **kwargs_564)
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # Getting the type of 'broken' (line 170)
                        broken_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'broken')
                        # Testing if the type of an if condition is none (line 170)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 170)
                            if_condition_567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566)
                            # Assigning a type to the variable 'if_condition_567' (line 170)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'if_condition_567', if_condition_567)
                            # SSA begins for if statement (line 170)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # SSA join for if statement (line 170)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # SSA join for if statement (line 151)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 136)
                if_condition_446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 12), result_le_445)
                # Assigning a type to the variable 'if_condition_446' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'if_condition_446', if_condition_446)
                # SSA begins for if statement (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to add(...): (line 137)
                # Processing the call arguments (line 137)
                
                # Call to boardRep(...): (line 137)
                # Processing the call arguments (line 137)
                # Getting the type of '_board' (line 137)
                _board_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 44), '_board', False)
                # Processing the call keyword arguments (line 137)
                kwargs_452 = {}
                # Getting the type of 'boardRep' (line 137)
                boardRep_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 35), 'boardRep', False)
                # Calling boardRep(args, kwargs) (line 137)
                boardRep_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 137, 35), boardRep_450, *[_board_451], **kwargs_452)
                
                # Processing the call keyword arguments (line 137)
                kwargs_454 = {}
                # Getting the type of 'self' (line 137)
                self_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'self', False)
                # Obtaining the member 'solutions' of a type (line 137)
                solutions_448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), self_447, 'solutions')
                # Obtaining the member 'add' of a type (line 137)
                add_449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), solutions_448, 'add')
                # Calling add(args, kwargs) (line 137)
                add_call_result_455 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), add_449, *[boardRep_call_result_453], **kwargs_454)
                
                
                # Getting the type of 'depth' (line 140)
                depth_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'depth')
                int_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 28), 'int')
                # Applying the binary operator '==' (line 140)
                result_eq_458 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 19), '==', depth_456, int_457)
                
                # Testing if the type of an if condition is none (line 140)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 140, 16), result_eq_458):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 140)
                    if_condition_459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 16), result_eq_458)
                    # Assigning a type to the variable 'if_condition_459' (line 140)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'if_condition_459', if_condition_459)
                    # SSA begins for if statement (line 140)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to onexit(...): (line 140)
                    # Processing the call keyword arguments (line 140)
                    kwargs_462 = {}
                    # Getting the type of 'self' (line 140)
                    self_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 31), 'self', False)
                    # Obtaining the member 'onexit' of a type (line 140)
                    onexit_461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 31), self_460, 'onexit')
                    # Calling onexit(args, kwargs) (line 140)
                    onexit_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 140, 31), onexit_461, *[], **kwargs_462)
                    
                    # SSA join for if statement (line 140)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'board' (line 141)
                board_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'board')
                # Obtaining the member 'completeSearch' of a type (line 141)
                completeSearch_465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 23), board_464, 'completeSearch')
                # Applying the 'not' unary operator (line 141)
                result_not__466 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 19), 'not', completeSearch_465)
                
                # Testing if the type of an if condition is none (line 141)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 16), result_not__466):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 141)
                    if_condition_467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 16), result_not__466)
                    # Assigning a type to the variable 'if_condition_467' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'if_condition_467', if_condition_467)
                    # SSA begins for if statement (line 141)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to onexit(...): (line 142)
                    # Processing the call keyword arguments (line 142)
                    kwargs_470 = {}
                    # Getting the type of 'self' (line 142)
                    self_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'self', False)
                    # Obtaining the member 'onexit' of a type (line 142)
                    onexit_469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), self_468, 'onexit')
                    # Calling onexit(args, kwargs) (line 142)
                    onexit_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), onexit_469, *[], **kwargs_470)
                    
                    # SSA join for if statement (line 141)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 136)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 146)
                self_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'self')
                # Obtaining the member '__turns' of a type (line 146)
                turns_473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), self_472, '__turns')
                int_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 32), 'int')
                # Applying the binary operator '+=' (line 146)
                result_iadd_475 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 16), '+=', turns_473, int_474)
                # Getting the type of 'self' (line 146)
                self_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'self')
                # Setting the type of the member '__turns' of a type (line 146)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), self_476, '__turns', result_iadd_475)
                
                
                # Getting the type of 'depth' (line 147)
                depth_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'depth')
                # Getting the type of 'self' (line 147)
                self_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'self')
                # Obtaining the member '__maxdepth' of a type (line 147)
                maxdepth_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 27), self_478, '__maxdepth')
                # Applying the binary operator '>' (line 147)
                result_gt_480 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), '>', depth_477, maxdepth_479)
                
                # Testing if the type of an if condition is none (line 147)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 147, 16), result_gt_480):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 147)
                    if_condition_481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 16), result_gt_480)
                    # Assigning a type to the variable 'if_condition_481' (line 147)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'if_condition_481', if_condition_481)
                    # SSA begins for if statement (line 147)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Attribute (line 147):
                    
                    # Assigning a Name to a Attribute (line 147):
                    # Getting the type of 'depth' (line 147)
                    depth_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 62), 'depth')
                    # Getting the type of 'self' (line 147)
                    self_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 44), 'self')
                    # Setting the type of the member '__maxdepth' of a type (line 147)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 44), self_483, '__maxdepth', depth_482)
                    # SSA join for if statement (line 147)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Tuple (line 150):
                
                # Assigning a Call to a Name:
                
                # Call to findmincounts(...): (line 150)
                # Processing the call keyword arguments (line 150)
                kwargs_486 = {}
                # Getting the type of '_board' (line 150)
                _board_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 33), '_board', False)
                # Obtaining the member 'findmincounts' of a type (line 150)
                findmincounts_485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 33), _board_484, 'findmincounts')
                # Calling findmincounts(args, kwargs) (line 150)
                findmincounts_call_result_487 = invoke(stypy.reporting.localization.Localization(__file__, 150, 33), findmincounts_485, *[], **kwargs_486)
                
                # Assigning a type to the variable 'call_assignment_1' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_1', findmincounts_call_result_487)
                
                # Assigning a Call to a Name (line 150):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1' (line 150)
                call_assignment_1_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_1', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_489 = stypy_get_value_from_tuple(call_assignment_1_488, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_2' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_2', stypy_get_value_from_tuple_call_result_489)
                
                # Assigning a Name to a Name (line 150):
                # Getting the type of 'call_assignment_2' (line 150)
                call_assignment_2_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_2')
                # Assigning a type to the variable 'mincnt' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'mincnt', call_assignment_2_490)
                
                # Assigning a Call to a Name (line 150):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1' (line 150)
                call_assignment_1_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_1', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_492 = stypy_get_value_from_tuple(call_assignment_1_491, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_3' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_3', stypy_get_value_from_tuple_call_result_492)
                
                # Assigning a Name to a Name (line 150):
                # Getting the type of 'call_assignment_3' (line 150)
                call_assignment_3_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_3')
                # Assigning a type to the variable 'coords' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'coords', call_assignment_3_493)
                
                # Getting the type of 'mincnt' (line 151)
                mincnt_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'mincnt')
                int_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'int')
                # Applying the binary operator '<=' (line 151)
                result_le_496 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 19), '<=', mincnt_494, int_495)
                
                # Testing if the type of an if condition is none (line 151)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 16), result_le_496):
                    
                    # Getting the type of 'coords' (line 157)
                    coords_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'coords')
                    # Assigning a type to the variable 'coords_511' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'coords_511', coords_511)
                    # Testing if the for loop is going to be iterated (line 157)
                    # Testing the type of a for loop iterable (line 157)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511):
                        # Getting the type of the for loop variable (line 157)
                        for_loop_var_512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)
                        # Assigning a type to the variable 'row' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 0))
                        # Assigning a type to the variable 'col' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 1))
                        # SSA begins for a for statement (line 157)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Assigning a Name to a Name (line 159):
                        
                        # Assigning a Name to a Name (line 159):
                        # Getting the type of 'False' (line 159)
                        False_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'False')
                        # Assigning a type to the variable 'broken' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'broken', False_513)
                        
                        # Calculating list comprehension
                        # Calculating comprehension expression
                        
                        # Call to xrange(...): (line 160)
                        # Processing the call arguments (line 160)
                        int_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 54), 'int')
                        # Processing the call keyword arguments (line 160)
                        kwargs_529 = {}
                        # Getting the type of 'xrange' (line 160)
                        xrange_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 47), 'xrange', False)
                        # Calling xrange(args, kwargs) (line 160)
                        xrange_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 160, 47), xrange_527, *[int_528], **kwargs_529)
                        
                        comprehension_531 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), xrange_call_result_530)
                        # Assigning a type to the variable 'i' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i', comprehension_531)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 160)
                        i_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 89), 'i')
                        
                        # Call to mergemask(...): (line 160)
                        # Processing the call arguments (line 160)
                        # Getting the type of 'row' (line 160)
                        row_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 77), 'row', False)
                        # Getting the type of 'col' (line 160)
                        col_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 82), 'col', False)
                        # Processing the call keyword arguments (line 160)
                        kwargs_520 = {}
                        # Getting the type of '_board' (line 160)
                        _board_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 60), '_board', False)
                        # Obtaining the member 'mergemask' of a type (line 160)
                        mergemask_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), _board_516, 'mergemask')
                        # Calling mergemask(args, kwargs) (line 160)
                        mergemask_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_517, *[row_518, col_519], **kwargs_520)
                        
                        # Obtaining the member 'v' of a type (line 160)
                        v_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_call_result_521, 'v')
                        # Obtaining the member '__getitem__' of a type (line 160)
                        getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), v_522, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
                        subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), getitem___523, i_515)
                        
                        # Getting the type of 'True' (line 160)
                        True_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 95), 'True')
                        # Applying the binary operator '==' (line 160)
                        result_eq_526 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 60), '==', subscript_call_result_524, True_525)
                        
                        # Getting the type of 'i' (line 160)
                        i_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i')
                        list_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'list')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), list_532, i_514)
                        # Assigning a type to the variable 'list_532' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'list_532', list_532)
                        # Testing if the for loop is going to be iterated (line 160)
                        # Testing the type of a for loop iterable (line 160)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 160, 24), list_532):
                            # Getting the type of the for loop variable (line 160)
                            for_loop_var_533 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)
                            # Assigning a type to the variable 'val' (line 160)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'val', for_loop_var_533)
                            # SSA begins for a for statement (line 160)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'board' (line 161)
                            board_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'board')
                            # Obtaining the member 'completeSearch' of a type (line 161)
                            completeSearch_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 35), board_534, 'completeSearch')
                            # Applying the 'not' unary operator (line 161)
                            result_not__536 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'not', completeSearch_535)
                            
                            
                            # Getting the type of 'self' (line 161)
                            self_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 60), 'self')
                            # Obtaining the member '__status' of a type (line 161)
                            status_538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 60), self_537, '__status')
                            int_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 77), 'int')
                            # Applying the binary operator '==' (line 161)
                            result_eq_540 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 60), '==', status_538, int_539)
                            
                            # Applying the binary operator 'and' (line 161)
                            result_and_keyword_541 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'and', result_not__536, result_eq_540)
                            
                            # Testing if the type of an if condition is none (line 161)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 161)
                                if_condition_542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541)
                                # Assigning a type to the variable 'if_condition_542' (line 161)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'if_condition_542', if_condition_542)
                                # SSA begins for if statement (line 161)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Name (line 162):
                                
                                # Assigning a Name to a Name (line 162):
                                # Getting the type of 'True' (line 162)
                                True_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'True')
                                # Assigning a type to the variable 'broken' (line 162)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'broken', True_543)
                                # SSA join for if statement (line 161)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Getting the type of 'val' (line 164)
                            val_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val')
                            int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 35), 'int')
                            # Applying the binary operator '+=' (line 164)
                            result_iadd_546 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 28), '+=', val_544, int_545)
                            # Assigning a type to the variable 'val' (line 164)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val', result_iadd_546)
                            
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Call to clone(...): (line 165)
                            # Processing the call keyword arguments (line 165)
                            kwargs_549 = {}
                            # Getting the type of '_board' (line 165)
                            _board_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), '_board', False)
                            # Obtaining the member 'clone' of a type (line 165)
                            clone_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), _board_547, 'clone')
                            # Calling clone(args, kwargs) (line 165)
                            clone_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), clone_548, *[], **kwargs_549)
                            
                            # Assigning a type to the variable 't' (line 165)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 't', clone_call_result_550)
                            
                            # Call to setval(...): (line 166)
                            # Processing the call arguments (line 166)
                            # Getting the type of 'row' (line 166)
                            row_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'row', False)
                            # Getting the type of 'col' (line 166)
                            col_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 42), 'col', False)
                            # Getting the type of 'val' (line 166)
                            val_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'val', False)
                            # Processing the call keyword arguments (line 166)
                            kwargs_556 = {}
                            # Getting the type of 't' (line 166)
                            t_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 't', False)
                            # Obtaining the member 'setval' of a type (line 166)
                            setval_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), t_551, 'setval')
                            # Calling setval(args, kwargs) (line 166)
                            setval_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 166, 28), setval_552, *[row_553, col_554, val_555], **kwargs_556)
                            
                            
                            # Call to __solve(...): (line 167)
                            # Processing the call arguments (line 167)
                            # Getting the type of 't' (line 167)
                            t_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 't', False)
                            # Getting the type of 'depth' (line 167)
                            depth_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'depth', False)
                            int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 52), 'int')
                            # Applying the binary operator '+' (line 167)
                            result_add_563 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 44), '+', depth_561, int_562)
                            
                            # Processing the call keyword arguments (line 167)
                            kwargs_564 = {}
                            # Getting the type of 'self' (line 167)
                            self_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'self', False)
                            # Obtaining the member '__solve' of a type (line 167)
                            solve_559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 28), self_558, '__solve')
                            # Calling __solve(args, kwargs) (line 167)
                            solve_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 167, 28), solve_559, *[t_560, result_add_563], **kwargs_564)
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # Getting the type of 'broken' (line 170)
                        broken_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'broken')
                        # Testing if the type of an if condition is none (line 170)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 170)
                            if_condition_567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566)
                            # Assigning a type to the variable 'if_condition_567' (line 170)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'if_condition_567', if_condition_567)
                            # SSA begins for if statement (line 170)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # SSA join for if statement (line 170)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                else:
                    
                    # Testing the type of an if condition (line 151)
                    if_condition_497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 16), result_le_496)
                    # Assigning a type to the variable 'if_condition_497' (line 151)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'if_condition_497', if_condition_497)
                    # SSA begins for if statement (line 151)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'self' (line 152)
                    self_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'self')
                    # Obtaining the member '__backtracks' of a type (line 152)
                    backtracks_499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), self_498, '__backtracks')
                    int_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 41), 'int')
                    # Applying the binary operator '+=' (line 152)
                    result_iadd_501 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 20), '+=', backtracks_499, int_500)
                    # Getting the type of 'self' (line 152)
                    self_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'self')
                    # Setting the type of the member '__backtracks' of a type (line 152)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), self_502, '__backtracks', result_iadd_501)
                    
                    
                    # Getting the type of 'depth' (line 153)
                    depth_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'depth')
                    int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'int')
                    # Applying the binary operator '==' (line 153)
                    result_eq_505 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 23), '==', depth_503, int_504)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 20), result_eq_505):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 20), result_eq_505)
                        # Assigning a type to the variable 'if_condition_506' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'if_condition_506', if_condition_506)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to onexit(...): (line 153)
                        # Processing the call keyword arguments (line 153)
                        kwargs_509 = {}
                        # Getting the type of 'self' (line 153)
                        self_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 35), 'self', False)
                        # Obtaining the member 'onexit' of a type (line 153)
                        onexit_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 35), self_507, 'onexit')
                        # Calling onexit(args, kwargs) (line 153)
                        onexit_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 153, 35), onexit_508, *[], **kwargs_509)
                        
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 151)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'coords' (line 157)
                    coords_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'coords')
                    # Assigning a type to the variable 'coords_511' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'coords_511', coords_511)
                    # Testing if the for loop is going to be iterated (line 157)
                    # Testing the type of a for loop iterable (line 157)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511):
                        # Getting the type of the for loop variable (line 157)
                        for_loop_var_512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 20), coords_511)
                        # Assigning a type to the variable 'row' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 0))
                        # Assigning a type to the variable 'col' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), for_loop_var_512, 2, 1))
                        # SSA begins for a for statement (line 157)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Assigning a Name to a Name (line 159):
                        
                        # Assigning a Name to a Name (line 159):
                        # Getting the type of 'False' (line 159)
                        False_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'False')
                        # Assigning a type to the variable 'broken' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'broken', False_513)
                        
                        # Calculating list comprehension
                        # Calculating comprehension expression
                        
                        # Call to xrange(...): (line 160)
                        # Processing the call arguments (line 160)
                        int_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 54), 'int')
                        # Processing the call keyword arguments (line 160)
                        kwargs_529 = {}
                        # Getting the type of 'xrange' (line 160)
                        xrange_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 47), 'xrange', False)
                        # Calling xrange(args, kwargs) (line 160)
                        xrange_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 160, 47), xrange_527, *[int_528], **kwargs_529)
                        
                        comprehension_531 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), xrange_call_result_530)
                        # Assigning a type to the variable 'i' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i', comprehension_531)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 160)
                        i_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 89), 'i')
                        
                        # Call to mergemask(...): (line 160)
                        # Processing the call arguments (line 160)
                        # Getting the type of 'row' (line 160)
                        row_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 77), 'row', False)
                        # Getting the type of 'col' (line 160)
                        col_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 82), 'col', False)
                        # Processing the call keyword arguments (line 160)
                        kwargs_520 = {}
                        # Getting the type of '_board' (line 160)
                        _board_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 60), '_board', False)
                        # Obtaining the member 'mergemask' of a type (line 160)
                        mergemask_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), _board_516, 'mergemask')
                        # Calling mergemask(args, kwargs) (line 160)
                        mergemask_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_517, *[row_518, col_519], **kwargs_520)
                        
                        # Obtaining the member 'v' of a type (line 160)
                        v_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), mergemask_call_result_521, 'v')
                        # Obtaining the member '__getitem__' of a type (line 160)
                        getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 60), v_522, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
                        subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 160, 60), getitem___523, i_515)
                        
                        # Getting the type of 'True' (line 160)
                        True_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 95), 'True')
                        # Applying the binary operator '==' (line 160)
                        result_eq_526 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 60), '==', subscript_call_result_524, True_525)
                        
                        # Getting the type of 'i' (line 160)
                        i_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'i')
                        list_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'list')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), list_532, i_514)
                        # Assigning a type to the variable 'list_532' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'list_532', list_532)
                        # Testing if the for loop is going to be iterated (line 160)
                        # Testing the type of a for loop iterable (line 160)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 160, 24), list_532):
                            # Getting the type of the for loop variable (line 160)
                            for_loop_var_533 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 24), list_532)
                            # Assigning a type to the variable 'val' (line 160)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'val', for_loop_var_533)
                            # SSA begins for a for statement (line 160)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'board' (line 161)
                            board_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'board')
                            # Obtaining the member 'completeSearch' of a type (line 161)
                            completeSearch_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 35), board_534, 'completeSearch')
                            # Applying the 'not' unary operator (line 161)
                            result_not__536 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'not', completeSearch_535)
                            
                            
                            # Getting the type of 'self' (line 161)
                            self_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 60), 'self')
                            # Obtaining the member '__status' of a type (line 161)
                            status_538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 60), self_537, '__status')
                            int_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 77), 'int')
                            # Applying the binary operator '==' (line 161)
                            result_eq_540 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 60), '==', status_538, int_539)
                            
                            # Applying the binary operator 'and' (line 161)
                            result_and_keyword_541 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), 'and', result_not__536, result_eq_540)
                            
                            # Testing if the type of an if condition is none (line 161)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 161)
                                if_condition_542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 28), result_and_keyword_541)
                                # Assigning a type to the variable 'if_condition_542' (line 161)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'if_condition_542', if_condition_542)
                                # SSA begins for if statement (line 161)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Name to a Name (line 162):
                                
                                # Assigning a Name to a Name (line 162):
                                # Getting the type of 'True' (line 162)
                                True_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'True')
                                # Assigning a type to the variable 'broken' (line 162)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'broken', True_543)
                                # SSA join for if statement (line 161)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Getting the type of 'val' (line 164)
                            val_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val')
                            int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 35), 'int')
                            # Applying the binary operator '+=' (line 164)
                            result_iadd_546 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 28), '+=', val_544, int_545)
                            # Assigning a type to the variable 'val' (line 164)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'val', result_iadd_546)
                            
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Assigning a Call to a Name (line 165):
                            
                            # Call to clone(...): (line 165)
                            # Processing the call keyword arguments (line 165)
                            kwargs_549 = {}
                            # Getting the type of '_board' (line 165)
                            _board_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), '_board', False)
                            # Obtaining the member 'clone' of a type (line 165)
                            clone_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), _board_547, 'clone')
                            # Calling clone(args, kwargs) (line 165)
                            clone_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), clone_548, *[], **kwargs_549)
                            
                            # Assigning a type to the variable 't' (line 165)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 't', clone_call_result_550)
                            
                            # Call to setval(...): (line 166)
                            # Processing the call arguments (line 166)
                            # Getting the type of 'row' (line 166)
                            row_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'row', False)
                            # Getting the type of 'col' (line 166)
                            col_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 42), 'col', False)
                            # Getting the type of 'val' (line 166)
                            val_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'val', False)
                            # Processing the call keyword arguments (line 166)
                            kwargs_556 = {}
                            # Getting the type of 't' (line 166)
                            t_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 't', False)
                            # Obtaining the member 'setval' of a type (line 166)
                            setval_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), t_551, 'setval')
                            # Calling setval(args, kwargs) (line 166)
                            setval_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 166, 28), setval_552, *[row_553, col_554, val_555], **kwargs_556)
                            
                            
                            # Call to __solve(...): (line 167)
                            # Processing the call arguments (line 167)
                            # Getting the type of 't' (line 167)
                            t_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 't', False)
                            # Getting the type of 'depth' (line 167)
                            depth_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'depth', False)
                            int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 52), 'int')
                            # Applying the binary operator '+' (line 167)
                            result_add_563 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 44), '+', depth_561, int_562)
                            
                            # Processing the call keyword arguments (line 167)
                            kwargs_564 = {}
                            # Getting the type of 'self' (line 167)
                            self_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'self', False)
                            # Obtaining the member '__solve' of a type (line 167)
                            solve_559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 28), self_558, '__solve')
                            # Calling __solve(args, kwargs) (line 167)
                            solve_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 167, 28), solve_559, *[t_560, result_add_563], **kwargs_564)
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # Getting the type of 'broken' (line 170)
                        broken_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'broken')
                        # Testing if the type of an if condition is none (line 170)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 170)
                            if_condition_567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 24), broken_566)
                            # Assigning a type to the variable 'if_condition_567' (line 170)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'if_condition_567', if_condition_567)
                            # SSA begins for if statement (line 170)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # SSA join for if statement (line 170)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # SSA join for if statement (line 151)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 132)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__solve' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_568)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__solve'
        return stypy_return_type_568


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.clone.__dict__.__setitem__('stypy_localization', localization)
        board.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.clone.__dict__.__setitem__('stypy_function_name', 'board.clone')
        board.clone.__dict__.__setitem__('stypy_param_names_list', [])
        board.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.clone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to board(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_570 = {}
        # Getting the type of 'board' (line 175)
        board_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'board', False)
        # Calling board(args, kwargs) (line 175)
        board_call_result_571 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), board_569, *[], **kwargs_570)
        
        # Assigning a type to the variable 'ret' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ret', board_call_result_571)
        
        
        # Call to xrange(...): (line 176)
        # Processing the call arguments (line 176)
        int_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 26), 'int')
        # Processing the call keyword arguments (line 176)
        kwargs_574 = {}
        # Getting the type of 'xrange' (line 176)
        xrange_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 176)
        xrange_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), xrange_572, *[int_573], **kwargs_574)
        
        # Assigning a type to the variable 'xrange_call_result_575' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'xrange_call_result_575', xrange_call_result_575)
        # Testing if the for loop is going to be iterated (line 176)
        # Testing the type of a for loop iterable (line 176)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 176, 8), xrange_call_result_575)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 176, 8), xrange_call_result_575):
            # Getting the type of the for loop variable (line 176)
            for_loop_var_576 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 176, 8), xrange_call_result_575)
            # Assigning a type to the variable 'row' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'row', for_loop_var_576)
            # SSA begins for a for statement (line 176)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to xrange(...): (line 177)
            # Processing the call arguments (line 177)
            int_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'int')
            # Processing the call keyword arguments (line 177)
            kwargs_579 = {}
            # Getting the type of 'xrange' (line 177)
            xrange_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'xrange', False)
            # Calling xrange(args, kwargs) (line 177)
            xrange_call_result_580 = invoke(stypy.reporting.localization.Localization(__file__, 177, 23), xrange_577, *[int_578], **kwargs_579)
            
            # Assigning a type to the variable 'xrange_call_result_580' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'xrange_call_result_580', xrange_call_result_580)
            # Testing if the for loop is going to be iterated (line 177)
            # Testing the type of a for loop iterable (line 177)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 12), xrange_call_result_580)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 177, 12), xrange_call_result_580):
                # Getting the type of the for loop variable (line 177)
                for_loop_var_581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 12), xrange_call_result_580)
                # Assigning a type to the variable 'col' (line 177)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'col', for_loop_var_581)
                # SSA begins for a for statement (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 178)
                col_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 178)
                row_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'row')
                # Getting the type of 'self' (line 178)
                self_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'self')
                # Obtaining the member 'final' of a type (line 178)
                final_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 19), self_584, 'final')
                # Obtaining the member '__getitem__' of a type (line 178)
                getitem___586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 19), final_585, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 178)
                subscript_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 178, 19), getitem___586, row_583)
                
                # Obtaining the member '__getitem__' of a type (line 178)
                getitem___588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 19), subscript_call_result_587, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 178)
                subscript_call_result_589 = invoke(stypy.reporting.localization.Localization(__file__, 178, 19), getitem___588, col_582)
                
                # Testing if the type of an if condition is none (line 178)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 178, 16), subscript_call_result_589):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 178)
                    if_condition_590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 16), subscript_call_result_589)
                    # Assigning a type to the variable 'if_condition_590' (line 178)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'if_condition_590', if_condition_590)
                    # SSA begins for if statement (line 178)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to setval(...): (line 179)
                    # Processing the call arguments (line 179)
                    # Getting the type of 'row' (line 179)
                    row_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'row', False)
                    # Getting the type of 'col' (line 179)
                    col_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 36), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 179)
                    col_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 57), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 179)
                    row_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 52), 'row', False)
                    # Getting the type of 'self' (line 179)
                    self_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 41), 'self', False)
                    # Obtaining the member 'final' of a type (line 179)
                    final_598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 41), self_597, 'final')
                    # Obtaining the member '__getitem__' of a type (line 179)
                    getitem___599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 41), final_598, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
                    subscript_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 179, 41), getitem___599, row_596)
                    
                    # Obtaining the member '__getitem__' of a type (line 179)
                    getitem___601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 41), subscript_call_result_600, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
                    subscript_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 179, 41), getitem___601, col_595)
                    
                    # Processing the call keyword arguments (line 179)
                    kwargs_603 = {}
                    # Getting the type of 'ret' (line 179)
                    ret_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'ret', False)
                    # Obtaining the member 'setval' of a type (line 179)
                    setval_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), ret_591, 'setval')
                    # Calling setval(args, kwargs) (line 179)
                    setval_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), setval_592, *[row_593, col_594, subscript_call_result_602], **kwargs_603)
                    
                    # SSA join for if statement (line 178)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'ret' (line 180)
        ret_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stypy_return_type', ret_605)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_606


    @norecursion
    def mergemask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mergemask'
        module_type_store = module_type_store.open_function_context('mergemask', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.mergemask.__dict__.__setitem__('stypy_localization', localization)
        board.mergemask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.mergemask.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.mergemask.__dict__.__setitem__('stypy_function_name', 'board.mergemask')
        board.mergemask.__dict__.__setitem__('stypy_param_names_list', ['row', 'col'])
        board.mergemask.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.mergemask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.mergemask.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.mergemask.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.mergemask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.mergemask.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.mergemask', ['row', 'col'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mergemask', localization, ['row', 'col'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mergemask(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 183)
        row_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 25), 'row')
        # Getting the type of 'self' (line 183)
        self_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'self')
        # Obtaining the member 'rows' of a type (line 183)
        rows_609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 15), self_608, 'rows')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 15), rows_609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 183, 15), getitem___610, row_607)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 183)
        col_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 42), 'col')
        # Getting the type of 'self' (line 183)
        self_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'self')
        # Obtaining the member 'cols' of a type (line 183)
        cols_614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 32), self_613, 'cols')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 32), cols_614, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 183, 32), getitem___615, col_612)
        
        # Applying the binary operator '&' (line 183)
        result_and__617 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 15), '&', subscript_call_result_611, subscript_call_result_616)
        
        
        # Obtaining the type of the subscript
        
        # Call to cell(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'col' (line 183)
        col_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 85), 'col', False)
        # Processing the call keyword arguments (line 183)
        kwargs_621 = {}
        # Getting the type of 'self' (line 183)
        self_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 75), 'self', False)
        # Obtaining the member 'cell' of a type (line 183)
        cell_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 75), self_618, 'cell')
        # Calling cell(args, kwargs) (line 183)
        cell_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 183, 75), cell_619, *[col_620], **kwargs_621)
        
        
        # Obtaining the type of the subscript
        
        # Call to cell(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'row' (line 183)
        row_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 69), 'row', False)
        # Processing the call keyword arguments (line 183)
        kwargs_626 = {}
        # Getting the type of 'self' (line 183)
        self_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 59), 'self', False)
        # Obtaining the member 'cell' of a type (line 183)
        cell_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 59), self_623, 'cell')
        # Calling cell(args, kwargs) (line 183)
        cell_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 183, 59), cell_624, *[row_625], **kwargs_626)
        
        # Getting the type of 'self' (line 183)
        self_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 49), 'self')
        # Obtaining the member 'cels' of a type (line 183)
        cels_629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 49), self_628, 'cels')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 49), cels_629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_631 = invoke(stypy.reporting.localization.Localization(__file__, 183, 49), getitem___630, cell_call_result_627)
        
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 49), subscript_call_result_631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 183, 49), getitem___632, cell_call_result_622)
        
        # Applying the binary operator '&' (line 183)
        result_and__634 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 47), '&', result_and__617, subscript_call_result_633)
        
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', result_and__634)
        
        # ################# End of 'mergemask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mergemask' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_635)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mergemask'
        return stypy_return_type_635


    @norecursion
    def findmincounts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'findmincounts'
        module_type_store = module_type_store.open_function_context('findmincounts', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.findmincounts.__dict__.__setitem__('stypy_localization', localization)
        board.findmincounts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.findmincounts.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.findmincounts.__dict__.__setitem__('stypy_function_name', 'board.findmincounts')
        board.findmincounts.__dict__.__setitem__('stypy_param_names_list', [])
        board.findmincounts.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.findmincounts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.findmincounts.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.findmincounts.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.findmincounts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.findmincounts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.findmincounts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'findmincounts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'findmincounts(...)' code ##################

        
        # Assigning a List to a Name (line 187):
        
        # Assigning a List to a Name (line 187):
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        
        # Assigning a type to the variable 'masks' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'masks', list_636)
        
        
        # Call to xrange(...): (line 188)
        # Processing the call arguments (line 188)
        int_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 26), 'int')
        # Processing the call keyword arguments (line 188)
        kwargs_639 = {}
        # Getting the type of 'xrange' (line 188)
        xrange_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 188)
        xrange_call_result_640 = invoke(stypy.reporting.localization.Localization(__file__, 188, 19), xrange_637, *[int_638], **kwargs_639)
        
        # Assigning a type to the variable 'xrange_call_result_640' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'xrange_call_result_640', xrange_call_result_640)
        # Testing if the for loop is going to be iterated (line 188)
        # Testing the type of a for loop iterable (line 188)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 8), xrange_call_result_640)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 188, 8), xrange_call_result_640):
            # Getting the type of the for loop variable (line 188)
            for_loop_var_641 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 8), xrange_call_result_640)
            # Assigning a type to the variable 'row' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'row', for_loop_var_641)
            # SSA begins for a for statement (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to xrange(...): (line 189)
            # Processing the call arguments (line 189)
            int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 30), 'int')
            # Processing the call keyword arguments (line 189)
            kwargs_644 = {}
            # Getting the type of 'xrange' (line 189)
            xrange_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'xrange', False)
            # Calling xrange(args, kwargs) (line 189)
            xrange_call_result_645 = invoke(stypy.reporting.localization.Localization(__file__, 189, 23), xrange_642, *[int_643], **kwargs_644)
            
            # Assigning a type to the variable 'xrange_call_result_645' (line 189)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'xrange_call_result_645', xrange_call_result_645)
            # Testing if the for loop is going to be iterated (line 189)
            # Testing the type of a for loop iterable (line 189)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 189, 12), xrange_call_result_645)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 189, 12), xrange_call_result_645):
                # Getting the type of the for loop variable (line 189)
                for_loop_var_646 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 189, 12), xrange_call_result_645)
                # Assigning a type to the variable 'col' (line 189)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'col', for_loop_var_646)
                # SSA begins for a for statement (line 189)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 190)
                col_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 35), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 190)
                row_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'row')
                # Getting the type of 'self' (line 190)
                self_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'self')
                # Obtaining the member 'final' of a type (line 190)
                final_650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), self_649, 'final')
                # Obtaining the member '__getitem__' of a type (line 190)
                getitem___651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), final_650, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 190)
                subscript_call_result_652 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), getitem___651, row_648)
                
                # Obtaining the member '__getitem__' of a type (line 190)
                getitem___653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), subscript_call_result_652, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 190)
                subscript_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), getitem___653, col_647)
                
                int_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 43), 'int')
                # Applying the binary operator '==' (line 190)
                result_eq_656 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 19), '==', subscript_call_result_654, int_655)
                
                # Testing if the type of an if condition is none (line 190)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 190, 16), result_eq_656):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 190)
                    if_condition_657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 16), result_eq_656)
                    # Assigning a type to the variable 'if_condition_657' (line 190)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'if_condition_657', if_condition_657)
                    # SSA begins for if statement (line 190)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Call to cnt(...): (line 191)
                    # Processing the call keyword arguments (line 191)
                    kwargs_665 = {}
                    
                    # Call to mergemask(...): (line 191)
                    # Processing the call arguments (line 191)
                    # Getting the type of 'row' (line 191)
                    row_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 48), 'row', False)
                    # Getting the type of 'col' (line 191)
                    col_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 53), 'col', False)
                    # Processing the call keyword arguments (line 191)
                    kwargs_662 = {}
                    # Getting the type of 'self' (line 191)
                    self_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 33), 'self', False)
                    # Obtaining the member 'mergemask' of a type (line 191)
                    mergemask_659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 33), self_658, 'mergemask')
                    # Calling mergemask(args, kwargs) (line 191)
                    mergemask_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 191, 33), mergemask_659, *[row_660, col_661], **kwargs_662)
                    
                    # Obtaining the member 'cnt' of a type (line 191)
                    cnt_664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 33), mergemask_call_result_663, 'cnt')
                    # Calling cnt(args, kwargs) (line 191)
                    cnt_call_result_666 = invoke(stypy.reporting.localization.Localization(__file__, 191, 33), cnt_664, *[], **kwargs_665)
                    
                    # Assigning a type to the variable 'numallowed' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'numallowed', cnt_call_result_666)
                    
                    # Call to append(...): (line 192)
                    # Processing the call arguments (line 192)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 192)
                    tuple_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 34), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 192)
                    # Adding element type (line 192)
                    # Getting the type of 'numallowed' (line 192)
                    numallowed_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 34), 'numallowed', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 34), tuple_669, numallowed_670)
                    # Adding element type (line 192)
                    # Getting the type of 'row' (line 192)
                    row_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 46), 'row', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 34), tuple_669, row_671)
                    # Adding element type (line 192)
                    # Getting the type of 'col' (line 192)
                    col_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 51), 'col', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 34), tuple_669, col_672)
                    
                    # Processing the call keyword arguments (line 192)
                    kwargs_673 = {}
                    # Getting the type of 'masks' (line 192)
                    masks_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'masks', False)
                    # Obtaining the member 'append' of a type (line 192)
                    append_668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 20), masks_667, 'append')
                    # Calling append(args, kwargs) (line 192)
                    append_call_result_674 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), append_668, *[tuple_669], **kwargs_673)
                    
                    # SSA join for if statement (line 190)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        
        # Obtaining the type of the subscript
        int_676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 26), 'int')
        
        # Call to min(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'masks' (line 195)
        masks_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'masks', False)
        # Processing the call keyword arguments (line 195)
        kwargs_679 = {}
        # Getting the type of 'min' (line 195)
        min_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'min', False)
        # Calling min(args, kwargs) (line 195)
        min_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), min_677, *[masks_678], **kwargs_679)
        
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), min_call_result_680, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_682 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), getitem___681, int_676)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 15), tuple_675, subscript_call_result_682)
        # Adding element type (line 195)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'masks' (line 195)
        masks_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 53), 'masks')
        comprehension_705 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 31), masks_704)
        # Assigning a type to the variable 'i' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 31), 'i', comprehension_705)
        
        
        # Obtaining the type of the subscript
        int_692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 64), 'int')
        # Getting the type of 'i' (line 195)
        i_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 62), 'i')
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 62), i_693, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 195, 62), getitem___694, int_692)
        
        
        # Obtaining the type of the subscript
        int_696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 81), 'int')
        
        # Call to min(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'masks' (line 195)
        masks_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 74), 'masks', False)
        # Processing the call keyword arguments (line 195)
        kwargs_699 = {}
        # Getting the type of 'min' (line 195)
        min_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 70), 'min', False)
        # Calling min(args, kwargs) (line 195)
        min_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 195, 70), min_697, *[masks_698], **kwargs_699)
        
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 70), min_call_result_700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_702 = invoke(stypy.reporting.localization.Localization(__file__, 195, 70), getitem___701, int_696)
        
        # Applying the binary operator '==' (line 195)
        result_eq_703 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 62), '==', subscript_call_result_695, subscript_call_result_702)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        
        # Obtaining the type of the subscript
        int_684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 34), 'int')
        # Getting the type of 'i' (line 195)
        i_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 32), 'i')
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 32), i_685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 195, 32), getitem___686, int_684)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 32), tuple_683, subscript_call_result_687)
        # Adding element type (line 195)
        
        # Obtaining the type of the subscript
        int_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 40), 'int')
        # Getting the type of 'i' (line 195)
        i_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 38), 'i')
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 38), i_689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 195, 38), getitem___690, int_688)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 32), tuple_683, subscript_call_result_691)
        
        list_706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 31), list_706, tuple_683)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 15), tuple_675, list_706)
        
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', tuple_675)
        
        # ################# End of 'findmincounts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'findmincounts' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_707)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'findmincounts'
        return stypy_return_type_707


    @norecursion
    def onexit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'onexit'
        module_type_store = module_type_store.open_function_context('onexit', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.onexit.__dict__.__setitem__('stypy_localization', localization)
        board.onexit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.onexit.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.onexit.__dict__.__setitem__('stypy_function_name', 'board.onexit')
        board.onexit.__dict__.__setitem__('stypy_param_names_list', [])
        board.onexit.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.onexit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.onexit.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.onexit.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.onexit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.onexit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.onexit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'onexit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'onexit(...)' code ##################

        
        # Assigning a Call to a Attribute (line 198):
        
        # Assigning a Call to a Attribute (line 198):
        
        # Call to time(...): (line 198)
        # Processing the call keyword arguments (line 198)
        kwargs_709 = {}
        # Getting the type of 'time' (line 198)
        time_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 25), 'time', False)
        # Calling time(args, kwargs) (line 198)
        time_call_result_710 = invoke(stypy.reporting.localization.Localization(__file__, 198, 25), time_708, *[], **kwargs_709)
        
        # Getting the type of 'self' (line 198)
        self_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member '__endtime' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_711, '__endtime', time_call_result_710)
        
        # Assigning a Num to a Attribute (line 199):
        
        # Assigning a Num to a Attribute (line 199):
        int_712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 24), 'int')
        # Getting the type of 'self' (line 199)
        self_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self')
        # Setting the type of the member '__status' of a type (line 199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_713, '__status', int_712)
        # Getting the type of 'board' (line 201)
        board_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'board')
        # Obtaining the member 'notifyOnCompletion' of a type (line 201)
        notifyOnCompletion_715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), board_714, 'notifyOnCompletion')
        # Testing if the type of an if condition is none (line 201)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 201, 8), notifyOnCompletion_715):
            pass
        else:
            
            # Testing the type of an if condition (line 201)
            if_condition_716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), notifyOnCompletion_715)
            # Assigning a type to the variable 'if_condition_716' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_716', if_condition_716)
            # SSA begins for if statement (line 201)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 201)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'onexit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'onexit' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_717)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'onexit'
        return stypy_return_type_717


    @norecursion
    def stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stats'
        module_type_store = module_type_store.open_function_context('stats', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        board.stats.__dict__.__setitem__('stypy_localization', localization)
        board.stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        board.stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        board.stats.__dict__.__setitem__('stypy_function_name', 'board.stats')
        board.stats.__dict__.__setitem__('stypy_param_names_list', [])
        board.stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        board.stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        board.stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        board.stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        board.stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        board.stats.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'board.stats', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stats', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stats(...)' code ##################

        
        # Getting the type of 'self' (line 204)
        self_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'self')
        # Obtaining the member '__status' of a type (line 204)
        status_719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), self_718, '__status')
        int_720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 28), 'int')
        # Applying the binary operator '==' (line 204)
        result_eq_721 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), '==', status_719, int_720)
        
        # Testing if the type of an if condition is none (line 204)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 204, 8), result_eq_721):
            
            # Assigning a BinOp to a Name (line 207):
            
            # Assigning a BinOp to a Name (line 207):
            # Getting the type of 'self' (line 207)
            self_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'self')
            # Obtaining the member '__endtime' of a type (line 207)
            endtime_730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), self_729, '__endtime')
            # Getting the type of 'self' (line 207)
            self_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'self')
            # Obtaining the member '__starttime' of a type (line 207)
            starttime_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 33), self_731, '__starttime')
            # Applying the binary operator '-' (line 207)
            result_sub_733 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 16), '-', endtime_730, starttime_732)
            
            # Assigning a type to the variable 't' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 't', result_sub_733)
        else:
            
            # Testing the type of an if condition (line 204)
            if_condition_722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_eq_721)
            # Assigning a type to the variable 'if_condition_722' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_722', if_condition_722)
            # SSA begins for if statement (line 204)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 205):
            
            # Assigning a BinOp to a Name (line 205):
            
            # Call to time(...): (line 205)
            # Processing the call keyword arguments (line 205)
            kwargs_724 = {}
            # Getting the type of 'time' (line 205)
            time_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'time', False)
            # Calling time(args, kwargs) (line 205)
            time_call_result_725 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), time_723, *[], **kwargs_724)
            
            # Getting the type of 'self' (line 205)
            self_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 25), 'self')
            # Obtaining the member '__starttime' of a type (line 205)
            starttime_727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 25), self_726, '__starttime')
            # Applying the binary operator '-' (line 205)
            result_sub_728 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 16), '-', time_call_result_725, starttime_727)
            
            # Assigning a type to the variable 't' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 't', result_sub_728)
            # SSA branch for the else part of an if statement (line 204)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 207):
            
            # Assigning a BinOp to a Name (line 207):
            # Getting the type of 'self' (line 207)
            self_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'self')
            # Obtaining the member '__endtime' of a type (line 207)
            endtime_730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), self_729, '__endtime')
            # Getting the type of 'self' (line 207)
            self_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'self')
            # Obtaining the member '__starttime' of a type (line 207)
            starttime_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 33), self_731, '__starttime')
            # Applying the binary operator '-' (line 207)
            result_sub_733 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 16), '-', endtime_730, starttime_732)
            
            # Assigning a type to the variable 't' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 't', result_sub_733)
            # SSA join for if statement (line 204)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'dict' (line 208)
        dict_734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 208)
        # Adding element type (key, value) (line 208)
        str_735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'str', 'max depth')
        # Getting the type of 'self' (line 208)
        self_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'self')
        # Obtaining the member '__maxdepth' of a type (line 208)
        maxdepth_737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 29), self_736, '__maxdepth')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), dict_734, (str_735, maxdepth_737))
        # Adding element type (key, value) (line 208)
        str_738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 46), 'str', 'turns')
        # Getting the type of 'self' (line 208)
        self_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 55), 'self')
        # Obtaining the member '__turns' of a type (line 208)
        turns_740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 55), self_739, '__turns')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), dict_734, (str_738, turns_740))
        # Adding element type (key, value) (line 208)
        str_741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 69), 'str', 'backtracks')
        # Getting the type of 'self' (line 208)
        self_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 83), 'self')
        # Obtaining the member '__backtracks' of a type (line 208)
        backtracks_743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 83), self_742, '__backtracks')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), dict_734, (str_741, backtracks_743))
        # Adding element type (key, value) (line 208)
        str_744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 16), 'str', 'elapsed time')
        
        # Call to int(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 't' (line 209)
        t_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 't', False)
        # Processing the call keyword arguments (line 209)
        kwargs_747 = {}
        # Getting the type of 'int' (line 209)
        int_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 32), 'int', False)
        # Calling int(args, kwargs) (line 209)
        int_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 209, 32), int_745, *[t_746], **kwargs_747)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), dict_734, (str_744, int_call_result_748))
        # Adding element type (key, value) (line 208)
        str_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 40), 'str', 'boards examined')
        
        # Call to len(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'self' (line 209)
        self_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 63), 'self', False)
        # Obtaining the member 'examined' of a type (line 209)
        examined_752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 63), self_751, 'examined')
        # Processing the call keyword arguments (line 209)
        kwargs_753 = {}
        # Getting the type of 'len' (line 209)
        len_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 59), 'len', False)
        # Calling len(args, kwargs) (line 209)
        len_call_result_754 = invoke(stypy.reporting.localization.Localization(__file__, 209, 59), len_750, *[examined_752], **kwargs_753)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), dict_734, (str_749, len_call_result_754))
        # Adding element type (key, value) (line 208)
        str_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 16), 'str', 'number of solutions')
        
        # Call to len(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'self' (line 210)
        self_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 43), 'self', False)
        # Obtaining the member 'solutions' of a type (line 210)
        solutions_758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 43), self_757, 'solutions')
        # Processing the call keyword arguments (line 210)
        kwargs_759 = {}
        # Getting the type of 'len' (line 210)
        len_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 39), 'len', False)
        # Calling len(args, kwargs) (line 210)
        len_call_result_760 = invoke(stypy.reporting.localization.Localization(__file__, 210, 39), len_756, *[solutions_758], **kwargs_759)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), dict_734, (str_755, len_call_result_760))
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', dict_734)
        
        # ################# End of 'stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stats' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_761)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stats'
        return stypy_return_type_761


# Assigning a type to the variable 'board' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'board', board)

# Assigning a Name to a Name (line 50):
# Getting the type of 'True' (line 50)
True_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'True')
# Getting the type of 'board'
board_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'board')
# Setting the type of the member 'notifyOnCompletion' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), board_763, 'notifyOnCompletion', True_762)

# Assigning a Name to a Name (line 51):
# Getting the type of 'False' (line 51)
False_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'False')
# Getting the type of 'board'
board_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'board')
# Setting the type of the member 'completeSearch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), board_765, 'completeSearch', False_764)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 213, 0, False)
    
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

    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to board(...): (line 214)
    # Processing the call keyword arguments (line 214)
    kwargs_767 = {}
    # Getting the type of 'board' (line 214)
    board_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 13), 'board', False)
    # Calling board(args, kwargs) (line 214)
    board_call_result_768 = invoke(stypy.reporting.localization.Localization(__file__, 214, 13), board_766, *[], **kwargs_767)
    
    # Assigning a type to the variable 'puzzle' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'puzzle', board_call_result_768)
    
    # Call to fread(...): (line 215)
    # Processing the call arguments (line 215)
    
    # Call to Relative(...): (line 215)
    # Processing the call arguments (line 215)
    str_772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 26), 'str', 'testdata/b6.pz')
    # Processing the call keyword arguments (line 215)
    kwargs_773 = {}
    # Getting the type of 'Relative' (line 215)
    Relative_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'Relative', False)
    # Calling Relative(args, kwargs) (line 215)
    Relative_call_result_774 = invoke(stypy.reporting.localization.Localization(__file__, 215, 17), Relative_771, *[str_772], **kwargs_773)
    
    # Processing the call keyword arguments (line 215)
    kwargs_775 = {}
    # Getting the type of 'puzzle' (line 215)
    puzzle_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'puzzle', False)
    # Obtaining the member 'fread' of a type (line 215)
    fread_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 4), puzzle_769, 'fread')
    # Calling fread(args, kwargs) (line 215)
    fread_call_result_776 = invoke(stypy.reporting.localization.Localization(__file__, 215, 4), fread_770, *[Relative_call_result_774], **kwargs_775)
    
    
    # Call to solve(...): (line 217)
    # Processing the call keyword arguments (line 217)
    kwargs_779 = {}
    # Getting the type of 'puzzle' (line 217)
    puzzle_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'puzzle', False)
    # Obtaining the member 'solve' of a type (line 217)
    solve_778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 4), puzzle_777, 'solve')
    # Calling solve(args, kwargs) (line 217)
    solve_call_result_780 = invoke(stypy.reporting.localization.Localization(__file__, 217, 4), solve_778, *[], **kwargs_779)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_781)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_781

# Assigning a type to the variable 'main' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 220, 0, False)
    
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

    
    # Call to main(...): (line 221)
    # Processing the call keyword arguments (line 221)
    kwargs_783 = {}
    # Getting the type of 'main' (line 221)
    main_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'main', False)
    # Calling main(args, kwargs) (line 221)
    main_call_result_784 = invoke(stypy.reporting.localization.Localization(__file__, 221, 4), main_782, *[], **kwargs_783)
    
    # Getting the type of 'True' (line 222)
    True_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type', True_785)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 220)
    stypy_return_type_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_786)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_786

# Assigning a type to the variable 'run' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'run', run)

# Call to run(...): (line 225)
# Processing the call keyword arguments (line 225)
kwargs_788 = {}
# Getting the type of 'run' (line 225)
run_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'run', False)
# Calling run(args, kwargs) (line 225)
run_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 225, 0), run_787, *[], **kwargs_788)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

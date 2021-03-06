
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # (c) Jack Ha
2: # --- jack.ha@gmail.com
3: #
4: # sudoku solver
5: 
6: def validMove(puzzle, x, y, number):
7:     # see if the number is in any row, column or his own 3x3 square
8:     blnOK = True
9:     px = x / 3
10:     py = y / 3
11:     if puzzle[x][y] != 0:
12:         blnOK = False
13:     if blnOK:
14:         for i in range(9):
15:             if puzzle[i][y] == number:
16:                 blnOK = False
17:     if blnOK:
18:         for j in range(9):
19:             if puzzle[x][j] == number:
20:                 blnOK = False
21:     if blnOK:
22:         for i in range(3):
23:             for j in range(3):
24:                 if puzzle[px * 3 + i][py * 3 + j] == number:
25:                     blnOK = False
26:     return blnOK
27: 
28: 
29: def findallMoves(puzzle, x, y):
30:     returnList = []
31:     for n in range(1, 10):
32:         if validMove(puzzle, x, y, n):
33:             returnList.append(n)
34:     return returnList
35: 
36: 
37: def solvePuzzleStep(puzzle):
38:     isChanged = False
39:     for y in range(9):
40:         for x in range(9):
41:             if puzzle[x][y] == 0:
42:                 allMoves = findallMoves(puzzle, x, y)
43:                 if len(allMoves) == 1:
44:                     puzzle[x][y] = allMoves[0]
45:                     isChanged = True
46:     return isChanged
47: 
48: 
49: # try to solve as much as possible without lookahead
50: def solvePuzzleSimple(puzzle):
51:     iterationCount = 0
52:     while solvePuzzleStep(puzzle) == True:
53:         iterationCount += 1
54: 
55: 
56: hashtable = {}
57: iterations = 0
58: 
59: 
60: def calc_hash(puzzle):
61:     hashcode = 0
62:     for c in range(9):
63:         hashcode = hashcode * 17 + hash(tuple(puzzle[c]))
64:     return hashcode
65: 
66: 
67: def hash_add(puzzle):
68:     hashtable[calc_hash(puzzle)] = 1
69: 
70: 
71: def hash_lookup(puzzle):
72:     return hashtable.has_key(calc_hash(puzzle))
73: 
74: 
75: # solve with lookahead
76: # unit is 3x3, (i,j) is coords of unit. l is the list of all todo's
77: def perm(puzzle, i, j, l, u):
78:     global iterations
79:     iterations += 1
80:     if (u == []) and (l == []):
81:         ##                print "Solved!"
82:         printpuzzle(puzzle)
83:         ##                print "iterations: ", iterations
84:         return True
85:     else:
86:         if l == []:
87:             # here we have all permutations for one unit
88: 
89:             # some simple moves
90:             puzzlebackup = []
91:             for c in range(9):
92:                 puzzlebackup.append(tuple(puzzle[c]))
93:             solvePuzzleSimple(puzzle)
94: 
95:             # next unit to fill
96:             for c in range(len(u)):
97:                 if not hash_lookup(puzzle):
98:                     inew, jnew = u.pop(c)
99:                     l = genMoveList(puzzle, inew, jnew)
100:                     # printpuzzle(puzzle)
101:                     # print "iterations: ", iterations
102:                     if perm(puzzle, inew, jnew, l, u):
103:                         return True
104:                     else:
105:                         hash_add(puzzle)
106:                     u.insert(c, (inew, jnew))
107: 
108:             # undo simple moves
109:             for y in range(9):
110:                 for x in range(9):
111:                     puzzle[x][y] = puzzlebackup[x][y]
112:             hash_add(puzzle)
113:             return False
114:         else:
115:             # try all possibilities of one unit
116:             ii = i * 3
117:             jj = j * 3
118:             for m in range(len(l)):
119:                 # find first empty
120:                 for y in range(3):
121:                     for x in range(3):
122:                         if validMove(puzzle, x + ii, y + jj, l[m]):
123:                             puzzle[x + ii][y + jj] = l[m]
124:                             backup = l.pop(m)
125:                             if (perm(puzzle, i, j, l, u)):
126:                                 return True
127:                             else:
128:                                 hash_add(puzzle)
129:                             l.insert(m, backup)
130:                             puzzle[x + ii][y + jj] = 0
131:             return False
132: 
133: 
134: # gen move list for unit (i,j)
135: def genMoveList(puzzle, i, j):
136:     l = range(1, 10)
137:     for y in range(3):
138:         for x in range(3):
139:             p = puzzle[i * 3 + x][j * 3 + y]
140:             if p != 0:
141:                 l.remove(p)
142:     return l
143: 
144: 
145: def printpuzzle(puzzle):
146:     for x in range(9):
147:         s = ' '
148:         for y in range(9):
149:             p = puzzle[x][y]
150:             if p == 0:
151:                 s += '.'
152:             else:
153:                 s += str(puzzle[x][y])
154:             s += ' '
155: 
156: 
157: ##                print s
158: 
159: def main():
160:     puzzle = [[0, 9, 3, 0, 8, 0, 4, 0, 0],
161:               [0, 4, 0, 0, 3, 0, 0, 0, 0],
162:               [6, 0, 0, 0, 0, 9, 2, 0, 5],
163:               [3, 0, 0, 0, 0, 0, 0, 9, 0],
164:               [0, 2, 7, 0, 0, 0, 5, 1, 0],
165:               [0, 8, 0, 0, 0, 0, 0, 0, 4],
166:               [7, 0, 1, 6, 0, 0, 0, 0, 2],
167:               [0, 0, 0, 0, 7, 0, 0, 6, 0],
168:               [0, 0, 4, 0, 1, 0, 8, 5, 0]]
169: 
170:     # create todo unit(each 3x3) list (this is also the order that they will be tried!)
171:     u = []
172:     lcount = []
173:     for y in range(3):
174:         for x in range(3):
175:             u.append((x, y))
176:             lcount.append(len(genMoveList(puzzle, x, y)))
177: 
178:     # sort
179:     for j in range(0, 9):
180:         for i in range(j, 9):
181:             if i != j:
182:                 if lcount[i] < lcount[j]:
183:                     u[i], u[j] = u[j], u[i]
184:                     lcount[i], lcount[j] = lcount[j], lcount[i]
185: 
186:     l = genMoveList(puzzle, 0, 0)
187:     perm(puzzle, 0, 0, l, u)
188: 
189: 
190: def run():
191:     iterations = 0
192:     for x in range(30):
193:         main()
194:     return True
195: 
196: 
197: run()
198: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def validMove(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'validMove'
    module_type_store = module_type_store.open_function_context('validMove', 6, 0, False)
    
    # Passed parameters checking function
    validMove.stypy_localization = localization
    validMove.stypy_type_of_self = None
    validMove.stypy_type_store = module_type_store
    validMove.stypy_function_name = 'validMove'
    validMove.stypy_param_names_list = ['puzzle', 'x', 'y', 'number']
    validMove.stypy_varargs_param_name = None
    validMove.stypy_kwargs_param_name = None
    validMove.stypy_call_defaults = defaults
    validMove.stypy_call_varargs = varargs
    validMove.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'validMove', ['puzzle', 'x', 'y', 'number'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'validMove', localization, ['puzzle', 'x', 'y', 'number'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'validMove(...)' code ##################

    
    # Assigning a Name to a Name (line 8):
    
    # Assigning a Name to a Name (line 8):
    # Getting the type of 'True' (line 8)
    True_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'True')
    # Assigning a type to the variable 'blnOK' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'blnOK', True_8)
    
    # Assigning a BinOp to a Name (line 9):
    
    # Assigning a BinOp to a Name (line 9):
    # Getting the type of 'x' (line 9)
    x_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 9), 'x')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'int')
    # Applying the binary operator 'div' (line 9)
    result_div_11 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 9), 'div', x_9, int_10)
    
    # Assigning a type to the variable 'px' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'px', result_div_11)
    
    # Assigning a BinOp to a Name (line 10):
    
    # Assigning a BinOp to a Name (line 10):
    # Getting the type of 'y' (line 10)
    y_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'y')
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'int')
    # Applying the binary operator 'div' (line 10)
    result_div_14 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), 'div', y_12, int_13)
    
    # Assigning a type to the variable 'py' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'py', result_div_14)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'y' (line 11)
    y_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'y')
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 11)
    x_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'x')
    # Getting the type of 'puzzle' (line 11)
    puzzle_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'puzzle')
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), puzzle_17, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), getitem___18, x_16)
    
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), subscript_call_result_19, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), getitem___20, y_15)
    
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
    # Applying the binary operator '!=' (line 11)
    result_ne_23 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 7), '!=', subscript_call_result_21, int_22)
    
    # Testing if the type of an if condition is none (line 11)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 11, 4), result_ne_23):
        pass
    else:
        
        # Testing the type of an if condition (line 11)
        if_condition_24 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 4), result_ne_23)
        # Assigning a type to the variable 'if_condition_24' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'if_condition_24', if_condition_24)
        # SSA begins for if statement (line 11)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 12):
        
        # Assigning a Name to a Name (line 12):
        # Getting the type of 'False' (line 12)
        False_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'False')
        # Assigning a type to the variable 'blnOK' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'blnOK', False_25)
        # SSA join for if statement (line 11)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'blnOK' (line 13)
    blnOK_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 7), 'blnOK')
    # Testing if the type of an if condition is none (line 13)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 13, 4), blnOK_26):
        pass
    else:
        
        # Testing the type of an if condition (line 13)
        if_condition_27 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 4), blnOK_26)
        # Assigning a type to the variable 'if_condition_27' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'if_condition_27', if_condition_27)
        # SSA begins for if statement (line 13)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to range(...): (line 14)
        # Processing the call arguments (line 14)
        int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'int')
        # Processing the call keyword arguments (line 14)
        kwargs_30 = {}
        # Getting the type of 'range' (line 14)
        range_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'range', False)
        # Calling range(args, kwargs) (line 14)
        range_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 14, 17), range_28, *[int_29], **kwargs_30)
        
        # Assigning a type to the variable 'range_call_result_31' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'range_call_result_31', range_call_result_31)
        # Testing if the for loop is going to be iterated (line 14)
        # Testing the type of a for loop iterable (line 14)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 14, 8), range_call_result_31)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 14, 8), range_call_result_31):
            # Getting the type of the for loop variable (line 14)
            for_loop_var_32 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 14, 8), range_call_result_31)
            # Assigning a type to the variable 'i' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'i', for_loop_var_32)
            # SSA begins for a for statement (line 14)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 15)
            y_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'y')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 15)
            i_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'i')
            # Getting the type of 'puzzle' (line 15)
            puzzle_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'puzzle')
            # Obtaining the member '__getitem__' of a type (line 15)
            getitem___36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 15), puzzle_35, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 15)
            subscript_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 15, 15), getitem___36, i_34)
            
            # Obtaining the member '__getitem__' of a type (line 15)
            getitem___38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 15), subscript_call_result_37, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 15)
            subscript_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 15, 15), getitem___38, y_33)
            
            # Getting the type of 'number' (line 15)
            number_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'number')
            # Applying the binary operator '==' (line 15)
            result_eq_41 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 15), '==', subscript_call_result_39, number_40)
            
            # Testing if the type of an if condition is none (line 15)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 15, 12), result_eq_41):
                pass
            else:
                
                # Testing the type of an if condition (line 15)
                if_condition_42 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 12), result_eq_41)
                # Assigning a type to the variable 'if_condition_42' (line 15)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'if_condition_42', if_condition_42)
                # SSA begins for if statement (line 15)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 16):
                
                # Assigning a Name to a Name (line 16):
                # Getting the type of 'False' (line 16)
                False_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'False')
                # Assigning a type to the variable 'blnOK' (line 16)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'blnOK', False_43)
                # SSA join for if statement (line 15)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 13)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'blnOK' (line 17)
    blnOK_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'blnOK')
    # Testing if the type of an if condition is none (line 17)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 17, 4), blnOK_44):
        pass
    else:
        
        # Testing the type of an if condition (line 17)
        if_condition_45 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), blnOK_44)
        # Assigning a type to the variable 'if_condition_45' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_45', if_condition_45)
        # SSA begins for if statement (line 17)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to range(...): (line 18)
        # Processing the call arguments (line 18)
        int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'int')
        # Processing the call keyword arguments (line 18)
        kwargs_48 = {}
        # Getting the type of 'range' (line 18)
        range_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'range', False)
        # Calling range(args, kwargs) (line 18)
        range_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), range_46, *[int_47], **kwargs_48)
        
        # Assigning a type to the variable 'range_call_result_49' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'range_call_result_49', range_call_result_49)
        # Testing if the for loop is going to be iterated (line 18)
        # Testing the type of a for loop iterable (line 18)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 8), range_call_result_49)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 18, 8), range_call_result_49):
            # Getting the type of the for loop variable (line 18)
            for_loop_var_50 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 8), range_call_result_49)
            # Assigning a type to the variable 'j' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'j', for_loop_var_50)
            # SSA begins for a for statement (line 18)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 19)
            j_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'j')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 19)
            x_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'x')
            # Getting the type of 'puzzle' (line 19)
            puzzle_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'puzzle')
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), puzzle_53, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), getitem___54, x_52)
            
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), subscript_call_result_55, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), getitem___56, j_51)
            
            # Getting the type of 'number' (line 19)
            number_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'number')
            # Applying the binary operator '==' (line 19)
            result_eq_59 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), '==', subscript_call_result_57, number_58)
            
            # Testing if the type of an if condition is none (line 19)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 19, 12), result_eq_59):
                pass
            else:
                
                # Testing the type of an if condition (line 19)
                if_condition_60 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 12), result_eq_59)
                # Assigning a type to the variable 'if_condition_60' (line 19)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'if_condition_60', if_condition_60)
                # SSA begins for if statement (line 19)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 20):
                
                # Assigning a Name to a Name (line 20):
                # Getting the type of 'False' (line 20)
                False_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'False')
                # Assigning a type to the variable 'blnOK' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'blnOK', False_61)
                # SSA join for if statement (line 19)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 17)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'blnOK' (line 21)
    blnOK_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'blnOK')
    # Testing if the type of an if condition is none (line 21)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 21, 4), blnOK_62):
        pass
    else:
        
        # Testing the type of an if condition (line 21)
        if_condition_63 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 4), blnOK_62)
        # Assigning a type to the variable 'if_condition_63' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'if_condition_63', if_condition_63)
        # SSA begins for if statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to range(...): (line 22)
        # Processing the call arguments (line 22)
        int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
        # Processing the call keyword arguments (line 22)
        kwargs_66 = {}
        # Getting the type of 'range' (line 22)
        range_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'range', False)
        # Calling range(args, kwargs) (line 22)
        range_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), range_64, *[int_65], **kwargs_66)
        
        # Assigning a type to the variable 'range_call_result_67' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'range_call_result_67', range_call_result_67)
        # Testing if the for loop is going to be iterated (line 22)
        # Testing the type of a for loop iterable (line 22)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 22, 8), range_call_result_67)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 22, 8), range_call_result_67):
            # Getting the type of the for loop variable (line 22)
            for_loop_var_68 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 22, 8), range_call_result_67)
            # Assigning a type to the variable 'i' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'i', for_loop_var_68)
            # SSA begins for a for statement (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 23)
            # Processing the call arguments (line 23)
            int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'int')
            # Processing the call keyword arguments (line 23)
            kwargs_71 = {}
            # Getting the type of 'range' (line 23)
            range_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'range', False)
            # Calling range(args, kwargs) (line 23)
            range_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 23, 21), range_69, *[int_70], **kwargs_71)
            
            # Assigning a type to the variable 'range_call_result_72' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'range_call_result_72', range_call_result_72)
            # Testing if the for loop is going to be iterated (line 23)
            # Testing the type of a for loop iterable (line 23)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 12), range_call_result_72)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 23, 12), range_call_result_72):
                # Getting the type of the for loop variable (line 23)
                for_loop_var_73 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 12), range_call_result_72)
                # Assigning a type to the variable 'j' (line 23)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'j', for_loop_var_73)
                # SSA begins for a for statement (line 23)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'py' (line 24)
                py_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 38), 'py')
                int_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'int')
                # Applying the binary operator '*' (line 24)
                result_mul_76 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 38), '*', py_74, int_75)
                
                # Getting the type of 'j' (line 24)
                j_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 47), 'j')
                # Applying the binary operator '+' (line 24)
                result_add_78 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 38), '+', result_mul_76, j_77)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'px' (line 24)
                px_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'px')
                int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'int')
                # Applying the binary operator '*' (line 24)
                result_mul_81 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 26), '*', px_79, int_80)
                
                # Getting the type of 'i' (line 24)
                i_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 35), 'i')
                # Applying the binary operator '+' (line 24)
                result_add_83 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 26), '+', result_mul_81, i_82)
                
                # Getting the type of 'puzzle' (line 24)
                puzzle_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'puzzle')
                # Obtaining the member '__getitem__' of a type (line 24)
                getitem___85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 19), puzzle_84, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 24)
                subscript_call_result_86 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), getitem___85, result_add_83)
                
                # Obtaining the member '__getitem__' of a type (line 24)
                getitem___87 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 19), subscript_call_result_86, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 24)
                subscript_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), getitem___87, result_add_78)
                
                # Getting the type of 'number' (line 24)
                number_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 53), 'number')
                # Applying the binary operator '==' (line 24)
                result_eq_90 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), '==', subscript_call_result_88, number_89)
                
                # Testing if the type of an if condition is none (line 24)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 24, 16), result_eq_90):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 24)
                    if_condition_91 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 16), result_eq_90)
                    # Assigning a type to the variable 'if_condition_91' (line 24)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'if_condition_91', if_condition_91)
                    # SSA begins for if statement (line 24)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 25):
                    
                    # Assigning a Name to a Name (line 25):
                    # Getting the type of 'False' (line 25)
                    False_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'False')
                    # Assigning a type to the variable 'blnOK' (line 25)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'blnOK', False_92)
                    # SSA join for if statement (line 24)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 21)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'blnOK' (line 26)
    blnOK_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'blnOK')
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', blnOK_93)
    
    # ################# End of 'validMove(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'validMove' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_94)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'validMove'
    return stypy_return_type_94

# Assigning a type to the variable 'validMove' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'validMove', validMove)

@norecursion
def findallMoves(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'findallMoves'
    module_type_store = module_type_store.open_function_context('findallMoves', 29, 0, False)
    
    # Passed parameters checking function
    findallMoves.stypy_localization = localization
    findallMoves.stypy_type_of_self = None
    findallMoves.stypy_type_store = module_type_store
    findallMoves.stypy_function_name = 'findallMoves'
    findallMoves.stypy_param_names_list = ['puzzle', 'x', 'y']
    findallMoves.stypy_varargs_param_name = None
    findallMoves.stypy_kwargs_param_name = None
    findallMoves.stypy_call_defaults = defaults
    findallMoves.stypy_call_varargs = varargs
    findallMoves.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findallMoves', ['puzzle', 'x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findallMoves', localization, ['puzzle', 'x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findallMoves(...)' code ##################

    
    # Assigning a List to a Name (line 30):
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    
    # Assigning a type to the variable 'returnList' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'returnList', list_95)
    
    
    # Call to range(...): (line 31)
    # Processing the call arguments (line 31)
    int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'int')
    int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 22), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_99 = {}
    # Getting the type of 'range' (line 31)
    range_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'range', False)
    # Calling range(args, kwargs) (line 31)
    range_call_result_100 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), range_96, *[int_97, int_98], **kwargs_99)
    
    # Assigning a type to the variable 'range_call_result_100' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'range_call_result_100', range_call_result_100)
    # Testing if the for loop is going to be iterated (line 31)
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), range_call_result_100)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 4), range_call_result_100):
        # Getting the type of the for loop variable (line 31)
        for_loop_var_101 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), range_call_result_100)
        # Assigning a type to the variable 'n' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'n', for_loop_var_101)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to validMove(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'puzzle' (line 32)
        puzzle_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'puzzle', False)
        # Getting the type of 'x' (line 32)
        x_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'x', False)
        # Getting the type of 'y' (line 32)
        y_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'y', False)
        # Getting the type of 'n' (line 32)
        n_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'n', False)
        # Processing the call keyword arguments (line 32)
        kwargs_107 = {}
        # Getting the type of 'validMove' (line 32)
        validMove_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'validMove', False)
        # Calling validMove(args, kwargs) (line 32)
        validMove_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), validMove_102, *[puzzle_103, x_104, y_105, n_106], **kwargs_107)
        
        # Testing if the type of an if condition is none (line 32)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 8), validMove_call_result_108):
            pass
        else:
            
            # Testing the type of an if condition (line 32)
            if_condition_109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), validMove_call_result_108)
            # Assigning a type to the variable 'if_condition_109' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_109', if_condition_109)
            # SSA begins for if statement (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 33)
            # Processing the call arguments (line 33)
            # Getting the type of 'n' (line 33)
            n_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 30), 'n', False)
            # Processing the call keyword arguments (line 33)
            kwargs_113 = {}
            # Getting the type of 'returnList' (line 33)
            returnList_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'returnList', False)
            # Obtaining the member 'append' of a type (line 33)
            append_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), returnList_110, 'append')
            # Calling append(args, kwargs) (line 33)
            append_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), append_111, *[n_112], **kwargs_113)
            
            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'returnList' (line 34)
    returnList_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'returnList')
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', returnList_115)
    
    # ################# End of 'findallMoves(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findallMoves' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findallMoves'
    return stypy_return_type_116

# Assigning a type to the variable 'findallMoves' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'findallMoves', findallMoves)

@norecursion
def solvePuzzleStep(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solvePuzzleStep'
    module_type_store = module_type_store.open_function_context('solvePuzzleStep', 37, 0, False)
    
    # Passed parameters checking function
    solvePuzzleStep.stypy_localization = localization
    solvePuzzleStep.stypy_type_of_self = None
    solvePuzzleStep.stypy_type_store = module_type_store
    solvePuzzleStep.stypy_function_name = 'solvePuzzleStep'
    solvePuzzleStep.stypy_param_names_list = ['puzzle']
    solvePuzzleStep.stypy_varargs_param_name = None
    solvePuzzleStep.stypy_kwargs_param_name = None
    solvePuzzleStep.stypy_call_defaults = defaults
    solvePuzzleStep.stypy_call_varargs = varargs
    solvePuzzleStep.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solvePuzzleStep', ['puzzle'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solvePuzzleStep', localization, ['puzzle'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solvePuzzleStep(...)' code ##################

    
    # Assigning a Name to a Name (line 38):
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'False' (line 38)
    False_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'False')
    # Assigning a type to the variable 'isChanged' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'isChanged', False_117)
    
    
    # Call to range(...): (line 39)
    # Processing the call arguments (line 39)
    int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'int')
    # Processing the call keyword arguments (line 39)
    kwargs_120 = {}
    # Getting the type of 'range' (line 39)
    range_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'range', False)
    # Calling range(args, kwargs) (line 39)
    range_call_result_121 = invoke(stypy.reporting.localization.Localization(__file__, 39, 13), range_118, *[int_119], **kwargs_120)
    
    # Assigning a type to the variable 'range_call_result_121' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'range_call_result_121', range_call_result_121)
    # Testing if the for loop is going to be iterated (line 39)
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 4), range_call_result_121)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 39, 4), range_call_result_121):
        # Getting the type of the for loop variable (line 39)
        for_loop_var_122 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 4), range_call_result_121)
        # Assigning a type to the variable 'y' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'y', for_loop_var_122)
        # SSA begins for a for statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 40)
        # Processing the call arguments (line 40)
        int_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'int')
        # Processing the call keyword arguments (line 40)
        kwargs_125 = {}
        # Getting the type of 'range' (line 40)
        range_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'range', False)
        # Calling range(args, kwargs) (line 40)
        range_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), range_123, *[int_124], **kwargs_125)
        
        # Assigning a type to the variable 'range_call_result_126' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'range_call_result_126', range_call_result_126)
        # Testing if the for loop is going to be iterated (line 40)
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), range_call_result_126)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 8), range_call_result_126):
            # Getting the type of the for loop variable (line 40)
            for_loop_var_127 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), range_call_result_126)
            # Assigning a type to the variable 'x' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'x', for_loop_var_127)
            # SSA begins for a for statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 41)
            y_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'y')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 41)
            x_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'x')
            # Getting the type of 'puzzle' (line 41)
            puzzle_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'puzzle')
            # Obtaining the member '__getitem__' of a type (line 41)
            getitem___131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), puzzle_130, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 41)
            subscript_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), getitem___131, x_129)
            
            # Obtaining the member '__getitem__' of a type (line 41)
            getitem___133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), subscript_call_result_132, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 41)
            subscript_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), getitem___133, y_128)
            
            int_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'int')
            # Applying the binary operator '==' (line 41)
            result_eq_136 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '==', subscript_call_result_134, int_135)
            
            # Testing if the type of an if condition is none (line 41)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 12), result_eq_136):
                pass
            else:
                
                # Testing the type of an if condition (line 41)
                if_condition_137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), result_eq_136)
                # Assigning a type to the variable 'if_condition_137' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_137', if_condition_137)
                # SSA begins for if statement (line 41)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 42):
                
                # Assigning a Call to a Name (line 42):
                
                # Call to findallMoves(...): (line 42)
                # Processing the call arguments (line 42)
                # Getting the type of 'puzzle' (line 42)
                puzzle_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'puzzle', False)
                # Getting the type of 'x' (line 42)
                x_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 48), 'x', False)
                # Getting the type of 'y' (line 42)
                y_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 51), 'y', False)
                # Processing the call keyword arguments (line 42)
                kwargs_142 = {}
                # Getting the type of 'findallMoves' (line 42)
                findallMoves_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'findallMoves', False)
                # Calling findallMoves(args, kwargs) (line 42)
                findallMoves_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), findallMoves_138, *[puzzle_139, x_140, y_141], **kwargs_142)
                
                # Assigning a type to the variable 'allMoves' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'allMoves', findallMoves_call_result_143)
                
                
                # Call to len(...): (line 43)
                # Processing the call arguments (line 43)
                # Getting the type of 'allMoves' (line 43)
                allMoves_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'allMoves', False)
                # Processing the call keyword arguments (line 43)
                kwargs_146 = {}
                # Getting the type of 'len' (line 43)
                len_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'len', False)
                # Calling len(args, kwargs) (line 43)
                len_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 43, 19), len_144, *[allMoves_145], **kwargs_146)
                
                int_148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'int')
                # Applying the binary operator '==' (line 43)
                result_eq_149 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 19), '==', len_call_result_147, int_148)
                
                # Testing if the type of an if condition is none (line 43)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 43, 16), result_eq_149):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 43)
                    if_condition_150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 16), result_eq_149)
                    # Assigning a type to the variable 'if_condition_150' (line 43)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'if_condition_150', if_condition_150)
                    # SSA begins for if statement (line 43)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Subscript (line 44):
                    
                    # Assigning a Subscript to a Subscript (line 44):
                    
                    # Obtaining the type of the subscript
                    int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 44), 'int')
                    # Getting the type of 'allMoves' (line 44)
                    allMoves_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'allMoves')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 35), allMoves_152, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 44, 35), getitem___153, int_151)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'x' (line 44)
                    x_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'x')
                    # Getting the type of 'puzzle' (line 44)
                    puzzle_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'puzzle')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), puzzle_156, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), getitem___157, x_155)
                    
                    # Getting the type of 'y' (line 44)
                    y_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'y')
                    # Storing an element on a container (line 44)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), subscript_call_result_158, (y_159, subscript_call_result_154))
                    
                    # Assigning a Name to a Name (line 45):
                    
                    # Assigning a Name to a Name (line 45):
                    # Getting the type of 'True' (line 45)
                    True_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 32), 'True')
                    # Assigning a type to the variable 'isChanged' (line 45)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'isChanged', True_160)
                    # SSA join for if statement (line 43)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 41)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'isChanged' (line 46)
    isChanged_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'isChanged')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', isChanged_161)
    
    # ################# End of 'solvePuzzleStep(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solvePuzzleStep' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solvePuzzleStep'
    return stypy_return_type_162

# Assigning a type to the variable 'solvePuzzleStep' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'solvePuzzleStep', solvePuzzleStep)

@norecursion
def solvePuzzleSimple(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solvePuzzleSimple'
    module_type_store = module_type_store.open_function_context('solvePuzzleSimple', 50, 0, False)
    
    # Passed parameters checking function
    solvePuzzleSimple.stypy_localization = localization
    solvePuzzleSimple.stypy_type_of_self = None
    solvePuzzleSimple.stypy_type_store = module_type_store
    solvePuzzleSimple.stypy_function_name = 'solvePuzzleSimple'
    solvePuzzleSimple.stypy_param_names_list = ['puzzle']
    solvePuzzleSimple.stypy_varargs_param_name = None
    solvePuzzleSimple.stypy_kwargs_param_name = None
    solvePuzzleSimple.stypy_call_defaults = defaults
    solvePuzzleSimple.stypy_call_varargs = varargs
    solvePuzzleSimple.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solvePuzzleSimple', ['puzzle'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solvePuzzleSimple', localization, ['puzzle'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solvePuzzleSimple(...)' code ##################

    
    # Assigning a Num to a Name (line 51):
    
    # Assigning a Num to a Name (line 51):
    int_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'int')
    # Assigning a type to the variable 'iterationCount' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'iterationCount', int_163)
    
    
    
    # Call to solvePuzzleStep(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'puzzle' (line 52)
    puzzle_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'puzzle', False)
    # Processing the call keyword arguments (line 52)
    kwargs_166 = {}
    # Getting the type of 'solvePuzzleStep' (line 52)
    solvePuzzleStep_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'solvePuzzleStep', False)
    # Calling solvePuzzleStep(args, kwargs) (line 52)
    solvePuzzleStep_call_result_167 = invoke(stypy.reporting.localization.Localization(__file__, 52, 10), solvePuzzleStep_164, *[puzzle_165], **kwargs_166)
    
    # Getting the type of 'True' (line 52)
    True_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 37), 'True')
    # Applying the binary operator '==' (line 52)
    result_eq_169 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 10), '==', solvePuzzleStep_call_result_167, True_168)
    
    # Assigning a type to the variable 'result_eq_169' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'result_eq_169', result_eq_169)
    # Testing if the while is going to be iterated (line 52)
    # Testing the type of an if condition (line 52)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), result_eq_169)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 52, 4), result_eq_169):
        # SSA begins for while statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'iterationCount' (line 53)
        iterationCount_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'iterationCount')
        int_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'int')
        # Applying the binary operator '+=' (line 53)
        result_iadd_172 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 8), '+=', iterationCount_170, int_171)
        # Assigning a type to the variable 'iterationCount' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'iterationCount', result_iadd_172)
        
        # SSA join for while statement (line 52)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'solvePuzzleSimple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solvePuzzleSimple' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solvePuzzleSimple'
    return stypy_return_type_173

# Assigning a type to the variable 'solvePuzzleSimple' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'solvePuzzleSimple', solvePuzzleSimple)

# Assigning a Dict to a Name (line 56):

# Assigning a Dict to a Name (line 56):

# Obtaining an instance of the builtin type 'dict' (line 56)
dict_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 56)

# Assigning a type to the variable 'hashtable' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'hashtable', dict_174)

# Assigning a Num to a Name (line 57):

# Assigning a Num to a Name (line 57):
int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'int')
# Assigning a type to the variable 'iterations' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'iterations', int_175)

@norecursion
def calc_hash(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'calc_hash'
    module_type_store = module_type_store.open_function_context('calc_hash', 60, 0, False)
    
    # Passed parameters checking function
    calc_hash.stypy_localization = localization
    calc_hash.stypy_type_of_self = None
    calc_hash.stypy_type_store = module_type_store
    calc_hash.stypy_function_name = 'calc_hash'
    calc_hash.stypy_param_names_list = ['puzzle']
    calc_hash.stypy_varargs_param_name = None
    calc_hash.stypy_kwargs_param_name = None
    calc_hash.stypy_call_defaults = defaults
    calc_hash.stypy_call_varargs = varargs
    calc_hash.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'calc_hash', ['puzzle'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'calc_hash', localization, ['puzzle'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'calc_hash(...)' code ##################

    
    # Assigning a Num to a Name (line 61):
    
    # Assigning a Num to a Name (line 61):
    int_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
    # Assigning a type to the variable 'hashcode' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'hashcode', int_176)
    
    
    # Call to range(...): (line 62)
    # Processing the call arguments (line 62)
    int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'int')
    # Processing the call keyword arguments (line 62)
    kwargs_179 = {}
    # Getting the type of 'range' (line 62)
    range_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'range', False)
    # Calling range(args, kwargs) (line 62)
    range_call_result_180 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), range_177, *[int_178], **kwargs_179)
    
    # Assigning a type to the variable 'range_call_result_180' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'range_call_result_180', range_call_result_180)
    # Testing if the for loop is going to be iterated (line 62)
    # Testing the type of a for loop iterable (line 62)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 4), range_call_result_180)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 62, 4), range_call_result_180):
        # Getting the type of the for loop variable (line 62)
        for_loop_var_181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 4), range_call_result_180)
        # Assigning a type to the variable 'c' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'c', for_loop_var_181)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 63):
        
        # Assigning a BinOp to a Name (line 63):
        # Getting the type of 'hashcode' (line 63)
        hashcode_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'hashcode')
        int_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'int')
        # Applying the binary operator '*' (line 63)
        result_mul_184 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 19), '*', hashcode_182, int_183)
        
        
        # Call to hash(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to tuple(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 63)
        c_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 53), 'c', False)
        # Getting the type of 'puzzle' (line 63)
        puzzle_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 46), 'puzzle', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 46), puzzle_188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 63, 46), getitem___189, c_187)
        
        # Processing the call keyword arguments (line 63)
        kwargs_191 = {}
        # Getting the type of 'tuple' (line 63)
        tuple_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'tuple', False)
        # Calling tuple(args, kwargs) (line 63)
        tuple_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 63, 40), tuple_186, *[subscript_call_result_190], **kwargs_191)
        
        # Processing the call keyword arguments (line 63)
        kwargs_193 = {}
        # Getting the type of 'hash' (line 63)
        hash_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 35), 'hash', False)
        # Calling hash(args, kwargs) (line 63)
        hash_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 63, 35), hash_185, *[tuple_call_result_192], **kwargs_193)
        
        # Applying the binary operator '+' (line 63)
        result_add_195 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 19), '+', result_mul_184, hash_call_result_194)
        
        # Assigning a type to the variable 'hashcode' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'hashcode', result_add_195)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'hashcode' (line 64)
    hashcode_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'hashcode')
    # Assigning a type to the variable 'stypy_return_type' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type', hashcode_196)
    
    # ################# End of 'calc_hash(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'calc_hash' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_197)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'calc_hash'
    return stypy_return_type_197

# Assigning a type to the variable 'calc_hash' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'calc_hash', calc_hash)

@norecursion
def hash_add(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hash_add'
    module_type_store = module_type_store.open_function_context('hash_add', 67, 0, False)
    
    # Passed parameters checking function
    hash_add.stypy_localization = localization
    hash_add.stypy_type_of_self = None
    hash_add.stypy_type_store = module_type_store
    hash_add.stypy_function_name = 'hash_add'
    hash_add.stypy_param_names_list = ['puzzle']
    hash_add.stypy_varargs_param_name = None
    hash_add.stypy_kwargs_param_name = None
    hash_add.stypy_call_defaults = defaults
    hash_add.stypy_call_varargs = varargs
    hash_add.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hash_add', ['puzzle'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hash_add', localization, ['puzzle'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hash_add(...)' code ##################

    
    # Assigning a Num to a Subscript (line 68):
    
    # Assigning a Num to a Subscript (line 68):
    int_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 35), 'int')
    # Getting the type of 'hashtable' (line 68)
    hashtable_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'hashtable')
    
    # Call to calc_hash(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'puzzle' (line 68)
    puzzle_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'puzzle', False)
    # Processing the call keyword arguments (line 68)
    kwargs_202 = {}
    # Getting the type of 'calc_hash' (line 68)
    calc_hash_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'calc_hash', False)
    # Calling calc_hash(args, kwargs) (line 68)
    calc_hash_call_result_203 = invoke(stypy.reporting.localization.Localization(__file__, 68, 14), calc_hash_200, *[puzzle_201], **kwargs_202)
    
    # Storing an element on a container (line 68)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), hashtable_199, (calc_hash_call_result_203, int_198))
    
    # ################# End of 'hash_add(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hash_add' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hash_add'
    return stypy_return_type_204

# Assigning a type to the variable 'hash_add' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'hash_add', hash_add)

@norecursion
def hash_lookup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hash_lookup'
    module_type_store = module_type_store.open_function_context('hash_lookup', 71, 0, False)
    
    # Passed parameters checking function
    hash_lookup.stypy_localization = localization
    hash_lookup.stypy_type_of_self = None
    hash_lookup.stypy_type_store = module_type_store
    hash_lookup.stypy_function_name = 'hash_lookup'
    hash_lookup.stypy_param_names_list = ['puzzle']
    hash_lookup.stypy_varargs_param_name = None
    hash_lookup.stypy_kwargs_param_name = None
    hash_lookup.stypy_call_defaults = defaults
    hash_lookup.stypy_call_varargs = varargs
    hash_lookup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hash_lookup', ['puzzle'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hash_lookup', localization, ['puzzle'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hash_lookup(...)' code ##################

    
    # Call to has_key(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to calc_hash(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'puzzle' (line 72)
    puzzle_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 39), 'puzzle', False)
    # Processing the call keyword arguments (line 72)
    kwargs_209 = {}
    # Getting the type of 'calc_hash' (line 72)
    calc_hash_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'calc_hash', False)
    # Calling calc_hash(args, kwargs) (line 72)
    calc_hash_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), calc_hash_207, *[puzzle_208], **kwargs_209)
    
    # Processing the call keyword arguments (line 72)
    kwargs_211 = {}
    # Getting the type of 'hashtable' (line 72)
    hashtable_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'hashtable', False)
    # Obtaining the member 'has_key' of a type (line 72)
    has_key_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), hashtable_205, 'has_key')
    # Calling has_key(args, kwargs) (line 72)
    has_key_call_result_212 = invoke(stypy.reporting.localization.Localization(__file__, 72, 11), has_key_206, *[calc_hash_call_result_210], **kwargs_211)
    
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type', has_key_call_result_212)
    
    # ################# End of 'hash_lookup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hash_lookup' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_213)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hash_lookup'
    return stypy_return_type_213

# Assigning a type to the variable 'hash_lookup' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'hash_lookup', hash_lookup)

@norecursion
def perm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'perm'
    module_type_store = module_type_store.open_function_context('perm', 77, 0, False)
    
    # Passed parameters checking function
    perm.stypy_localization = localization
    perm.stypy_type_of_self = None
    perm.stypy_type_store = module_type_store
    perm.stypy_function_name = 'perm'
    perm.stypy_param_names_list = ['puzzle', 'i', 'j', 'l', 'u']
    perm.stypy_varargs_param_name = None
    perm.stypy_kwargs_param_name = None
    perm.stypy_call_defaults = defaults
    perm.stypy_call_varargs = varargs
    perm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'perm', ['puzzle', 'i', 'j', 'l', 'u'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'perm', localization, ['puzzle', 'i', 'j', 'l', 'u'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'perm(...)' code ##################

    # Marking variables as global (line 78)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 78, 4), 'iterations')
    
    # Getting the type of 'iterations' (line 79)
    iterations_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'iterations')
    int_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 18), 'int')
    # Applying the binary operator '+=' (line 79)
    result_iadd_216 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 4), '+=', iterations_214, int_215)
    # Assigning a type to the variable 'iterations' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'iterations', result_iadd_216)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'u' (line 80)
    u_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'u')
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    
    # Applying the binary operator '==' (line 80)
    result_eq_219 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 8), '==', u_217, list_218)
    
    
    # Getting the type of 'l' (line 80)
    l_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'l')
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    
    # Applying the binary operator '==' (line 80)
    result_eq_222 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 22), '==', l_220, list_221)
    
    # Applying the binary operator 'and' (line 80)
    result_and_keyword_223 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), 'and', result_eq_219, result_eq_222)
    
    # Testing if the type of an if condition is none (line 80)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 4), result_and_keyword_223):
        
        # Getting the type of 'l' (line 86)
        l_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'l')
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        
        # Applying the binary operator '==' (line 86)
        result_eq_232 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), '==', l_230, list_231)
        
        # Testing if the type of an if condition is none (line 86)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_232):
            
            # Assigning a BinOp to a Name (line 116):
            
            # Assigning a BinOp to a Name (line 116):
            # Getting the type of 'i' (line 116)
            i_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'i')
            int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
            # Applying the binary operator '*' (line 116)
            result_mul_337 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '*', i_335, int_336)
            
            # Assigning a type to the variable 'ii' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'ii', result_mul_337)
            
            # Assigning a BinOp to a Name (line 117):
            
            # Assigning a BinOp to a Name (line 117):
            # Getting the type of 'j' (line 117)
            j_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'j')
            int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'int')
            # Applying the binary operator '*' (line 117)
            result_mul_340 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 17), '*', j_338, int_339)
            
            # Assigning a type to the variable 'jj' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'jj', result_mul_340)
            
            
            # Call to range(...): (line 118)
            # Processing the call arguments (line 118)
            
            # Call to len(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'l' (line 118)
            l_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'l', False)
            # Processing the call keyword arguments (line 118)
            kwargs_344 = {}
            # Getting the type of 'len' (line 118)
            len_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'len', False)
            # Calling len(args, kwargs) (line 118)
            len_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 118, 27), len_342, *[l_343], **kwargs_344)
            
            # Processing the call keyword arguments (line 118)
            kwargs_346 = {}
            # Getting the type of 'range' (line 118)
            range_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'range', False)
            # Calling range(args, kwargs) (line 118)
            range_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), range_341, *[len_call_result_345], **kwargs_346)
            
            # Assigning a type to the variable 'range_call_result_347' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'range_call_result_347', range_call_result_347)
            # Testing if the for loop is going to be iterated (line 118)
            # Testing the type of a for loop iterable (line 118)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347):
                # Getting the type of the for loop variable (line 118)
                for_loop_var_348 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)
                # Assigning a type to the variable 'm' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'm', for_loop_var_348)
                # SSA begins for a for statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 120)
                # Processing the call arguments (line 120)
                int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'int')
                # Processing the call keyword arguments (line 120)
                kwargs_351 = {}
                # Getting the type of 'range' (line 120)
                range_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'range', False)
                # Calling range(args, kwargs) (line 120)
                range_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), range_349, *[int_350], **kwargs_351)
                
                # Assigning a type to the variable 'range_call_result_352' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'range_call_result_352', range_call_result_352)
                # Testing if the for loop is going to be iterated (line 120)
                # Testing the type of a for loop iterable (line 120)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352):
                    # Getting the type of the for loop variable (line 120)
                    for_loop_var_353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)
                    # Assigning a type to the variable 'y' (line 120)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'y', for_loop_var_353)
                    # SSA begins for a for statement (line 120)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to range(...): (line 121)
                    # Processing the call arguments (line 121)
                    int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'int')
                    # Processing the call keyword arguments (line 121)
                    kwargs_356 = {}
                    # Getting the type of 'range' (line 121)
                    range_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'range', False)
                    # Calling range(args, kwargs) (line 121)
                    range_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), range_354, *[int_355], **kwargs_356)
                    
                    # Assigning a type to the variable 'range_call_result_357' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'range_call_result_357', range_call_result_357)
                    # Testing if the for loop is going to be iterated (line 121)
                    # Testing the type of a for loop iterable (line 121)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357):
                        # Getting the type of the for loop variable (line 121)
                        for_loop_var_358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)
                        # Assigning a type to the variable 'x' (line 121)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'x', for_loop_var_358)
                        # SSA begins for a for statement (line 121)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to validMove(...): (line 122)
                        # Processing the call arguments (line 122)
                        # Getting the type of 'puzzle' (line 122)
                        puzzle_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'puzzle', False)
                        # Getting the type of 'x' (line 122)
                        x_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'x', False)
                        # Getting the type of 'ii' (line 122)
                        ii_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 49), 'ii', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 45), '+', x_361, ii_362)
                        
                        # Getting the type of 'y' (line 122)
                        y_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 53), 'y', False)
                        # Getting the type of 'jj' (line 122)
                        jj_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 57), 'jj', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_366 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 53), '+', y_364, jj_365)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 122)
                        m_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 63), 'm', False)
                        # Getting the type of 'l' (line 122)
                        l_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'l', False)
                        # Obtaining the member '__getitem__' of a type (line 122)
                        getitem___369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 61), l_368, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
                        subscript_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 122, 61), getitem___369, m_367)
                        
                        # Processing the call keyword arguments (line 122)
                        kwargs_371 = {}
                        # Getting the type of 'validMove' (line 122)
                        validMove_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'validMove', False)
                        # Calling validMove(args, kwargs) (line 122)
                        validMove_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), validMove_359, *[puzzle_360, result_add_363, result_add_366, subscript_call_result_370], **kwargs_371)
                        
                        # Testing if the type of an if condition is none (line 122)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 122)
                            if_condition_373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372)
                            # Assigning a type to the variable 'if_condition_373' (line 122)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'if_condition_373', if_condition_373)
                            # SSA begins for if statement (line 122)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'm' (line 123)
                            m_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 55), 'm')
                            # Getting the type of 'l' (line 123)
                            l_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 53), 'l')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 53), l_375, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 123, 53), getitem___376, m_374)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 123)
                            x_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'x')
                            # Getting the type of 'ii' (line 123)
                            ii_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'ii')
                            # Applying the binary operator '+' (line 123)
                            result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 35), '+', x_378, ii_379)
                            
                            # Getting the type of 'puzzle' (line 123)
                            puzzle_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), puzzle_381, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 123, 28), getitem___382, result_add_380)
                            
                            # Getting the type of 'y' (line 123)
                            y_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'y')
                            # Getting the type of 'jj' (line 123)
                            jj_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'jj')
                            # Applying the binary operator '+' (line 123)
                            result_add_386 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 43), '+', y_384, jj_385)
                            
                            # Storing an element on a container (line 123)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 28), subscript_call_result_383, (result_add_386, subscript_call_result_377))
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Call to pop(...): (line 124)
                            # Processing the call arguments (line 124)
                            # Getting the type of 'm' (line 124)
                            m_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'm', False)
                            # Processing the call keyword arguments (line 124)
                            kwargs_390 = {}
                            # Getting the type of 'l' (line 124)
                            l_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'l', False)
                            # Obtaining the member 'pop' of a type (line 124)
                            pop_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 37), l_387, 'pop')
                            # Calling pop(args, kwargs) (line 124)
                            pop_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 124, 37), pop_388, *[m_389], **kwargs_390)
                            
                            # Assigning a type to the variable 'backup' (line 124)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'backup', pop_call_result_391)
                            
                            # Call to perm(...): (line 125)
                            # Processing the call arguments (line 125)
                            # Getting the type of 'puzzle' (line 125)
                            puzzle_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'puzzle', False)
                            # Getting the type of 'i' (line 125)
                            i_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 45), 'i', False)
                            # Getting the type of 'j' (line 125)
                            j_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 48), 'j', False)
                            # Getting the type of 'l' (line 125)
                            l_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 51), 'l', False)
                            # Getting the type of 'u' (line 125)
                            u_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 54), 'u', False)
                            # Processing the call keyword arguments (line 125)
                            kwargs_398 = {}
                            # Getting the type of 'perm' (line 125)
                            perm_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'perm', False)
                            # Calling perm(args, kwargs) (line 125)
                            perm_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 125, 32), perm_392, *[puzzle_393, i_394, j_395, l_396, u_397], **kwargs_398)
                            
                            # Testing if the type of an if condition is none (line 125)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399):
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                            else:
                                
                                # Testing the type of an if condition (line 125)
                                if_condition_400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399)
                                # Assigning a type to the variable 'if_condition_400' (line 125)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'if_condition_400', if_condition_400)
                                # SSA begins for if statement (line 125)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                # Getting the type of 'True' (line 126)
                                True_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'True')
                                # Assigning a type to the variable 'stypy_return_type' (line 126)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'stypy_return_type', True_401)
                                # SSA branch for the else part of an if statement (line 125)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                                # SSA join for if statement (line 125)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Call to insert(...): (line 129)
                            # Processing the call arguments (line 129)
                            # Getting the type of 'm' (line 129)
                            m_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'm', False)
                            # Getting the type of 'backup' (line 129)
                            backup_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'backup', False)
                            # Processing the call keyword arguments (line 129)
                            kwargs_410 = {}
                            # Getting the type of 'l' (line 129)
                            l_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'l', False)
                            # Obtaining the member 'insert' of a type (line 129)
                            insert_407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 28), l_406, 'insert')
                            # Calling insert(args, kwargs) (line 129)
                            insert_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 129, 28), insert_407, *[m_408, backup_409], **kwargs_410)
                            
                            
                            # Assigning a Num to a Subscript (line 130):
                            
                            # Assigning a Num to a Subscript (line 130):
                            int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 53), 'int')
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 130)
                            x_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'x')
                            # Getting the type of 'ii' (line 130)
                            ii_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 39), 'ii')
                            # Applying the binary operator '+' (line 130)
                            result_add_415 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '+', x_413, ii_414)
                            
                            # Getting the type of 'puzzle' (line 130)
                            puzzle_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 130)
                            getitem___417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 28), puzzle_416, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                            subscript_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 130, 28), getitem___417, result_add_415)
                            
                            # Getting the type of 'y' (line 130)
                            y_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'y')
                            # Getting the type of 'jj' (line 130)
                            jj_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'jj')
                            # Applying the binary operator '+' (line 130)
                            result_add_421 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 43), '+', y_419, jj_420)
                            
                            # Storing an element on a container (line 130)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 28), subscript_call_result_418, (result_add_421, int_412))
                            # SSA join for if statement (line 122)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'False' (line 131)
            False_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', False_422)
        else:
            
            # Testing the type of an if condition (line 86)
            if_condition_233 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_232)
            # Assigning a type to the variable 'if_condition_233' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_233', if_condition_233)
            # SSA begins for if statement (line 86)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Name (line 90):
            
            # Assigning a List to a Name (line 90):
            
            # Obtaining an instance of the builtin type 'list' (line 90)
            list_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 90)
            
            # Assigning a type to the variable 'puzzlebackup' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'puzzlebackup', list_234)
            
            
            # Call to range(...): (line 91)
            # Processing the call arguments (line 91)
            int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'int')
            # Processing the call keyword arguments (line 91)
            kwargs_237 = {}
            # Getting the type of 'range' (line 91)
            range_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'range', False)
            # Calling range(args, kwargs) (line 91)
            range_call_result_238 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), range_235, *[int_236], **kwargs_237)
            
            # Assigning a type to the variable 'range_call_result_238' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'range_call_result_238', range_call_result_238)
            # Testing if the for loop is going to be iterated (line 91)
            # Testing the type of a for loop iterable (line 91)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_238)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_238):
                # Getting the type of the for loop variable (line 91)
                for_loop_var_239 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_238)
                # Assigning a type to the variable 'c' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'c', for_loop_var_239)
                # SSA begins for a for statement (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 92)
                # Processing the call arguments (line 92)
                
                # Call to tuple(...): (line 92)
                # Processing the call arguments (line 92)
                
                # Obtaining the type of the subscript
                # Getting the type of 'c' (line 92)
                c_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'c', False)
                # Getting the type of 'puzzle' (line 92)
                puzzle_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'puzzle', False)
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 42), puzzle_244, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 92, 42), getitem___245, c_243)
                
                # Processing the call keyword arguments (line 92)
                kwargs_247 = {}
                # Getting the type of 'tuple' (line 92)
                tuple_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'tuple', False)
                # Calling tuple(args, kwargs) (line 92)
                tuple_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 92, 36), tuple_242, *[subscript_call_result_246], **kwargs_247)
                
                # Processing the call keyword arguments (line 92)
                kwargs_249 = {}
                # Getting the type of 'puzzlebackup' (line 92)
                puzzlebackup_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'puzzlebackup', False)
                # Obtaining the member 'append' of a type (line 92)
                append_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), puzzlebackup_240, 'append')
                # Calling append(args, kwargs) (line 92)
                append_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), append_241, *[tuple_call_result_248], **kwargs_249)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to solvePuzzleSimple(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'puzzle' (line 93)
            puzzle_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'puzzle', False)
            # Processing the call keyword arguments (line 93)
            kwargs_253 = {}
            # Getting the type of 'solvePuzzleSimple' (line 93)
            solvePuzzleSimple_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'solvePuzzleSimple', False)
            # Calling solvePuzzleSimple(args, kwargs) (line 93)
            solvePuzzleSimple_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), solvePuzzleSimple_251, *[puzzle_252], **kwargs_253)
            
            
            
            # Call to range(...): (line 96)
            # Processing the call arguments (line 96)
            
            # Call to len(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'u' (line 96)
            u_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'u', False)
            # Processing the call keyword arguments (line 96)
            kwargs_258 = {}
            # Getting the type of 'len' (line 96)
            len_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'len', False)
            # Calling len(args, kwargs) (line 96)
            len_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 96, 27), len_256, *[u_257], **kwargs_258)
            
            # Processing the call keyword arguments (line 96)
            kwargs_260 = {}
            # Getting the type of 'range' (line 96)
            range_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'range', False)
            # Calling range(args, kwargs) (line 96)
            range_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 96, 21), range_255, *[len_call_result_259], **kwargs_260)
            
            # Assigning a type to the variable 'range_call_result_261' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'range_call_result_261', range_call_result_261)
            # Testing if the for loop is going to be iterated (line 96)
            # Testing the type of a for loop iterable (line 96)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 12), range_call_result_261)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 96, 12), range_call_result_261):
                # Getting the type of the for loop variable (line 96)
                for_loop_var_262 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 12), range_call_result_261)
                # Assigning a type to the variable 'c' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'c', for_loop_var_262)
                # SSA begins for a for statement (line 96)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to hash_lookup(...): (line 97)
                # Processing the call arguments (line 97)
                # Getting the type of 'puzzle' (line 97)
                puzzle_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 35), 'puzzle', False)
                # Processing the call keyword arguments (line 97)
                kwargs_265 = {}
                # Getting the type of 'hash_lookup' (line 97)
                hash_lookup_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'hash_lookup', False)
                # Calling hash_lookup(args, kwargs) (line 97)
                hash_lookup_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 97, 23), hash_lookup_263, *[puzzle_264], **kwargs_265)
                
                # Applying the 'not' unary operator (line 97)
                result_not__267 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 19), 'not', hash_lookup_call_result_266)
                
                # Testing if the type of an if condition is none (line 97)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__267):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 97)
                    if_condition_268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__267)
                    # Assigning a type to the variable 'if_condition_268' (line 97)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'if_condition_268', if_condition_268)
                    # SSA begins for if statement (line 97)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Tuple (line 98):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to pop(...): (line 98)
                    # Processing the call arguments (line 98)
                    # Getting the type of 'c' (line 98)
                    c_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 39), 'c', False)
                    # Processing the call keyword arguments (line 98)
                    kwargs_272 = {}
                    # Getting the type of 'u' (line 98)
                    u_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'u', False)
                    # Obtaining the member 'pop' of a type (line 98)
                    pop_270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 33), u_269, 'pop')
                    # Calling pop(args, kwargs) (line 98)
                    pop_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 98, 33), pop_270, *[c_271], **kwargs_272)
                    
                    # Assigning a type to the variable 'call_assignment_1' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_1', pop_call_result_273)
                    
                    # Assigning a Call to a Name (line 98):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_1' (line 98)
                    call_assignment_1_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_1', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_275 = stypy_get_value_from_tuple(call_assignment_1_274, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_2' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_2', stypy_get_value_from_tuple_call_result_275)
                    
                    # Assigning a Name to a Name (line 98):
                    # Getting the type of 'call_assignment_2' (line 98)
                    call_assignment_2_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_2')
                    # Assigning a type to the variable 'inew' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'inew', call_assignment_2_276)
                    
                    # Assigning a Call to a Name (line 98):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_1' (line 98)
                    call_assignment_1_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_1', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_278 = stypy_get_value_from_tuple(call_assignment_1_277, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_3' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_3', stypy_get_value_from_tuple_call_result_278)
                    
                    # Assigning a Name to a Name (line 98):
                    # Getting the type of 'call_assignment_3' (line 98)
                    call_assignment_3_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_3')
                    # Assigning a type to the variable 'jnew' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'jnew', call_assignment_3_279)
                    
                    # Assigning a Call to a Name (line 99):
                    
                    # Assigning a Call to a Name (line 99):
                    
                    # Call to genMoveList(...): (line 99)
                    # Processing the call arguments (line 99)
                    # Getting the type of 'puzzle' (line 99)
                    puzzle_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'puzzle', False)
                    # Getting the type of 'inew' (line 99)
                    inew_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 44), 'inew', False)
                    # Getting the type of 'jnew' (line 99)
                    jnew_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 50), 'jnew', False)
                    # Processing the call keyword arguments (line 99)
                    kwargs_284 = {}
                    # Getting the type of 'genMoveList' (line 99)
                    genMoveList_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'genMoveList', False)
                    # Calling genMoveList(args, kwargs) (line 99)
                    genMoveList_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 99, 24), genMoveList_280, *[puzzle_281, inew_282, jnew_283], **kwargs_284)
                    
                    # Assigning a type to the variable 'l' (line 99)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'l', genMoveList_call_result_285)
                    
                    # Call to perm(...): (line 102)
                    # Processing the call arguments (line 102)
                    # Getting the type of 'puzzle' (line 102)
                    puzzle_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'puzzle', False)
                    # Getting the type of 'inew' (line 102)
                    inew_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'inew', False)
                    # Getting the type of 'jnew' (line 102)
                    jnew_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 42), 'jnew', False)
                    # Getting the type of 'l' (line 102)
                    l_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 48), 'l', False)
                    # Getting the type of 'u' (line 102)
                    u_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'u', False)
                    # Processing the call keyword arguments (line 102)
                    kwargs_292 = {}
                    # Getting the type of 'perm' (line 102)
                    perm_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'perm', False)
                    # Calling perm(args, kwargs) (line 102)
                    perm_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), perm_286, *[puzzle_287, inew_288, jnew_289, l_290, u_291], **kwargs_292)
                    
                    # Testing if the type of an if condition is none (line 102)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 20), perm_call_result_293):
                        
                        # Call to hash_add(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'puzzle' (line 105)
                        puzzle_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'puzzle', False)
                        # Processing the call keyword arguments (line 105)
                        kwargs_298 = {}
                        # Getting the type of 'hash_add' (line 105)
                        hash_add_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'hash_add', False)
                        # Calling hash_add(args, kwargs) (line 105)
                        hash_add_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), hash_add_296, *[puzzle_297], **kwargs_298)
                        
                    else:
                        
                        # Testing the type of an if condition (line 102)
                        if_condition_294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 20), perm_call_result_293)
                        # Assigning a type to the variable 'if_condition_294' (line 102)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'if_condition_294', if_condition_294)
                        # SSA begins for if statement (line 102)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'True' (line 103)
                        True_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'True')
                        # Assigning a type to the variable 'stypy_return_type' (line 103)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'stypy_return_type', True_295)
                        # SSA branch for the else part of an if statement (line 102)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to hash_add(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'puzzle' (line 105)
                        puzzle_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'puzzle', False)
                        # Processing the call keyword arguments (line 105)
                        kwargs_298 = {}
                        # Getting the type of 'hash_add' (line 105)
                        hash_add_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'hash_add', False)
                        # Calling hash_add(args, kwargs) (line 105)
                        hash_add_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), hash_add_296, *[puzzle_297], **kwargs_298)
                        
                        # SSA join for if statement (line 102)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Call to insert(...): (line 106)
                    # Processing the call arguments (line 106)
                    # Getting the type of 'c' (line 106)
                    c_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'c', False)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 106)
                    tuple_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 33), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 106)
                    # Adding element type (line 106)
                    # Getting the type of 'inew' (line 106)
                    inew_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'inew', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 33), tuple_303, inew_304)
                    # Adding element type (line 106)
                    # Getting the type of 'jnew' (line 106)
                    jnew_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'jnew', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 33), tuple_303, jnew_305)
                    
                    # Processing the call keyword arguments (line 106)
                    kwargs_306 = {}
                    # Getting the type of 'u' (line 106)
                    u_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'u', False)
                    # Obtaining the member 'insert' of a type (line 106)
                    insert_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), u_300, 'insert')
                    # Calling insert(args, kwargs) (line 106)
                    insert_call_result_307 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), insert_301, *[c_302, tuple_303], **kwargs_306)
                    
                    # SSA join for if statement (line 97)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            
            # Call to range(...): (line 109)
            # Processing the call arguments (line 109)
            int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 27), 'int')
            # Processing the call keyword arguments (line 109)
            kwargs_310 = {}
            # Getting the type of 'range' (line 109)
            range_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'range', False)
            # Calling range(args, kwargs) (line 109)
            range_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 109, 21), range_308, *[int_309], **kwargs_310)
            
            # Assigning a type to the variable 'range_call_result_311' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'range_call_result_311', range_call_result_311)
            # Testing if the for loop is going to be iterated (line 109)
            # Testing the type of a for loop iterable (line 109)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 12), range_call_result_311)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 12), range_call_result_311):
                # Getting the type of the for loop variable (line 109)
                for_loop_var_312 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 12), range_call_result_311)
                # Assigning a type to the variable 'y' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'y', for_loop_var_312)
                # SSA begins for a for statement (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 110)
                # Processing the call arguments (line 110)
                int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
                # Processing the call keyword arguments (line 110)
                kwargs_315 = {}
                # Getting the type of 'range' (line 110)
                range_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'range', False)
                # Calling range(args, kwargs) (line 110)
                range_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), range_313, *[int_314], **kwargs_315)
                
                # Assigning a type to the variable 'range_call_result_316' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'range_call_result_316', range_call_result_316)
                # Testing if the for loop is going to be iterated (line 110)
                # Testing the type of a for loop iterable (line 110)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_316)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_316):
                    # Getting the type of the for loop variable (line 110)
                    for_loop_var_317 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_316)
                    # Assigning a type to the variable 'x' (line 110)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'x', for_loop_var_317)
                    # SSA begins for a for statement (line 110)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Subscript to a Subscript (line 111):
                    
                    # Assigning a Subscript to a Subscript (line 111):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'y' (line 111)
                    y_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 51), 'y')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'x' (line 111)
                    x_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 48), 'x')
                    # Getting the type of 'puzzlebackup' (line 111)
                    puzzlebackup_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'puzzlebackup')
                    # Obtaining the member '__getitem__' of a type (line 111)
                    getitem___321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 35), puzzlebackup_320, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                    subscript_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 111, 35), getitem___321, x_319)
                    
                    # Obtaining the member '__getitem__' of a type (line 111)
                    getitem___323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 35), subscript_call_result_322, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                    subscript_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 111, 35), getitem___323, y_318)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'x' (line 111)
                    x_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'x')
                    # Getting the type of 'puzzle' (line 111)
                    puzzle_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'puzzle')
                    # Obtaining the member '__getitem__' of a type (line 111)
                    getitem___327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 20), puzzle_326, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                    subscript_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), getitem___327, x_325)
                    
                    # Getting the type of 'y' (line 111)
                    y_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'y')
                    # Storing an element on a container (line 111)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 20), subscript_call_result_328, (y_329, subscript_call_result_324))
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to hash_add(...): (line 112)
            # Processing the call arguments (line 112)
            # Getting the type of 'puzzle' (line 112)
            puzzle_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'puzzle', False)
            # Processing the call keyword arguments (line 112)
            kwargs_332 = {}
            # Getting the type of 'hash_add' (line 112)
            hash_add_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'hash_add', False)
            # Calling hash_add(args, kwargs) (line 112)
            hash_add_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), hash_add_330, *[puzzle_331], **kwargs_332)
            
            # Getting the type of 'False' (line 113)
            False_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'stypy_return_type', False_334)
            # SSA branch for the else part of an if statement (line 86)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 116):
            
            # Assigning a BinOp to a Name (line 116):
            # Getting the type of 'i' (line 116)
            i_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'i')
            int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
            # Applying the binary operator '*' (line 116)
            result_mul_337 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '*', i_335, int_336)
            
            # Assigning a type to the variable 'ii' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'ii', result_mul_337)
            
            # Assigning a BinOp to a Name (line 117):
            
            # Assigning a BinOp to a Name (line 117):
            # Getting the type of 'j' (line 117)
            j_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'j')
            int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'int')
            # Applying the binary operator '*' (line 117)
            result_mul_340 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 17), '*', j_338, int_339)
            
            # Assigning a type to the variable 'jj' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'jj', result_mul_340)
            
            
            # Call to range(...): (line 118)
            # Processing the call arguments (line 118)
            
            # Call to len(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'l' (line 118)
            l_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'l', False)
            # Processing the call keyword arguments (line 118)
            kwargs_344 = {}
            # Getting the type of 'len' (line 118)
            len_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'len', False)
            # Calling len(args, kwargs) (line 118)
            len_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 118, 27), len_342, *[l_343], **kwargs_344)
            
            # Processing the call keyword arguments (line 118)
            kwargs_346 = {}
            # Getting the type of 'range' (line 118)
            range_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'range', False)
            # Calling range(args, kwargs) (line 118)
            range_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), range_341, *[len_call_result_345], **kwargs_346)
            
            # Assigning a type to the variable 'range_call_result_347' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'range_call_result_347', range_call_result_347)
            # Testing if the for loop is going to be iterated (line 118)
            # Testing the type of a for loop iterable (line 118)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347):
                # Getting the type of the for loop variable (line 118)
                for_loop_var_348 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)
                # Assigning a type to the variable 'm' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'm', for_loop_var_348)
                # SSA begins for a for statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 120)
                # Processing the call arguments (line 120)
                int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'int')
                # Processing the call keyword arguments (line 120)
                kwargs_351 = {}
                # Getting the type of 'range' (line 120)
                range_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'range', False)
                # Calling range(args, kwargs) (line 120)
                range_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), range_349, *[int_350], **kwargs_351)
                
                # Assigning a type to the variable 'range_call_result_352' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'range_call_result_352', range_call_result_352)
                # Testing if the for loop is going to be iterated (line 120)
                # Testing the type of a for loop iterable (line 120)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352):
                    # Getting the type of the for loop variable (line 120)
                    for_loop_var_353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)
                    # Assigning a type to the variable 'y' (line 120)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'y', for_loop_var_353)
                    # SSA begins for a for statement (line 120)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to range(...): (line 121)
                    # Processing the call arguments (line 121)
                    int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'int')
                    # Processing the call keyword arguments (line 121)
                    kwargs_356 = {}
                    # Getting the type of 'range' (line 121)
                    range_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'range', False)
                    # Calling range(args, kwargs) (line 121)
                    range_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), range_354, *[int_355], **kwargs_356)
                    
                    # Assigning a type to the variable 'range_call_result_357' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'range_call_result_357', range_call_result_357)
                    # Testing if the for loop is going to be iterated (line 121)
                    # Testing the type of a for loop iterable (line 121)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357):
                        # Getting the type of the for loop variable (line 121)
                        for_loop_var_358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)
                        # Assigning a type to the variable 'x' (line 121)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'x', for_loop_var_358)
                        # SSA begins for a for statement (line 121)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to validMove(...): (line 122)
                        # Processing the call arguments (line 122)
                        # Getting the type of 'puzzle' (line 122)
                        puzzle_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'puzzle', False)
                        # Getting the type of 'x' (line 122)
                        x_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'x', False)
                        # Getting the type of 'ii' (line 122)
                        ii_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 49), 'ii', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 45), '+', x_361, ii_362)
                        
                        # Getting the type of 'y' (line 122)
                        y_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 53), 'y', False)
                        # Getting the type of 'jj' (line 122)
                        jj_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 57), 'jj', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_366 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 53), '+', y_364, jj_365)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 122)
                        m_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 63), 'm', False)
                        # Getting the type of 'l' (line 122)
                        l_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'l', False)
                        # Obtaining the member '__getitem__' of a type (line 122)
                        getitem___369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 61), l_368, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
                        subscript_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 122, 61), getitem___369, m_367)
                        
                        # Processing the call keyword arguments (line 122)
                        kwargs_371 = {}
                        # Getting the type of 'validMove' (line 122)
                        validMove_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'validMove', False)
                        # Calling validMove(args, kwargs) (line 122)
                        validMove_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), validMove_359, *[puzzle_360, result_add_363, result_add_366, subscript_call_result_370], **kwargs_371)
                        
                        # Testing if the type of an if condition is none (line 122)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 122)
                            if_condition_373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372)
                            # Assigning a type to the variable 'if_condition_373' (line 122)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'if_condition_373', if_condition_373)
                            # SSA begins for if statement (line 122)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'm' (line 123)
                            m_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 55), 'm')
                            # Getting the type of 'l' (line 123)
                            l_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 53), 'l')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 53), l_375, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 123, 53), getitem___376, m_374)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 123)
                            x_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'x')
                            # Getting the type of 'ii' (line 123)
                            ii_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'ii')
                            # Applying the binary operator '+' (line 123)
                            result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 35), '+', x_378, ii_379)
                            
                            # Getting the type of 'puzzle' (line 123)
                            puzzle_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), puzzle_381, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 123, 28), getitem___382, result_add_380)
                            
                            # Getting the type of 'y' (line 123)
                            y_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'y')
                            # Getting the type of 'jj' (line 123)
                            jj_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'jj')
                            # Applying the binary operator '+' (line 123)
                            result_add_386 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 43), '+', y_384, jj_385)
                            
                            # Storing an element on a container (line 123)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 28), subscript_call_result_383, (result_add_386, subscript_call_result_377))
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Call to pop(...): (line 124)
                            # Processing the call arguments (line 124)
                            # Getting the type of 'm' (line 124)
                            m_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'm', False)
                            # Processing the call keyword arguments (line 124)
                            kwargs_390 = {}
                            # Getting the type of 'l' (line 124)
                            l_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'l', False)
                            # Obtaining the member 'pop' of a type (line 124)
                            pop_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 37), l_387, 'pop')
                            # Calling pop(args, kwargs) (line 124)
                            pop_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 124, 37), pop_388, *[m_389], **kwargs_390)
                            
                            # Assigning a type to the variable 'backup' (line 124)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'backup', pop_call_result_391)
                            
                            # Call to perm(...): (line 125)
                            # Processing the call arguments (line 125)
                            # Getting the type of 'puzzle' (line 125)
                            puzzle_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'puzzle', False)
                            # Getting the type of 'i' (line 125)
                            i_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 45), 'i', False)
                            # Getting the type of 'j' (line 125)
                            j_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 48), 'j', False)
                            # Getting the type of 'l' (line 125)
                            l_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 51), 'l', False)
                            # Getting the type of 'u' (line 125)
                            u_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 54), 'u', False)
                            # Processing the call keyword arguments (line 125)
                            kwargs_398 = {}
                            # Getting the type of 'perm' (line 125)
                            perm_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'perm', False)
                            # Calling perm(args, kwargs) (line 125)
                            perm_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 125, 32), perm_392, *[puzzle_393, i_394, j_395, l_396, u_397], **kwargs_398)
                            
                            # Testing if the type of an if condition is none (line 125)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399):
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                            else:
                                
                                # Testing the type of an if condition (line 125)
                                if_condition_400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399)
                                # Assigning a type to the variable 'if_condition_400' (line 125)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'if_condition_400', if_condition_400)
                                # SSA begins for if statement (line 125)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                # Getting the type of 'True' (line 126)
                                True_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'True')
                                # Assigning a type to the variable 'stypy_return_type' (line 126)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'stypy_return_type', True_401)
                                # SSA branch for the else part of an if statement (line 125)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                                # SSA join for if statement (line 125)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Call to insert(...): (line 129)
                            # Processing the call arguments (line 129)
                            # Getting the type of 'm' (line 129)
                            m_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'm', False)
                            # Getting the type of 'backup' (line 129)
                            backup_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'backup', False)
                            # Processing the call keyword arguments (line 129)
                            kwargs_410 = {}
                            # Getting the type of 'l' (line 129)
                            l_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'l', False)
                            # Obtaining the member 'insert' of a type (line 129)
                            insert_407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 28), l_406, 'insert')
                            # Calling insert(args, kwargs) (line 129)
                            insert_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 129, 28), insert_407, *[m_408, backup_409], **kwargs_410)
                            
                            
                            # Assigning a Num to a Subscript (line 130):
                            
                            # Assigning a Num to a Subscript (line 130):
                            int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 53), 'int')
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 130)
                            x_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'x')
                            # Getting the type of 'ii' (line 130)
                            ii_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 39), 'ii')
                            # Applying the binary operator '+' (line 130)
                            result_add_415 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '+', x_413, ii_414)
                            
                            # Getting the type of 'puzzle' (line 130)
                            puzzle_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 130)
                            getitem___417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 28), puzzle_416, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                            subscript_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 130, 28), getitem___417, result_add_415)
                            
                            # Getting the type of 'y' (line 130)
                            y_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'y')
                            # Getting the type of 'jj' (line 130)
                            jj_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'jj')
                            # Applying the binary operator '+' (line 130)
                            result_add_421 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 43), '+', y_419, jj_420)
                            
                            # Storing an element on a container (line 130)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 28), subscript_call_result_418, (result_add_421, int_412))
                            # SSA join for if statement (line 122)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'False' (line 131)
            False_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', False_422)
            # SSA join for if statement (line 86)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 80)
        if_condition_224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_and_keyword_223)
        # Assigning a type to the variable 'if_condition_224' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_224', if_condition_224)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to printpuzzle(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'puzzle' (line 82)
        puzzle_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'puzzle', False)
        # Processing the call keyword arguments (line 82)
        kwargs_227 = {}
        # Getting the type of 'printpuzzle' (line 82)
        printpuzzle_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'printpuzzle', False)
        # Calling printpuzzle(args, kwargs) (line 82)
        printpuzzle_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), printpuzzle_225, *[puzzle_226], **kwargs_227)
        
        # Getting the type of 'True' (line 84)
        True_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', True_229)
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'l' (line 86)
        l_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'l')
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        
        # Applying the binary operator '==' (line 86)
        result_eq_232 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), '==', l_230, list_231)
        
        # Testing if the type of an if condition is none (line 86)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_232):
            
            # Assigning a BinOp to a Name (line 116):
            
            # Assigning a BinOp to a Name (line 116):
            # Getting the type of 'i' (line 116)
            i_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'i')
            int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
            # Applying the binary operator '*' (line 116)
            result_mul_337 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '*', i_335, int_336)
            
            # Assigning a type to the variable 'ii' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'ii', result_mul_337)
            
            # Assigning a BinOp to a Name (line 117):
            
            # Assigning a BinOp to a Name (line 117):
            # Getting the type of 'j' (line 117)
            j_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'j')
            int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'int')
            # Applying the binary operator '*' (line 117)
            result_mul_340 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 17), '*', j_338, int_339)
            
            # Assigning a type to the variable 'jj' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'jj', result_mul_340)
            
            
            # Call to range(...): (line 118)
            # Processing the call arguments (line 118)
            
            # Call to len(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'l' (line 118)
            l_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'l', False)
            # Processing the call keyword arguments (line 118)
            kwargs_344 = {}
            # Getting the type of 'len' (line 118)
            len_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'len', False)
            # Calling len(args, kwargs) (line 118)
            len_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 118, 27), len_342, *[l_343], **kwargs_344)
            
            # Processing the call keyword arguments (line 118)
            kwargs_346 = {}
            # Getting the type of 'range' (line 118)
            range_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'range', False)
            # Calling range(args, kwargs) (line 118)
            range_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), range_341, *[len_call_result_345], **kwargs_346)
            
            # Assigning a type to the variable 'range_call_result_347' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'range_call_result_347', range_call_result_347)
            # Testing if the for loop is going to be iterated (line 118)
            # Testing the type of a for loop iterable (line 118)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347):
                # Getting the type of the for loop variable (line 118)
                for_loop_var_348 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)
                # Assigning a type to the variable 'm' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'm', for_loop_var_348)
                # SSA begins for a for statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 120)
                # Processing the call arguments (line 120)
                int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'int')
                # Processing the call keyword arguments (line 120)
                kwargs_351 = {}
                # Getting the type of 'range' (line 120)
                range_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'range', False)
                # Calling range(args, kwargs) (line 120)
                range_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), range_349, *[int_350], **kwargs_351)
                
                # Assigning a type to the variable 'range_call_result_352' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'range_call_result_352', range_call_result_352)
                # Testing if the for loop is going to be iterated (line 120)
                # Testing the type of a for loop iterable (line 120)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352):
                    # Getting the type of the for loop variable (line 120)
                    for_loop_var_353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)
                    # Assigning a type to the variable 'y' (line 120)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'y', for_loop_var_353)
                    # SSA begins for a for statement (line 120)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to range(...): (line 121)
                    # Processing the call arguments (line 121)
                    int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'int')
                    # Processing the call keyword arguments (line 121)
                    kwargs_356 = {}
                    # Getting the type of 'range' (line 121)
                    range_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'range', False)
                    # Calling range(args, kwargs) (line 121)
                    range_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), range_354, *[int_355], **kwargs_356)
                    
                    # Assigning a type to the variable 'range_call_result_357' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'range_call_result_357', range_call_result_357)
                    # Testing if the for loop is going to be iterated (line 121)
                    # Testing the type of a for loop iterable (line 121)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357):
                        # Getting the type of the for loop variable (line 121)
                        for_loop_var_358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)
                        # Assigning a type to the variable 'x' (line 121)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'x', for_loop_var_358)
                        # SSA begins for a for statement (line 121)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to validMove(...): (line 122)
                        # Processing the call arguments (line 122)
                        # Getting the type of 'puzzle' (line 122)
                        puzzle_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'puzzle', False)
                        # Getting the type of 'x' (line 122)
                        x_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'x', False)
                        # Getting the type of 'ii' (line 122)
                        ii_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 49), 'ii', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 45), '+', x_361, ii_362)
                        
                        # Getting the type of 'y' (line 122)
                        y_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 53), 'y', False)
                        # Getting the type of 'jj' (line 122)
                        jj_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 57), 'jj', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_366 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 53), '+', y_364, jj_365)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 122)
                        m_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 63), 'm', False)
                        # Getting the type of 'l' (line 122)
                        l_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'l', False)
                        # Obtaining the member '__getitem__' of a type (line 122)
                        getitem___369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 61), l_368, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
                        subscript_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 122, 61), getitem___369, m_367)
                        
                        # Processing the call keyword arguments (line 122)
                        kwargs_371 = {}
                        # Getting the type of 'validMove' (line 122)
                        validMove_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'validMove', False)
                        # Calling validMove(args, kwargs) (line 122)
                        validMove_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), validMove_359, *[puzzle_360, result_add_363, result_add_366, subscript_call_result_370], **kwargs_371)
                        
                        # Testing if the type of an if condition is none (line 122)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 122)
                            if_condition_373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372)
                            # Assigning a type to the variable 'if_condition_373' (line 122)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'if_condition_373', if_condition_373)
                            # SSA begins for if statement (line 122)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'm' (line 123)
                            m_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 55), 'm')
                            # Getting the type of 'l' (line 123)
                            l_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 53), 'l')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 53), l_375, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 123, 53), getitem___376, m_374)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 123)
                            x_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'x')
                            # Getting the type of 'ii' (line 123)
                            ii_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'ii')
                            # Applying the binary operator '+' (line 123)
                            result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 35), '+', x_378, ii_379)
                            
                            # Getting the type of 'puzzle' (line 123)
                            puzzle_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), puzzle_381, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 123, 28), getitem___382, result_add_380)
                            
                            # Getting the type of 'y' (line 123)
                            y_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'y')
                            # Getting the type of 'jj' (line 123)
                            jj_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'jj')
                            # Applying the binary operator '+' (line 123)
                            result_add_386 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 43), '+', y_384, jj_385)
                            
                            # Storing an element on a container (line 123)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 28), subscript_call_result_383, (result_add_386, subscript_call_result_377))
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Call to pop(...): (line 124)
                            # Processing the call arguments (line 124)
                            # Getting the type of 'm' (line 124)
                            m_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'm', False)
                            # Processing the call keyword arguments (line 124)
                            kwargs_390 = {}
                            # Getting the type of 'l' (line 124)
                            l_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'l', False)
                            # Obtaining the member 'pop' of a type (line 124)
                            pop_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 37), l_387, 'pop')
                            # Calling pop(args, kwargs) (line 124)
                            pop_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 124, 37), pop_388, *[m_389], **kwargs_390)
                            
                            # Assigning a type to the variable 'backup' (line 124)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'backup', pop_call_result_391)
                            
                            # Call to perm(...): (line 125)
                            # Processing the call arguments (line 125)
                            # Getting the type of 'puzzle' (line 125)
                            puzzle_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'puzzle', False)
                            # Getting the type of 'i' (line 125)
                            i_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 45), 'i', False)
                            # Getting the type of 'j' (line 125)
                            j_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 48), 'j', False)
                            # Getting the type of 'l' (line 125)
                            l_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 51), 'l', False)
                            # Getting the type of 'u' (line 125)
                            u_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 54), 'u', False)
                            # Processing the call keyword arguments (line 125)
                            kwargs_398 = {}
                            # Getting the type of 'perm' (line 125)
                            perm_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'perm', False)
                            # Calling perm(args, kwargs) (line 125)
                            perm_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 125, 32), perm_392, *[puzzle_393, i_394, j_395, l_396, u_397], **kwargs_398)
                            
                            # Testing if the type of an if condition is none (line 125)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399):
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                            else:
                                
                                # Testing the type of an if condition (line 125)
                                if_condition_400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399)
                                # Assigning a type to the variable 'if_condition_400' (line 125)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'if_condition_400', if_condition_400)
                                # SSA begins for if statement (line 125)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                # Getting the type of 'True' (line 126)
                                True_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'True')
                                # Assigning a type to the variable 'stypy_return_type' (line 126)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'stypy_return_type', True_401)
                                # SSA branch for the else part of an if statement (line 125)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                                # SSA join for if statement (line 125)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Call to insert(...): (line 129)
                            # Processing the call arguments (line 129)
                            # Getting the type of 'm' (line 129)
                            m_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'm', False)
                            # Getting the type of 'backup' (line 129)
                            backup_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'backup', False)
                            # Processing the call keyword arguments (line 129)
                            kwargs_410 = {}
                            # Getting the type of 'l' (line 129)
                            l_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'l', False)
                            # Obtaining the member 'insert' of a type (line 129)
                            insert_407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 28), l_406, 'insert')
                            # Calling insert(args, kwargs) (line 129)
                            insert_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 129, 28), insert_407, *[m_408, backup_409], **kwargs_410)
                            
                            
                            # Assigning a Num to a Subscript (line 130):
                            
                            # Assigning a Num to a Subscript (line 130):
                            int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 53), 'int')
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 130)
                            x_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'x')
                            # Getting the type of 'ii' (line 130)
                            ii_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 39), 'ii')
                            # Applying the binary operator '+' (line 130)
                            result_add_415 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '+', x_413, ii_414)
                            
                            # Getting the type of 'puzzle' (line 130)
                            puzzle_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 130)
                            getitem___417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 28), puzzle_416, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                            subscript_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 130, 28), getitem___417, result_add_415)
                            
                            # Getting the type of 'y' (line 130)
                            y_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'y')
                            # Getting the type of 'jj' (line 130)
                            jj_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'jj')
                            # Applying the binary operator '+' (line 130)
                            result_add_421 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 43), '+', y_419, jj_420)
                            
                            # Storing an element on a container (line 130)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 28), subscript_call_result_418, (result_add_421, int_412))
                            # SSA join for if statement (line 122)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'False' (line 131)
            False_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', False_422)
        else:
            
            # Testing the type of an if condition (line 86)
            if_condition_233 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_232)
            # Assigning a type to the variable 'if_condition_233' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_233', if_condition_233)
            # SSA begins for if statement (line 86)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Name (line 90):
            
            # Assigning a List to a Name (line 90):
            
            # Obtaining an instance of the builtin type 'list' (line 90)
            list_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 90)
            
            # Assigning a type to the variable 'puzzlebackup' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'puzzlebackup', list_234)
            
            
            # Call to range(...): (line 91)
            # Processing the call arguments (line 91)
            int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'int')
            # Processing the call keyword arguments (line 91)
            kwargs_237 = {}
            # Getting the type of 'range' (line 91)
            range_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'range', False)
            # Calling range(args, kwargs) (line 91)
            range_call_result_238 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), range_235, *[int_236], **kwargs_237)
            
            # Assigning a type to the variable 'range_call_result_238' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'range_call_result_238', range_call_result_238)
            # Testing if the for loop is going to be iterated (line 91)
            # Testing the type of a for loop iterable (line 91)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_238)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_238):
                # Getting the type of the for loop variable (line 91)
                for_loop_var_239 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_238)
                # Assigning a type to the variable 'c' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'c', for_loop_var_239)
                # SSA begins for a for statement (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 92)
                # Processing the call arguments (line 92)
                
                # Call to tuple(...): (line 92)
                # Processing the call arguments (line 92)
                
                # Obtaining the type of the subscript
                # Getting the type of 'c' (line 92)
                c_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'c', False)
                # Getting the type of 'puzzle' (line 92)
                puzzle_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'puzzle', False)
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 42), puzzle_244, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 92, 42), getitem___245, c_243)
                
                # Processing the call keyword arguments (line 92)
                kwargs_247 = {}
                # Getting the type of 'tuple' (line 92)
                tuple_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'tuple', False)
                # Calling tuple(args, kwargs) (line 92)
                tuple_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 92, 36), tuple_242, *[subscript_call_result_246], **kwargs_247)
                
                # Processing the call keyword arguments (line 92)
                kwargs_249 = {}
                # Getting the type of 'puzzlebackup' (line 92)
                puzzlebackup_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'puzzlebackup', False)
                # Obtaining the member 'append' of a type (line 92)
                append_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), puzzlebackup_240, 'append')
                # Calling append(args, kwargs) (line 92)
                append_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), append_241, *[tuple_call_result_248], **kwargs_249)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to solvePuzzleSimple(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'puzzle' (line 93)
            puzzle_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'puzzle', False)
            # Processing the call keyword arguments (line 93)
            kwargs_253 = {}
            # Getting the type of 'solvePuzzleSimple' (line 93)
            solvePuzzleSimple_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'solvePuzzleSimple', False)
            # Calling solvePuzzleSimple(args, kwargs) (line 93)
            solvePuzzleSimple_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), solvePuzzleSimple_251, *[puzzle_252], **kwargs_253)
            
            
            
            # Call to range(...): (line 96)
            # Processing the call arguments (line 96)
            
            # Call to len(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'u' (line 96)
            u_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'u', False)
            # Processing the call keyword arguments (line 96)
            kwargs_258 = {}
            # Getting the type of 'len' (line 96)
            len_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'len', False)
            # Calling len(args, kwargs) (line 96)
            len_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 96, 27), len_256, *[u_257], **kwargs_258)
            
            # Processing the call keyword arguments (line 96)
            kwargs_260 = {}
            # Getting the type of 'range' (line 96)
            range_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'range', False)
            # Calling range(args, kwargs) (line 96)
            range_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 96, 21), range_255, *[len_call_result_259], **kwargs_260)
            
            # Assigning a type to the variable 'range_call_result_261' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'range_call_result_261', range_call_result_261)
            # Testing if the for loop is going to be iterated (line 96)
            # Testing the type of a for loop iterable (line 96)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 12), range_call_result_261)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 96, 12), range_call_result_261):
                # Getting the type of the for loop variable (line 96)
                for_loop_var_262 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 12), range_call_result_261)
                # Assigning a type to the variable 'c' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'c', for_loop_var_262)
                # SSA begins for a for statement (line 96)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to hash_lookup(...): (line 97)
                # Processing the call arguments (line 97)
                # Getting the type of 'puzzle' (line 97)
                puzzle_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 35), 'puzzle', False)
                # Processing the call keyword arguments (line 97)
                kwargs_265 = {}
                # Getting the type of 'hash_lookup' (line 97)
                hash_lookup_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'hash_lookup', False)
                # Calling hash_lookup(args, kwargs) (line 97)
                hash_lookup_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 97, 23), hash_lookup_263, *[puzzle_264], **kwargs_265)
                
                # Applying the 'not' unary operator (line 97)
                result_not__267 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 19), 'not', hash_lookup_call_result_266)
                
                # Testing if the type of an if condition is none (line 97)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__267):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 97)
                    if_condition_268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__267)
                    # Assigning a type to the variable 'if_condition_268' (line 97)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'if_condition_268', if_condition_268)
                    # SSA begins for if statement (line 97)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Tuple (line 98):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to pop(...): (line 98)
                    # Processing the call arguments (line 98)
                    # Getting the type of 'c' (line 98)
                    c_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 39), 'c', False)
                    # Processing the call keyword arguments (line 98)
                    kwargs_272 = {}
                    # Getting the type of 'u' (line 98)
                    u_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'u', False)
                    # Obtaining the member 'pop' of a type (line 98)
                    pop_270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 33), u_269, 'pop')
                    # Calling pop(args, kwargs) (line 98)
                    pop_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 98, 33), pop_270, *[c_271], **kwargs_272)
                    
                    # Assigning a type to the variable 'call_assignment_1' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_1', pop_call_result_273)
                    
                    # Assigning a Call to a Name (line 98):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_1' (line 98)
                    call_assignment_1_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_1', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_275 = stypy_get_value_from_tuple(call_assignment_1_274, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_2' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_2', stypy_get_value_from_tuple_call_result_275)
                    
                    # Assigning a Name to a Name (line 98):
                    # Getting the type of 'call_assignment_2' (line 98)
                    call_assignment_2_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_2')
                    # Assigning a type to the variable 'inew' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'inew', call_assignment_2_276)
                    
                    # Assigning a Call to a Name (line 98):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_1' (line 98)
                    call_assignment_1_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_1', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_278 = stypy_get_value_from_tuple(call_assignment_1_277, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_3' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_3', stypy_get_value_from_tuple_call_result_278)
                    
                    # Assigning a Name to a Name (line 98):
                    # Getting the type of 'call_assignment_3' (line 98)
                    call_assignment_3_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_assignment_3')
                    # Assigning a type to the variable 'jnew' (line 98)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'jnew', call_assignment_3_279)
                    
                    # Assigning a Call to a Name (line 99):
                    
                    # Assigning a Call to a Name (line 99):
                    
                    # Call to genMoveList(...): (line 99)
                    # Processing the call arguments (line 99)
                    # Getting the type of 'puzzle' (line 99)
                    puzzle_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'puzzle', False)
                    # Getting the type of 'inew' (line 99)
                    inew_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 44), 'inew', False)
                    # Getting the type of 'jnew' (line 99)
                    jnew_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 50), 'jnew', False)
                    # Processing the call keyword arguments (line 99)
                    kwargs_284 = {}
                    # Getting the type of 'genMoveList' (line 99)
                    genMoveList_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'genMoveList', False)
                    # Calling genMoveList(args, kwargs) (line 99)
                    genMoveList_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 99, 24), genMoveList_280, *[puzzle_281, inew_282, jnew_283], **kwargs_284)
                    
                    # Assigning a type to the variable 'l' (line 99)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'l', genMoveList_call_result_285)
                    
                    # Call to perm(...): (line 102)
                    # Processing the call arguments (line 102)
                    # Getting the type of 'puzzle' (line 102)
                    puzzle_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'puzzle', False)
                    # Getting the type of 'inew' (line 102)
                    inew_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'inew', False)
                    # Getting the type of 'jnew' (line 102)
                    jnew_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 42), 'jnew', False)
                    # Getting the type of 'l' (line 102)
                    l_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 48), 'l', False)
                    # Getting the type of 'u' (line 102)
                    u_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'u', False)
                    # Processing the call keyword arguments (line 102)
                    kwargs_292 = {}
                    # Getting the type of 'perm' (line 102)
                    perm_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'perm', False)
                    # Calling perm(args, kwargs) (line 102)
                    perm_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), perm_286, *[puzzle_287, inew_288, jnew_289, l_290, u_291], **kwargs_292)
                    
                    # Testing if the type of an if condition is none (line 102)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 20), perm_call_result_293):
                        
                        # Call to hash_add(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'puzzle' (line 105)
                        puzzle_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'puzzle', False)
                        # Processing the call keyword arguments (line 105)
                        kwargs_298 = {}
                        # Getting the type of 'hash_add' (line 105)
                        hash_add_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'hash_add', False)
                        # Calling hash_add(args, kwargs) (line 105)
                        hash_add_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), hash_add_296, *[puzzle_297], **kwargs_298)
                        
                    else:
                        
                        # Testing the type of an if condition (line 102)
                        if_condition_294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 20), perm_call_result_293)
                        # Assigning a type to the variable 'if_condition_294' (line 102)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'if_condition_294', if_condition_294)
                        # SSA begins for if statement (line 102)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'True' (line 103)
                        True_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'True')
                        # Assigning a type to the variable 'stypy_return_type' (line 103)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'stypy_return_type', True_295)
                        # SSA branch for the else part of an if statement (line 102)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to hash_add(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'puzzle' (line 105)
                        puzzle_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'puzzle', False)
                        # Processing the call keyword arguments (line 105)
                        kwargs_298 = {}
                        # Getting the type of 'hash_add' (line 105)
                        hash_add_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'hash_add', False)
                        # Calling hash_add(args, kwargs) (line 105)
                        hash_add_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), hash_add_296, *[puzzle_297], **kwargs_298)
                        
                        # SSA join for if statement (line 102)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Call to insert(...): (line 106)
                    # Processing the call arguments (line 106)
                    # Getting the type of 'c' (line 106)
                    c_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'c', False)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 106)
                    tuple_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 33), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 106)
                    # Adding element type (line 106)
                    # Getting the type of 'inew' (line 106)
                    inew_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'inew', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 33), tuple_303, inew_304)
                    # Adding element type (line 106)
                    # Getting the type of 'jnew' (line 106)
                    jnew_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'jnew', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 33), tuple_303, jnew_305)
                    
                    # Processing the call keyword arguments (line 106)
                    kwargs_306 = {}
                    # Getting the type of 'u' (line 106)
                    u_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'u', False)
                    # Obtaining the member 'insert' of a type (line 106)
                    insert_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), u_300, 'insert')
                    # Calling insert(args, kwargs) (line 106)
                    insert_call_result_307 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), insert_301, *[c_302, tuple_303], **kwargs_306)
                    
                    # SSA join for if statement (line 97)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            
            # Call to range(...): (line 109)
            # Processing the call arguments (line 109)
            int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 27), 'int')
            # Processing the call keyword arguments (line 109)
            kwargs_310 = {}
            # Getting the type of 'range' (line 109)
            range_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'range', False)
            # Calling range(args, kwargs) (line 109)
            range_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 109, 21), range_308, *[int_309], **kwargs_310)
            
            # Assigning a type to the variable 'range_call_result_311' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'range_call_result_311', range_call_result_311)
            # Testing if the for loop is going to be iterated (line 109)
            # Testing the type of a for loop iterable (line 109)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 12), range_call_result_311)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 12), range_call_result_311):
                # Getting the type of the for loop variable (line 109)
                for_loop_var_312 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 12), range_call_result_311)
                # Assigning a type to the variable 'y' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'y', for_loop_var_312)
                # SSA begins for a for statement (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 110)
                # Processing the call arguments (line 110)
                int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
                # Processing the call keyword arguments (line 110)
                kwargs_315 = {}
                # Getting the type of 'range' (line 110)
                range_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'range', False)
                # Calling range(args, kwargs) (line 110)
                range_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), range_313, *[int_314], **kwargs_315)
                
                # Assigning a type to the variable 'range_call_result_316' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'range_call_result_316', range_call_result_316)
                # Testing if the for loop is going to be iterated (line 110)
                # Testing the type of a for loop iterable (line 110)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_316)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_316):
                    # Getting the type of the for loop variable (line 110)
                    for_loop_var_317 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_316)
                    # Assigning a type to the variable 'x' (line 110)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'x', for_loop_var_317)
                    # SSA begins for a for statement (line 110)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Subscript to a Subscript (line 111):
                    
                    # Assigning a Subscript to a Subscript (line 111):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'y' (line 111)
                    y_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 51), 'y')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'x' (line 111)
                    x_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 48), 'x')
                    # Getting the type of 'puzzlebackup' (line 111)
                    puzzlebackup_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'puzzlebackup')
                    # Obtaining the member '__getitem__' of a type (line 111)
                    getitem___321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 35), puzzlebackup_320, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                    subscript_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 111, 35), getitem___321, x_319)
                    
                    # Obtaining the member '__getitem__' of a type (line 111)
                    getitem___323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 35), subscript_call_result_322, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                    subscript_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 111, 35), getitem___323, y_318)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'x' (line 111)
                    x_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'x')
                    # Getting the type of 'puzzle' (line 111)
                    puzzle_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'puzzle')
                    # Obtaining the member '__getitem__' of a type (line 111)
                    getitem___327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 20), puzzle_326, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                    subscript_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), getitem___327, x_325)
                    
                    # Getting the type of 'y' (line 111)
                    y_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'y')
                    # Storing an element on a container (line 111)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 20), subscript_call_result_328, (y_329, subscript_call_result_324))
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to hash_add(...): (line 112)
            # Processing the call arguments (line 112)
            # Getting the type of 'puzzle' (line 112)
            puzzle_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'puzzle', False)
            # Processing the call keyword arguments (line 112)
            kwargs_332 = {}
            # Getting the type of 'hash_add' (line 112)
            hash_add_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'hash_add', False)
            # Calling hash_add(args, kwargs) (line 112)
            hash_add_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), hash_add_330, *[puzzle_331], **kwargs_332)
            
            # Getting the type of 'False' (line 113)
            False_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'stypy_return_type', False_334)
            # SSA branch for the else part of an if statement (line 86)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 116):
            
            # Assigning a BinOp to a Name (line 116):
            # Getting the type of 'i' (line 116)
            i_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'i')
            int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
            # Applying the binary operator '*' (line 116)
            result_mul_337 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '*', i_335, int_336)
            
            # Assigning a type to the variable 'ii' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'ii', result_mul_337)
            
            # Assigning a BinOp to a Name (line 117):
            
            # Assigning a BinOp to a Name (line 117):
            # Getting the type of 'j' (line 117)
            j_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'j')
            int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'int')
            # Applying the binary operator '*' (line 117)
            result_mul_340 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 17), '*', j_338, int_339)
            
            # Assigning a type to the variable 'jj' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'jj', result_mul_340)
            
            
            # Call to range(...): (line 118)
            # Processing the call arguments (line 118)
            
            # Call to len(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'l' (line 118)
            l_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'l', False)
            # Processing the call keyword arguments (line 118)
            kwargs_344 = {}
            # Getting the type of 'len' (line 118)
            len_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'len', False)
            # Calling len(args, kwargs) (line 118)
            len_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 118, 27), len_342, *[l_343], **kwargs_344)
            
            # Processing the call keyword arguments (line 118)
            kwargs_346 = {}
            # Getting the type of 'range' (line 118)
            range_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'range', False)
            # Calling range(args, kwargs) (line 118)
            range_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), range_341, *[len_call_result_345], **kwargs_346)
            
            # Assigning a type to the variable 'range_call_result_347' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'range_call_result_347', range_call_result_347)
            # Testing if the for loop is going to be iterated (line 118)
            # Testing the type of a for loop iterable (line 118)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347):
                # Getting the type of the for loop variable (line 118)
                for_loop_var_348 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), range_call_result_347)
                # Assigning a type to the variable 'm' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'm', for_loop_var_348)
                # SSA begins for a for statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 120)
                # Processing the call arguments (line 120)
                int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'int')
                # Processing the call keyword arguments (line 120)
                kwargs_351 = {}
                # Getting the type of 'range' (line 120)
                range_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'range', False)
                # Calling range(args, kwargs) (line 120)
                range_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), range_349, *[int_350], **kwargs_351)
                
                # Assigning a type to the variable 'range_call_result_352' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'range_call_result_352', range_call_result_352)
                # Testing if the for loop is going to be iterated (line 120)
                # Testing the type of a for loop iterable (line 120)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352):
                    # Getting the type of the for loop variable (line 120)
                    for_loop_var_353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 16), range_call_result_352)
                    # Assigning a type to the variable 'y' (line 120)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'y', for_loop_var_353)
                    # SSA begins for a for statement (line 120)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to range(...): (line 121)
                    # Processing the call arguments (line 121)
                    int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'int')
                    # Processing the call keyword arguments (line 121)
                    kwargs_356 = {}
                    # Getting the type of 'range' (line 121)
                    range_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'range', False)
                    # Calling range(args, kwargs) (line 121)
                    range_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), range_354, *[int_355], **kwargs_356)
                    
                    # Assigning a type to the variable 'range_call_result_357' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'range_call_result_357', range_call_result_357)
                    # Testing if the for loop is going to be iterated (line 121)
                    # Testing the type of a for loop iterable (line 121)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357):
                        # Getting the type of the for loop variable (line 121)
                        for_loop_var_358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 20), range_call_result_357)
                        # Assigning a type to the variable 'x' (line 121)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'x', for_loop_var_358)
                        # SSA begins for a for statement (line 121)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to validMove(...): (line 122)
                        # Processing the call arguments (line 122)
                        # Getting the type of 'puzzle' (line 122)
                        puzzle_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'puzzle', False)
                        # Getting the type of 'x' (line 122)
                        x_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'x', False)
                        # Getting the type of 'ii' (line 122)
                        ii_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 49), 'ii', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 45), '+', x_361, ii_362)
                        
                        # Getting the type of 'y' (line 122)
                        y_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 53), 'y', False)
                        # Getting the type of 'jj' (line 122)
                        jj_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 57), 'jj', False)
                        # Applying the binary operator '+' (line 122)
                        result_add_366 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 53), '+', y_364, jj_365)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 122)
                        m_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 63), 'm', False)
                        # Getting the type of 'l' (line 122)
                        l_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'l', False)
                        # Obtaining the member '__getitem__' of a type (line 122)
                        getitem___369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 61), l_368, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
                        subscript_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 122, 61), getitem___369, m_367)
                        
                        # Processing the call keyword arguments (line 122)
                        kwargs_371 = {}
                        # Getting the type of 'validMove' (line 122)
                        validMove_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'validMove', False)
                        # Calling validMove(args, kwargs) (line 122)
                        validMove_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), validMove_359, *[puzzle_360, result_add_363, result_add_366, subscript_call_result_370], **kwargs_371)
                        
                        # Testing if the type of an if condition is none (line 122)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 122)
                            if_condition_373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 24), validMove_call_result_372)
                            # Assigning a type to the variable 'if_condition_373' (line 122)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'if_condition_373', if_condition_373)
                            # SSA begins for if statement (line 122)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Assigning a Subscript to a Subscript (line 123):
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'm' (line 123)
                            m_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 55), 'm')
                            # Getting the type of 'l' (line 123)
                            l_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 53), 'l')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 53), l_375, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 123, 53), getitem___376, m_374)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 123)
                            x_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'x')
                            # Getting the type of 'ii' (line 123)
                            ii_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'ii')
                            # Applying the binary operator '+' (line 123)
                            result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 35), '+', x_378, ii_379)
                            
                            # Getting the type of 'puzzle' (line 123)
                            puzzle_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 123)
                            getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), puzzle_381, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 123)
                            subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 123, 28), getitem___382, result_add_380)
                            
                            # Getting the type of 'y' (line 123)
                            y_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'y')
                            # Getting the type of 'jj' (line 123)
                            jj_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'jj')
                            # Applying the binary operator '+' (line 123)
                            result_add_386 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 43), '+', y_384, jj_385)
                            
                            # Storing an element on a container (line 123)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 28), subscript_call_result_383, (result_add_386, subscript_call_result_377))
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Assigning a Call to a Name (line 124):
                            
                            # Call to pop(...): (line 124)
                            # Processing the call arguments (line 124)
                            # Getting the type of 'm' (line 124)
                            m_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'm', False)
                            # Processing the call keyword arguments (line 124)
                            kwargs_390 = {}
                            # Getting the type of 'l' (line 124)
                            l_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'l', False)
                            # Obtaining the member 'pop' of a type (line 124)
                            pop_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 37), l_387, 'pop')
                            # Calling pop(args, kwargs) (line 124)
                            pop_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 124, 37), pop_388, *[m_389], **kwargs_390)
                            
                            # Assigning a type to the variable 'backup' (line 124)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'backup', pop_call_result_391)
                            
                            # Call to perm(...): (line 125)
                            # Processing the call arguments (line 125)
                            # Getting the type of 'puzzle' (line 125)
                            puzzle_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'puzzle', False)
                            # Getting the type of 'i' (line 125)
                            i_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 45), 'i', False)
                            # Getting the type of 'j' (line 125)
                            j_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 48), 'j', False)
                            # Getting the type of 'l' (line 125)
                            l_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 51), 'l', False)
                            # Getting the type of 'u' (line 125)
                            u_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 54), 'u', False)
                            # Processing the call keyword arguments (line 125)
                            kwargs_398 = {}
                            # Getting the type of 'perm' (line 125)
                            perm_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'perm', False)
                            # Calling perm(args, kwargs) (line 125)
                            perm_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 125, 32), perm_392, *[puzzle_393, i_394, j_395, l_396, u_397], **kwargs_398)
                            
                            # Testing if the type of an if condition is none (line 125)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399):
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                            else:
                                
                                # Testing the type of an if condition (line 125)
                                if_condition_400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 28), perm_call_result_399)
                                # Assigning a type to the variable 'if_condition_400' (line 125)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'if_condition_400', if_condition_400)
                                # SSA begins for if statement (line 125)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                # Getting the type of 'True' (line 126)
                                True_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'True')
                                # Assigning a type to the variable 'stypy_return_type' (line 126)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'stypy_return_type', True_401)
                                # SSA branch for the else part of an if statement (line 125)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to hash_add(...): (line 128)
                                # Processing the call arguments (line 128)
                                # Getting the type of 'puzzle' (line 128)
                                puzzle_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'puzzle', False)
                                # Processing the call keyword arguments (line 128)
                                kwargs_404 = {}
                                # Getting the type of 'hash_add' (line 128)
                                hash_add_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'hash_add', False)
                                # Calling hash_add(args, kwargs) (line 128)
                                hash_add_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), hash_add_402, *[puzzle_403], **kwargs_404)
                                
                                # SSA join for if statement (line 125)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Call to insert(...): (line 129)
                            # Processing the call arguments (line 129)
                            # Getting the type of 'm' (line 129)
                            m_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'm', False)
                            # Getting the type of 'backup' (line 129)
                            backup_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'backup', False)
                            # Processing the call keyword arguments (line 129)
                            kwargs_410 = {}
                            # Getting the type of 'l' (line 129)
                            l_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'l', False)
                            # Obtaining the member 'insert' of a type (line 129)
                            insert_407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 28), l_406, 'insert')
                            # Calling insert(args, kwargs) (line 129)
                            insert_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 129, 28), insert_407, *[m_408, backup_409], **kwargs_410)
                            
                            
                            # Assigning a Num to a Subscript (line 130):
                            
                            # Assigning a Num to a Subscript (line 130):
                            int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 53), 'int')
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'x' (line 130)
                            x_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'x')
                            # Getting the type of 'ii' (line 130)
                            ii_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 39), 'ii')
                            # Applying the binary operator '+' (line 130)
                            result_add_415 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '+', x_413, ii_414)
                            
                            # Getting the type of 'puzzle' (line 130)
                            puzzle_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'puzzle')
                            # Obtaining the member '__getitem__' of a type (line 130)
                            getitem___417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 28), puzzle_416, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                            subscript_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 130, 28), getitem___417, result_add_415)
                            
                            # Getting the type of 'y' (line 130)
                            y_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'y')
                            # Getting the type of 'jj' (line 130)
                            jj_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'jj')
                            # Applying the binary operator '+' (line 130)
                            result_add_421 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 43), '+', y_419, jj_420)
                            
                            # Storing an element on a container (line 130)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 28), subscript_call_result_418, (result_add_421, int_412))
                            # SSA join for if statement (line 122)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'False' (line 131)
            False_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', False_422)
            # SSA join for if statement (line 86)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'perm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'perm' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_423)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'perm'
    return stypy_return_type_423

# Assigning a type to the variable 'perm' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'perm', perm)

@norecursion
def genMoveList(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'genMoveList'
    module_type_store = module_type_store.open_function_context('genMoveList', 135, 0, False)
    
    # Passed parameters checking function
    genMoveList.stypy_localization = localization
    genMoveList.stypy_type_of_self = None
    genMoveList.stypy_type_store = module_type_store
    genMoveList.stypy_function_name = 'genMoveList'
    genMoveList.stypy_param_names_list = ['puzzle', 'i', 'j']
    genMoveList.stypy_varargs_param_name = None
    genMoveList.stypy_kwargs_param_name = None
    genMoveList.stypy_call_defaults = defaults
    genMoveList.stypy_call_varargs = varargs
    genMoveList.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'genMoveList', ['puzzle', 'i', 'j'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'genMoveList', localization, ['puzzle', 'i', 'j'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'genMoveList(...)' code ##################

    
    # Assigning a Call to a Name (line 136):
    
    # Assigning a Call to a Name (line 136):
    
    # Call to range(...): (line 136)
    # Processing the call arguments (line 136)
    int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 14), 'int')
    int_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 17), 'int')
    # Processing the call keyword arguments (line 136)
    kwargs_427 = {}
    # Getting the type of 'range' (line 136)
    range_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'range', False)
    # Calling range(args, kwargs) (line 136)
    range_call_result_428 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), range_424, *[int_425, int_426], **kwargs_427)
    
    # Assigning a type to the variable 'l' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'l', range_call_result_428)
    
    
    # Call to range(...): (line 137)
    # Processing the call arguments (line 137)
    int_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 19), 'int')
    # Processing the call keyword arguments (line 137)
    kwargs_431 = {}
    # Getting the type of 'range' (line 137)
    range_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'range', False)
    # Calling range(args, kwargs) (line 137)
    range_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 137, 13), range_429, *[int_430], **kwargs_431)
    
    # Assigning a type to the variable 'range_call_result_432' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'range_call_result_432', range_call_result_432)
    # Testing if the for loop is going to be iterated (line 137)
    # Testing the type of a for loop iterable (line 137)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 4), range_call_result_432)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 137, 4), range_call_result_432):
        # Getting the type of the for loop variable (line 137)
        for_loop_var_433 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 4), range_call_result_432)
        # Assigning a type to the variable 'y' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'y', for_loop_var_433)
        # SSA begins for a for statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 138)
        # Processing the call arguments (line 138)
        int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
        # Processing the call keyword arguments (line 138)
        kwargs_436 = {}
        # Getting the type of 'range' (line 138)
        range_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 17), 'range', False)
        # Calling range(args, kwargs) (line 138)
        range_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 138, 17), range_434, *[int_435], **kwargs_436)
        
        # Assigning a type to the variable 'range_call_result_437' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'range_call_result_437', range_call_result_437)
        # Testing if the for loop is going to be iterated (line 138)
        # Testing the type of a for loop iterable (line 138)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 8), range_call_result_437)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 138, 8), range_call_result_437):
            # Getting the type of the for loop variable (line 138)
            for_loop_var_438 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 8), range_call_result_437)
            # Assigning a type to the variable 'x' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'x', for_loop_var_438)
            # SSA begins for a for statement (line 138)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 139):
            
            # Assigning a Subscript to a Name (line 139):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 139)
            j_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'j')
            int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 38), 'int')
            # Applying the binary operator '*' (line 139)
            result_mul_441 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 34), '*', j_439, int_440)
            
            # Getting the type of 'y' (line 139)
            y_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 42), 'y')
            # Applying the binary operator '+' (line 139)
            result_add_443 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 34), '+', result_mul_441, y_442)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 139)
            i_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'i')
            int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 27), 'int')
            # Applying the binary operator '*' (line 139)
            result_mul_446 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 23), '*', i_444, int_445)
            
            # Getting the type of 'x' (line 139)
            x_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'x')
            # Applying the binary operator '+' (line 139)
            result_add_448 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 23), '+', result_mul_446, x_447)
            
            # Getting the type of 'puzzle' (line 139)
            puzzle_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'puzzle')
            # Obtaining the member '__getitem__' of a type (line 139)
            getitem___450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), puzzle_449, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 139)
            subscript_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), getitem___450, result_add_448)
            
            # Obtaining the member '__getitem__' of a type (line 139)
            getitem___452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), subscript_call_result_451, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 139)
            subscript_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), getitem___452, result_add_443)
            
            # Assigning a type to the variable 'p' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'p', subscript_call_result_453)
            
            # Getting the type of 'p' (line 140)
            p_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'p')
            int_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 20), 'int')
            # Applying the binary operator '!=' (line 140)
            result_ne_456 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 15), '!=', p_454, int_455)
            
            # Testing if the type of an if condition is none (line 140)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 140, 12), result_ne_456):
                pass
            else:
                
                # Testing the type of an if condition (line 140)
                if_condition_457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 12), result_ne_456)
                # Assigning a type to the variable 'if_condition_457' (line 140)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'if_condition_457', if_condition_457)
                # SSA begins for if statement (line 140)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to remove(...): (line 141)
                # Processing the call arguments (line 141)
                # Getting the type of 'p' (line 141)
                p_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'p', False)
                # Processing the call keyword arguments (line 141)
                kwargs_461 = {}
                # Getting the type of 'l' (line 141)
                l_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'l', False)
                # Obtaining the member 'remove' of a type (line 141)
                remove_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), l_458, 'remove')
                # Calling remove(args, kwargs) (line 141)
                remove_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), remove_459, *[p_460], **kwargs_461)
                
                # SSA join for if statement (line 140)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'l' (line 142)
    l_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'l')
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type', l_463)
    
    # ################# End of 'genMoveList(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'genMoveList' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_464)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'genMoveList'
    return stypy_return_type_464

# Assigning a type to the variable 'genMoveList' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'genMoveList', genMoveList)

@norecursion
def printpuzzle(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'printpuzzle'
    module_type_store = module_type_store.open_function_context('printpuzzle', 145, 0, False)
    
    # Passed parameters checking function
    printpuzzle.stypy_localization = localization
    printpuzzle.stypy_type_of_self = None
    printpuzzle.stypy_type_store = module_type_store
    printpuzzle.stypy_function_name = 'printpuzzle'
    printpuzzle.stypy_param_names_list = ['puzzle']
    printpuzzle.stypy_varargs_param_name = None
    printpuzzle.stypy_kwargs_param_name = None
    printpuzzle.stypy_call_defaults = defaults
    printpuzzle.stypy_call_varargs = varargs
    printpuzzle.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'printpuzzle', ['puzzle'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'printpuzzle', localization, ['puzzle'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'printpuzzle(...)' code ##################

    
    
    # Call to range(...): (line 146)
    # Processing the call arguments (line 146)
    int_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
    # Processing the call keyword arguments (line 146)
    kwargs_467 = {}
    # Getting the type of 'range' (line 146)
    range_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), 'range', False)
    # Calling range(args, kwargs) (line 146)
    range_call_result_468 = invoke(stypy.reporting.localization.Localization(__file__, 146, 13), range_465, *[int_466], **kwargs_467)
    
    # Assigning a type to the variable 'range_call_result_468' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'range_call_result_468', range_call_result_468)
    # Testing if the for loop is going to be iterated (line 146)
    # Testing the type of a for loop iterable (line 146)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 4), range_call_result_468)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 146, 4), range_call_result_468):
        # Getting the type of the for loop variable (line 146)
        for_loop_var_469 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 4), range_call_result_468)
        # Assigning a type to the variable 'x' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'x', for_loop_var_469)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Str to a Name (line 147):
        
        # Assigning a Str to a Name (line 147):
        str_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'str', ' ')
        # Assigning a type to the variable 's' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 's', str_470)
        
        
        # Call to range(...): (line 148)
        # Processing the call arguments (line 148)
        int_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'int')
        # Processing the call keyword arguments (line 148)
        kwargs_473 = {}
        # Getting the type of 'range' (line 148)
        range_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 17), 'range', False)
        # Calling range(args, kwargs) (line 148)
        range_call_result_474 = invoke(stypy.reporting.localization.Localization(__file__, 148, 17), range_471, *[int_472], **kwargs_473)
        
        # Assigning a type to the variable 'range_call_result_474' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'range_call_result_474', range_call_result_474)
        # Testing if the for loop is going to be iterated (line 148)
        # Testing the type of a for loop iterable (line 148)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 8), range_call_result_474)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 148, 8), range_call_result_474):
            # Getting the type of the for loop variable (line 148)
            for_loop_var_475 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 8), range_call_result_474)
            # Assigning a type to the variable 'y' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'y', for_loop_var_475)
            # SSA begins for a for statement (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 149):
            
            # Assigning a Subscript to a Name (line 149):
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 149)
            y_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'y')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 149)
            x_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'x')
            # Getting the type of 'puzzle' (line 149)
            puzzle_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'puzzle')
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), puzzle_478, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_480 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), getitem___479, x_477)
            
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), subscript_call_result_480, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), getitem___481, y_476)
            
            # Assigning a type to the variable 'p' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'p', subscript_call_result_482)
            
            # Getting the type of 'p' (line 150)
            p_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'p')
            int_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'int')
            # Applying the binary operator '==' (line 150)
            result_eq_485 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '==', p_483, int_484)
            
            # Testing if the type of an if condition is none (line 150)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_485):
                
                # Getting the type of 's' (line 153)
                s_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 's')
                
                # Call to str(...): (line 153)
                # Processing the call arguments (line 153)
                
                # Obtaining the type of the subscript
                # Getting the type of 'y' (line 153)
                y_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 35), 'y', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'x' (line 153)
                x_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 32), 'x', False)
                # Getting the type of 'puzzle' (line 153)
                puzzle_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'puzzle', False)
                # Obtaining the member '__getitem__' of a type (line 153)
                getitem___495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 25), puzzle_494, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                subscript_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), getitem___495, x_493)
                
                # Obtaining the member '__getitem__' of a type (line 153)
                getitem___497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 25), subscript_call_result_496, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                subscript_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), getitem___497, y_492)
                
                # Processing the call keyword arguments (line 153)
                kwargs_499 = {}
                # Getting the type of 'str' (line 153)
                str_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'str', False)
                # Calling str(args, kwargs) (line 153)
                str_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), str_491, *[subscript_call_result_498], **kwargs_499)
                
                # Applying the binary operator '+=' (line 153)
                result_iadd_501 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), '+=', s_490, str_call_result_500)
                # Assigning a type to the variable 's' (line 153)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 's', result_iadd_501)
                
            else:
                
                # Testing the type of an if condition (line 150)
                if_condition_486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_485)
                # Assigning a type to the variable 'if_condition_486' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_486', if_condition_486)
                # SSA begins for if statement (line 150)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 's' (line 151)
                s_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 's')
                str_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 21), 'str', '.')
                # Applying the binary operator '+=' (line 151)
                result_iadd_489 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 16), '+=', s_487, str_488)
                # Assigning a type to the variable 's' (line 151)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 's', result_iadd_489)
                
                # SSA branch for the else part of an if statement (line 150)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 's' (line 153)
                s_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 's')
                
                # Call to str(...): (line 153)
                # Processing the call arguments (line 153)
                
                # Obtaining the type of the subscript
                # Getting the type of 'y' (line 153)
                y_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 35), 'y', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'x' (line 153)
                x_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 32), 'x', False)
                # Getting the type of 'puzzle' (line 153)
                puzzle_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'puzzle', False)
                # Obtaining the member '__getitem__' of a type (line 153)
                getitem___495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 25), puzzle_494, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                subscript_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), getitem___495, x_493)
                
                # Obtaining the member '__getitem__' of a type (line 153)
                getitem___497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 25), subscript_call_result_496, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                subscript_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), getitem___497, y_492)
                
                # Processing the call keyword arguments (line 153)
                kwargs_499 = {}
                # Getting the type of 'str' (line 153)
                str_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'str', False)
                # Calling str(args, kwargs) (line 153)
                str_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), str_491, *[subscript_call_result_498], **kwargs_499)
                
                # Applying the binary operator '+=' (line 153)
                result_iadd_501 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), '+=', s_490, str_call_result_500)
                # Assigning a type to the variable 's' (line 153)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 's', result_iadd_501)
                
                # SSA join for if statement (line 150)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 's' (line 154)
            s_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 's')
            str_503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'str', ' ')
            # Applying the binary operator '+=' (line 154)
            result_iadd_504 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 12), '+=', s_502, str_503)
            # Assigning a type to the variable 's' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 's', result_iadd_504)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'printpuzzle(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'printpuzzle' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_505)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'printpuzzle'
    return stypy_return_type_505

# Assigning a type to the variable 'printpuzzle' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'printpuzzle', printpuzzle)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 159, 0, False)
    
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

    
    # Assigning a List to a Name (line 160):
    
    # Assigning a List to a Name (line 160):
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    # Adding element type (line 160)
    int_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_508)
    # Adding element type (line 160)
    int_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_509)
    # Adding element type (line 160)
    int_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_510)
    # Adding element type (line 160)
    int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_511)
    # Adding element type (line 160)
    int_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_512)
    # Adding element type (line 160)
    int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_513)
    # Adding element type (line 160)
    int_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_514)
    # Adding element type (line 160)
    int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_515)
    # Adding element type (line 160)
    int_516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 14), list_507, int_516)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_507)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    int_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_518)
    # Adding element type (line 161)
    int_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_519)
    # Adding element type (line 161)
    int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_520)
    # Adding element type (line 161)
    int_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_521)
    # Adding element type (line 161)
    int_522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_522)
    # Adding element type (line 161)
    int_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_523)
    # Adding element type (line 161)
    int_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_524)
    # Adding element type (line 161)
    int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_525)
    # Adding element type (line 161)
    int_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 14), list_517, int_526)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_517)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 162)
    list_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 162)
    # Adding element type (line 162)
    int_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_528)
    # Adding element type (line 162)
    int_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_529)
    # Adding element type (line 162)
    int_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_530)
    # Adding element type (line 162)
    int_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_531)
    # Adding element type (line 162)
    int_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_532)
    # Adding element type (line 162)
    int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_533)
    # Adding element type (line 162)
    int_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_534)
    # Adding element type (line 162)
    int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_535)
    # Adding element type (line 162)
    int_536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 14), list_527, int_536)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_527)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_538)
    # Adding element type (line 163)
    int_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_539)
    # Adding element type (line 163)
    int_540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_540)
    # Adding element type (line 163)
    int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_541)
    # Adding element type (line 163)
    int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_542)
    # Adding element type (line 163)
    int_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_543)
    # Adding element type (line 163)
    int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_544)
    # Adding element type (line 163)
    int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_545)
    # Adding element type (line 163)
    int_546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 14), list_537, int_546)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_537)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 164)
    list_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 164)
    # Adding element type (line 164)
    int_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_548)
    # Adding element type (line 164)
    int_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_549)
    # Adding element type (line 164)
    int_550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_550)
    # Adding element type (line 164)
    int_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_551)
    # Adding element type (line 164)
    int_552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_552)
    # Adding element type (line 164)
    int_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_553)
    # Adding element type (line 164)
    int_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_554)
    # Adding element type (line 164)
    int_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_555)
    # Adding element type (line 164)
    int_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 14), list_547, int_556)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_547)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 165)
    list_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 165)
    # Adding element type (line 165)
    int_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_558)
    # Adding element type (line 165)
    int_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_559)
    # Adding element type (line 165)
    int_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_560)
    # Adding element type (line 165)
    int_561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_561)
    # Adding element type (line 165)
    int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_562)
    # Adding element type (line 165)
    int_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_563)
    # Adding element type (line 165)
    int_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_564)
    # Adding element type (line 165)
    int_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_565)
    # Adding element type (line 165)
    int_566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 14), list_557, int_566)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_557)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 166)
    list_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 166)
    # Adding element type (line 166)
    int_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_568)
    # Adding element type (line 166)
    int_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_569)
    # Adding element type (line 166)
    int_570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_570)
    # Adding element type (line 166)
    int_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_571)
    # Adding element type (line 166)
    int_572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_572)
    # Adding element type (line 166)
    int_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_573)
    # Adding element type (line 166)
    int_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_574)
    # Adding element type (line 166)
    int_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_575)
    # Adding element type (line 166)
    int_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 14), list_567, int_576)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_567)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    # Adding element type (line 167)
    int_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_578)
    # Adding element type (line 167)
    int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_579)
    # Adding element type (line 167)
    int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_580)
    # Adding element type (line 167)
    int_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_581)
    # Adding element type (line 167)
    int_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_582)
    # Adding element type (line 167)
    int_583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_583)
    # Adding element type (line 167)
    int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_584)
    # Adding element type (line 167)
    int_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_585)
    # Adding element type (line 167)
    int_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 14), list_577, int_586)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_577)
    # Adding element type (line 160)
    
    # Obtaining an instance of the builtin type 'list' (line 168)
    list_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 168)
    # Adding element type (line 168)
    int_588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_588)
    # Adding element type (line 168)
    int_589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_589)
    # Adding element type (line 168)
    int_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_590)
    # Adding element type (line 168)
    int_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_591)
    # Adding element type (line 168)
    int_592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_592)
    # Adding element type (line 168)
    int_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_593)
    # Adding element type (line 168)
    int_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_594)
    # Adding element type (line 168)
    int_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_595)
    # Adding element type (line 168)
    int_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 14), list_587, int_596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 13), list_506, list_587)
    
    # Assigning a type to the variable 'puzzle' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'puzzle', list_506)
    
    # Assigning a List to a Name (line 171):
    
    # Assigning a List to a Name (line 171):
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    
    # Assigning a type to the variable 'u' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'u', list_597)
    
    # Assigning a List to a Name (line 172):
    
    # Assigning a List to a Name (line 172):
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    
    # Assigning a type to the variable 'lcount' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'lcount', list_598)
    
    
    # Call to range(...): (line 173)
    # Processing the call arguments (line 173)
    int_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'int')
    # Processing the call keyword arguments (line 173)
    kwargs_601 = {}
    # Getting the type of 'range' (line 173)
    range_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'range', False)
    # Calling range(args, kwargs) (line 173)
    range_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), range_599, *[int_600], **kwargs_601)
    
    # Assigning a type to the variable 'range_call_result_602' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'range_call_result_602', range_call_result_602)
    # Testing if the for loop is going to be iterated (line 173)
    # Testing the type of a for loop iterable (line 173)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_602)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_602):
        # Getting the type of the for loop variable (line 173)
        for_loop_var_603 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_602)
        # Assigning a type to the variable 'y' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'y', for_loop_var_603)
        # SSA begins for a for statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 174)
        # Processing the call arguments (line 174)
        int_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 23), 'int')
        # Processing the call keyword arguments (line 174)
        kwargs_606 = {}
        # Getting the type of 'range' (line 174)
        range_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'range', False)
        # Calling range(args, kwargs) (line 174)
        range_call_result_607 = invoke(stypy.reporting.localization.Localization(__file__, 174, 17), range_604, *[int_605], **kwargs_606)
        
        # Assigning a type to the variable 'range_call_result_607' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'range_call_result_607', range_call_result_607)
        # Testing if the for loop is going to be iterated (line 174)
        # Testing the type of a for loop iterable (line 174)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 174, 8), range_call_result_607)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 174, 8), range_call_result_607):
            # Getting the type of the for loop variable (line 174)
            for_loop_var_608 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 174, 8), range_call_result_607)
            # Assigning a type to the variable 'x' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'x', for_loop_var_608)
            # SSA begins for a for statement (line 174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 175)
            # Processing the call arguments (line 175)
            
            # Obtaining an instance of the builtin type 'tuple' (line 175)
            tuple_611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 22), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 175)
            # Adding element type (line 175)
            # Getting the type of 'x' (line 175)
            x_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 22), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 22), tuple_611, x_612)
            # Adding element type (line 175)
            # Getting the type of 'y' (line 175)
            y_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'y', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 22), tuple_611, y_613)
            
            # Processing the call keyword arguments (line 175)
            kwargs_614 = {}
            # Getting the type of 'u' (line 175)
            u_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'u', False)
            # Obtaining the member 'append' of a type (line 175)
            append_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), u_609, 'append')
            # Calling append(args, kwargs) (line 175)
            append_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), append_610, *[tuple_611], **kwargs_614)
            
            
            # Call to append(...): (line 176)
            # Processing the call arguments (line 176)
            
            # Call to len(...): (line 176)
            # Processing the call arguments (line 176)
            
            # Call to genMoveList(...): (line 176)
            # Processing the call arguments (line 176)
            # Getting the type of 'puzzle' (line 176)
            puzzle_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'puzzle', False)
            # Getting the type of 'x' (line 176)
            x_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 50), 'x', False)
            # Getting the type of 'y' (line 176)
            y_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 53), 'y', False)
            # Processing the call keyword arguments (line 176)
            kwargs_623 = {}
            # Getting the type of 'genMoveList' (line 176)
            genMoveList_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'genMoveList', False)
            # Calling genMoveList(args, kwargs) (line 176)
            genMoveList_call_result_624 = invoke(stypy.reporting.localization.Localization(__file__, 176, 30), genMoveList_619, *[puzzle_620, x_621, y_622], **kwargs_623)
            
            # Processing the call keyword arguments (line 176)
            kwargs_625 = {}
            # Getting the type of 'len' (line 176)
            len_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'len', False)
            # Calling len(args, kwargs) (line 176)
            len_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 176, 26), len_618, *[genMoveList_call_result_624], **kwargs_625)
            
            # Processing the call keyword arguments (line 176)
            kwargs_627 = {}
            # Getting the type of 'lcount' (line 176)
            lcount_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'lcount', False)
            # Obtaining the member 'append' of a type (line 176)
            append_617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), lcount_616, 'append')
            # Calling append(args, kwargs) (line 176)
            append_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), append_617, *[len_call_result_626], **kwargs_627)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to range(...): (line 179)
    # Processing the call arguments (line 179)
    int_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 19), 'int')
    int_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'int')
    # Processing the call keyword arguments (line 179)
    kwargs_632 = {}
    # Getting the type of 'range' (line 179)
    range_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'range', False)
    # Calling range(args, kwargs) (line 179)
    range_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 179, 13), range_629, *[int_630, int_631], **kwargs_632)
    
    # Assigning a type to the variable 'range_call_result_633' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'range_call_result_633', range_call_result_633)
    # Testing if the for loop is going to be iterated (line 179)
    # Testing the type of a for loop iterable (line 179)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 179, 4), range_call_result_633)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 179, 4), range_call_result_633):
        # Getting the type of the for loop variable (line 179)
        for_loop_var_634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 179, 4), range_call_result_633)
        # Assigning a type to the variable 'j' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'j', for_loop_var_634)
        # SSA begins for a for statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'j' (line 180)
        j_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'j', False)
        int_637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 26), 'int')
        # Processing the call keyword arguments (line 180)
        kwargs_638 = {}
        # Getting the type of 'range' (line 180)
        range_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'range', False)
        # Calling range(args, kwargs) (line 180)
        range_call_result_639 = invoke(stypy.reporting.localization.Localization(__file__, 180, 17), range_635, *[j_636, int_637], **kwargs_638)
        
        # Assigning a type to the variable 'range_call_result_639' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'range_call_result_639', range_call_result_639)
        # Testing if the for loop is going to be iterated (line 180)
        # Testing the type of a for loop iterable (line 180)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 8), range_call_result_639)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 180, 8), range_call_result_639):
            # Getting the type of the for loop variable (line 180)
            for_loop_var_640 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 8), range_call_result_639)
            # Assigning a type to the variable 'i' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'i', for_loop_var_640)
            # SSA begins for a for statement (line 180)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'i' (line 181)
            i_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'i')
            # Getting the type of 'j' (line 181)
            j_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'j')
            # Applying the binary operator '!=' (line 181)
            result_ne_643 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 15), '!=', i_641, j_642)
            
            # Testing if the type of an if condition is none (line 181)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 181, 12), result_ne_643):
                pass
            else:
                
                # Testing the type of an if condition (line 181)
                if_condition_644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 12), result_ne_643)
                # Assigning a type to the variable 'if_condition_644' (line 181)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'if_condition_644', if_condition_644)
                # SSA begins for if statement (line 181)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 182)
                i_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'i')
                # Getting the type of 'lcount' (line 182)
                lcount_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'lcount')
                # Obtaining the member '__getitem__' of a type (line 182)
                getitem___647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 19), lcount_646, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 182)
                subscript_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 182, 19), getitem___647, i_645)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 182)
                j_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 38), 'j')
                # Getting the type of 'lcount' (line 182)
                lcount_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'lcount')
                # Obtaining the member '__getitem__' of a type (line 182)
                getitem___651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 31), lcount_650, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 182)
                subscript_call_result_652 = invoke(stypy.reporting.localization.Localization(__file__, 182, 31), getitem___651, j_649)
                
                # Applying the binary operator '<' (line 182)
                result_lt_653 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 19), '<', subscript_call_result_648, subscript_call_result_652)
                
                # Testing if the type of an if condition is none (line 182)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 182, 16), result_lt_653):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 182)
                    if_condition_654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 16), result_lt_653)
                    # Assigning a type to the variable 'if_condition_654' (line 182)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'if_condition_654', if_condition_654)
                    # SSA begins for if statement (line 182)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Tuple to a Tuple (line 183):
                    
                    # Assigning a Subscript to a Name (line 183):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 183)
                    j_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 35), 'j')
                    # Getting the type of 'u' (line 183)
                    u_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'u')
                    # Obtaining the member '__getitem__' of a type (line 183)
                    getitem___657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 33), u_656, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
                    subscript_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 183, 33), getitem___657, j_655)
                    
                    # Assigning a type to the variable 'tuple_assignment_4' (line 183)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'tuple_assignment_4', subscript_call_result_658)
                    
                    # Assigning a Subscript to a Name (line 183):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 183)
                    i_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 41), 'i')
                    # Getting the type of 'u' (line 183)
                    u_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'u')
                    # Obtaining the member '__getitem__' of a type (line 183)
                    getitem___661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 39), u_660, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
                    subscript_call_result_662 = invoke(stypy.reporting.localization.Localization(__file__, 183, 39), getitem___661, i_659)
                    
                    # Assigning a type to the variable 'tuple_assignment_5' (line 183)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'tuple_assignment_5', subscript_call_result_662)
                    
                    # Assigning a Name to a Subscript (line 183):
                    # Getting the type of 'tuple_assignment_4' (line 183)
                    tuple_assignment_4_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'tuple_assignment_4')
                    # Getting the type of 'u' (line 183)
                    u_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'u')
                    # Getting the type of 'i' (line 183)
                    i_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'i')
                    # Storing an element on a container (line 183)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 20), u_664, (i_665, tuple_assignment_4_663))
                    
                    # Assigning a Name to a Subscript (line 183):
                    # Getting the type of 'tuple_assignment_5' (line 183)
                    tuple_assignment_5_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'tuple_assignment_5')
                    # Getting the type of 'u' (line 183)
                    u_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'u')
                    # Getting the type of 'j' (line 183)
                    j_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'j')
                    # Storing an element on a container (line 183)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 26), u_667, (j_668, tuple_assignment_5_666))
                    
                    # Assigning a Tuple to a Tuple (line 184):
                    
                    # Assigning a Subscript to a Name (line 184):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 184)
                    j_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 50), 'j')
                    # Getting the type of 'lcount' (line 184)
                    lcount_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 43), 'lcount')
                    # Obtaining the member '__getitem__' of a type (line 184)
                    getitem___671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 43), lcount_670, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
                    subscript_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 184, 43), getitem___671, j_669)
                    
                    # Assigning a type to the variable 'tuple_assignment_6' (line 184)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_assignment_6', subscript_call_result_672)
                    
                    # Assigning a Subscript to a Name (line 184):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 184)
                    i_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 61), 'i')
                    # Getting the type of 'lcount' (line 184)
                    lcount_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 54), 'lcount')
                    # Obtaining the member '__getitem__' of a type (line 184)
                    getitem___675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 54), lcount_674, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
                    subscript_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 184, 54), getitem___675, i_673)
                    
                    # Assigning a type to the variable 'tuple_assignment_7' (line 184)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_assignment_7', subscript_call_result_676)
                    
                    # Assigning a Name to a Subscript (line 184):
                    # Getting the type of 'tuple_assignment_6' (line 184)
                    tuple_assignment_6_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_assignment_6')
                    # Getting the type of 'lcount' (line 184)
                    lcount_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'lcount')
                    # Getting the type of 'i' (line 184)
                    i_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'i')
                    # Storing an element on a container (line 184)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), lcount_678, (i_679, tuple_assignment_6_677))
                    
                    # Assigning a Name to a Subscript (line 184):
                    # Getting the type of 'tuple_assignment_7' (line 184)
                    tuple_assignment_7_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_assignment_7')
                    # Getting the type of 'lcount' (line 184)
                    lcount_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 31), 'lcount')
                    # Getting the type of 'j' (line 184)
                    j_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 38), 'j')
                    # Storing an element on a container (line 184)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 31), lcount_681, (j_682, tuple_assignment_7_680))
                    # SSA join for if statement (line 182)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 181)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to genMoveList(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'puzzle' (line 186)
    puzzle_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'puzzle', False)
    int_685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 28), 'int')
    int_686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'int')
    # Processing the call keyword arguments (line 186)
    kwargs_687 = {}
    # Getting the type of 'genMoveList' (line 186)
    genMoveList_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'genMoveList', False)
    # Calling genMoveList(args, kwargs) (line 186)
    genMoveList_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), genMoveList_683, *[puzzle_684, int_685, int_686], **kwargs_687)
    
    # Assigning a type to the variable 'l' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'l', genMoveList_call_result_688)
    
    # Call to perm(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'puzzle' (line 187)
    puzzle_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 9), 'puzzle', False)
    int_691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 17), 'int')
    int_692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'int')
    # Getting the type of 'l' (line 187)
    l_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'l', False)
    # Getting the type of 'u' (line 187)
    u_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'u', False)
    # Processing the call keyword arguments (line 187)
    kwargs_695 = {}
    # Getting the type of 'perm' (line 187)
    perm_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'perm', False)
    # Calling perm(args, kwargs) (line 187)
    perm_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), perm_689, *[puzzle_690, int_691, int_692, l_693, u_694], **kwargs_695)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 159)
    stypy_return_type_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_697

# Assigning a type to the variable 'main' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 190, 0, False)
    
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

    
    # Assigning a Num to a Name (line 191):
    
    # Assigning a Num to a Name (line 191):
    int_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'int')
    # Assigning a type to the variable 'iterations' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'iterations', int_698)
    
    
    # Call to range(...): (line 192)
    # Processing the call arguments (line 192)
    int_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 19), 'int')
    # Processing the call keyword arguments (line 192)
    kwargs_701 = {}
    # Getting the type of 'range' (line 192)
    range_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 'range', False)
    # Calling range(args, kwargs) (line 192)
    range_call_result_702 = invoke(stypy.reporting.localization.Localization(__file__, 192, 13), range_699, *[int_700], **kwargs_701)
    
    # Assigning a type to the variable 'range_call_result_702' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'range_call_result_702', range_call_result_702)
    # Testing if the for loop is going to be iterated (line 192)
    # Testing the type of a for loop iterable (line 192)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 192, 4), range_call_result_702)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 192, 4), range_call_result_702):
        # Getting the type of the for loop variable (line 192)
        for_loop_var_703 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 192, 4), range_call_result_702)
        # Assigning a type to the variable 'x' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'x', for_loop_var_703)
        # SSA begins for a for statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to main(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_705 = {}
        # Getting the type of 'main' (line 193)
        main_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'main', False)
        # Calling main(args, kwargs) (line 193)
        main_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), main_704, *[], **kwargs_705)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 194)
    True_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type', True_707)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_708

# Assigning a type to the variable 'run' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'run', run)

# Call to run(...): (line 197)
# Processing the call keyword arguments (line 197)
kwargs_710 = {}
# Getting the type of 'run' (line 197)
run_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'run', False)
# Calling run(args, kwargs) (line 197)
run_call_result_711 = invoke(stypy.reporting.localization.Localization(__file__, 197, 0), run_709, *[], **kwargs_710)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # connect four / four-in-a-row 
2: # http://users.softlab.ece.ntua.gr/~ttsiod/score4.html
3: 
4: from sys import argv
5: 
6: WIDTH = 7
7: HEIGHT = 6
8: ORANGE_WINS = 1000000
9: YELLOW_WINS = -ORANGE_WINS
10: 
11: g_max_depth = 7
12: g_debug = False
13: 
14: 
15: class Cell:
16:     Barren = 0
17:     Orange = 1
18:     Yellow = -1
19: 
20: 
21: def score_board(board):
22:     counters = [0] * 9
23: 
24:     # Horizontal spans
25:     for y in xrange(HEIGHT):
26:         score = board[y][0] + board[y][1] + board[y][2]
27:         for x in xrange(3, WIDTH):
28:             score += board[y][x]
29:             counters[score + 4] += 1
30:             score -= board[y][x - 3]
31: 
32:     # Vertical spans
33:     for x in xrange(WIDTH):
34:         score = board[0][x] + board[1][x] + board[2][x]
35:         for y in xrange(3, HEIGHT):
36:             score += board[y][x]
37:             counters[score + 4] += 1
38:             score -= board[y - 3][x]
39: 
40:     # Down-right (and up-left) diagonals
41:     for y in xrange(HEIGHT - 3):
42:         for x in xrange(WIDTH - 3):
43:             score = 0
44:             for idx in xrange(4):
45:                 yy = y + idx
46:                 xx = x + idx
47:                 score += board[yy][xx]
48:             counters[score + 4] += 1
49: 
50:     # up-right (and down-left) diagonals
51:     for y in xrange(3, HEIGHT):
52:         for x in xrange(WIDTH - 3):
53:             score = 0
54:             for idx in xrange(4):
55:                 yy = y - idx
56:                 xx = x + idx
57:                 score += board[yy][xx]
58:             counters[score + 4] += 1
59: 
60:     if counters[0] != 0:
61:         return YELLOW_WINS
62:     elif counters[8] != 0:
63:         return ORANGE_WINS
64:     else:
65:         return (counters[5] + 2 * counters[6] + 5 * counters[7] +
66:                 10 * counters[8] - counters[3] - 2 * counters[2] -
67:                 5 * counters[1] - 10 * counters[0])
68: 
69: 
70: def drop_disk(board, column, color):
71:     for y in xrange(HEIGHT - 1, -1, -1):
72:         if board[y][column] == Cell.Barren:
73:             board[y][column] = color
74:             return y
75:     return -1
76: 
77: 
78: def load_board(args):
79:     global g_debug, g_max_depth
80:     new_board = [[Cell.Barren] * WIDTH for _ in xrange(HEIGHT)]
81: 
82:     for i, arg in enumerate(args[1:]):
83:         if arg[0] == 'o' or arg[0] == 'y':
84:             new_board[ord(arg[1]) - ord('0')][ord(arg[2]) - ord('0')] = \
85:                 Cell.Orange if arg[0] == 'o' else Cell.Yellow
86:         elif arg == "-debug":
87:             g_debug = True
88:         elif arg == "-level" and i < (len(args) - 2):
89:             g_max_depth = int(args[i + 2])
90: 
91:     return new_board
92: 
93: 
94: def ab_minimax(maximize_or_minimize, color, depth, board):
95:     global g_max_depth, g_debug
96:     if depth == 0:
97:         return (-1, score_board(board))
98:     else:
99:         best_score = -10000000 if maximize_or_minimize else 10000000
100:         bestMove = -1
101:         for column in xrange(WIDTH):
102:             if board[0][column] != Cell.Barren:
103:                 continue
104:             rowFilled = drop_disk(board, column, color)
105:             if rowFilled == -1:
106:                 continue
107:             s = score_board(board)
108:             if s == (ORANGE_WINS if maximize_or_minimize else YELLOW_WINS):
109:                 bestMove = column
110:                 best_score = s
111:                 board[rowFilled][column] = Cell.Barren
112:                 break
113: 
114:             move, score = ab_minimax(not maximize_or_minimize,
115:                                      Cell.Yellow if color == Cell.Orange else Cell.Orange,
116:                                      depth - 1, board)
117:             board[rowFilled][column] = Cell.Barren
118:             if depth == g_max_depth and g_debug:
119:                 pass  # print "Depth %d, placing on %d, score:%d" % (depth, column, score)
120:             if maximize_or_minimize:
121:                 if score >= best_score:
122:                     best_score = score
123:                     bestMove = column
124:             else:
125:                 if score <= best_score:
126:                     best_score = score
127:                     bestMove = column
128: 
129:         return (bestMove, best_score)
130: 
131: 
132: def main(args):
133:     global g_max_depth
134:     board = load_board(args)
135:     score_orig = score_board(board)
136: 
137:     if score_orig == ORANGE_WINS:
138:         # print "I win."
139:         return -1
140:     elif score_orig == YELLOW_WINS:
141:         # print "You win."
142:         return -1
143:     else:
144:         move, score = ab_minimax(True, Cell.Orange, g_max_depth, board)
145: 
146:         if move != -1:
147:             # print move
148:             drop_disk(board, move, Cell.Orange)
149:             score_orig = score_board(board)
150:             if score_orig == ORANGE_WINS:
151:                 # print "I win."
152:                 return -1
153:             elif score_orig == YELLOW_WINS:
154:                 # print "You win."
155:                 return -1
156:             else:
157:                 return 0
158:         else:
159:             # print "No move possible."
160:             return -1
161: 
162: 
163: def run():
164:     main(["score4", "o53", "y43"])
165:     return True
166: 
167: 
168: run()
169: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from sys import argv' statement (line 4)
try:
    from sys import argv

except:
    argv = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', None, module_type_store, ['argv'], [argv])


# Assigning a Num to a Name (line 6):

# Assigning a Num to a Name (line 6):
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
# Assigning a type to the variable 'WIDTH' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'WIDTH', int_7)

# Assigning a Num to a Name (line 7):

# Assigning a Num to a Name (line 7):
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 9), 'int')
# Assigning a type to the variable 'HEIGHT' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'HEIGHT', int_8)

# Assigning a Num to a Name (line 8):

# Assigning a Num to a Name (line 8):
int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
# Assigning a type to the variable 'ORANGE_WINS' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'ORANGE_WINS', int_9)

# Assigning a UnaryOp to a Name (line 9):

# Assigning a UnaryOp to a Name (line 9):

# Getting the type of 'ORANGE_WINS' (line 9)
ORANGE_WINS_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'ORANGE_WINS')
# Applying the 'usub' unary operator (line 9)
result___neg___11 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 14), 'usub', ORANGE_WINS_10)

# Assigning a type to the variable 'YELLOW_WINS' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'YELLOW_WINS', result___neg___11)

# Assigning a Num to a Name (line 11):

# Assigning a Num to a Name (line 11):
int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
# Assigning a type to the variable 'g_max_depth' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'g_max_depth', int_12)

# Assigning a Name to a Name (line 12):

# Assigning a Name to a Name (line 12):
# Getting the type of 'False' (line 12)
False_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'False')
# Assigning a type to the variable 'g_debug' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'g_debug', False_13)
# Declaration of the 'Cell' class

class Cell:
    
    # Assigning a Num to a Name (line 16):
    
    # Assigning a Num to a Name (line 17):
    
    # Assigning a Num to a Name (line 18):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Cell' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Cell', Cell)

# Assigning a Num to a Name (line 16):
int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
# Getting the type of 'Cell'
Cell_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cell')
# Setting the type of the member 'Barren' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cell_15, 'Barren', int_14)

# Assigning a Num to a Name (line 17):
int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'int')
# Getting the type of 'Cell'
Cell_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cell')
# Setting the type of the member 'Orange' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cell_17, 'Orange', int_16)

# Assigning a Num to a Name (line 18):
int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'int')
# Getting the type of 'Cell'
Cell_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cell')
# Setting the type of the member 'Yellow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cell_19, 'Yellow', int_18)

@norecursion
def score_board(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'score_board'
    module_type_store = module_type_store.open_function_context('score_board', 21, 0, False)
    
    # Passed parameters checking function
    score_board.stypy_localization = localization
    score_board.stypy_type_of_self = None
    score_board.stypy_type_store = module_type_store
    score_board.stypy_function_name = 'score_board'
    score_board.stypy_param_names_list = ['board']
    score_board.stypy_varargs_param_name = None
    score_board.stypy_kwargs_param_name = None
    score_board.stypy_call_defaults = defaults
    score_board.stypy_call_varargs = varargs
    score_board.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'score_board', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'score_board', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'score_board(...)' code ##################

    
    # Assigning a BinOp to a Name (line 22):
    
    # Assigning a BinOp to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_20, int_21)
    
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    # Applying the binary operator '*' (line 22)
    result_mul_23 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '*', list_20, int_22)
    
    # Assigning a type to the variable 'counters' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'counters', result_mul_23)
    
    
    # Call to xrange(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'HEIGHT' (line 25)
    HEIGHT_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'HEIGHT', False)
    # Processing the call keyword arguments (line 25)
    kwargs_26 = {}
    # Getting the type of 'xrange' (line 25)
    xrange_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 25)
    xrange_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), xrange_24, *[HEIGHT_25], **kwargs_26)
    
    # Assigning a type to the variable 'xrange_call_result_27' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'xrange_call_result_27', xrange_call_result_27)
    # Testing if the for loop is going to be iterated (line 25)
    # Testing the type of a for loop iterable (line 25)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 4), xrange_call_result_27)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 25, 4), xrange_call_result_27):
        # Getting the type of the for loop variable (line 25)
        for_loop_var_28 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 4), xrange_call_result_27)
        # Assigning a type to the variable 'y' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'y', for_loop_var_28)
        # SSA begins for a for statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 26):
        
        # Assigning a BinOp to a Name (line 26):
        
        # Obtaining the type of the subscript
        int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 26)
        y_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'y')
        # Getting the type of 'board' (line 26)
        board_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'board')
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), board_31, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), getitem___32, y_30)
        
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), subscript_call_result_33, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), getitem___34, int_29)
        
        
        # Obtaining the type of the subscript
        int_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 26)
        y_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'y')
        # Getting the type of 'board' (line 26)
        board_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'board')
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 30), board_38, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 26, 30), getitem___39, y_37)
        
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 30), subscript_call_result_40, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 26, 30), getitem___41, int_36)
        
        # Applying the binary operator '+' (line 26)
        result_add_43 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 16), '+', subscript_call_result_35, subscript_call_result_42)
        
        
        # Obtaining the type of the subscript
        int_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 53), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 26)
        y_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 50), 'y')
        # Getting the type of 'board' (line 26)
        board_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 44), 'board')
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 44), board_46, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 26, 44), getitem___47, y_45)
        
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 44), subscript_call_result_48, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 26, 44), getitem___49, int_44)
        
        # Applying the binary operator '+' (line 26)
        result_add_51 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 42), '+', result_add_43, subscript_call_result_50)
        
        # Assigning a type to the variable 'score' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'score', result_add_51)
        
        
        # Call to xrange(...): (line 27)
        # Processing the call arguments (line 27)
        int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'int')
        # Getting the type of 'WIDTH' (line 27)
        WIDTH_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'WIDTH', False)
        # Processing the call keyword arguments (line 27)
        kwargs_55 = {}
        # Getting the type of 'xrange' (line 27)
        xrange_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 27)
        xrange_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 27, 17), xrange_52, *[int_53, WIDTH_54], **kwargs_55)
        
        # Assigning a type to the variable 'xrange_call_result_56' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'xrange_call_result_56', xrange_call_result_56)
        # Testing if the for loop is going to be iterated (line 27)
        # Testing the type of a for loop iterable (line 27)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 8), xrange_call_result_56)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 8), xrange_call_result_56):
            # Getting the type of the for loop variable (line 27)
            for_loop_var_57 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 8), xrange_call_result_56)
            # Assigning a type to the variable 'x' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'x', for_loop_var_57)
            # SSA begins for a for statement (line 27)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'score' (line 28)
            score_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'score')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 28)
            x_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'x')
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 28)
            y_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 27), 'y')
            # Getting the type of 'board' (line 28)
            board_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'board')
            # Obtaining the member '__getitem__' of a type (line 28)
            getitem___62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 21), board_61, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 28)
            subscript_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 28, 21), getitem___62, y_60)
            
            # Obtaining the member '__getitem__' of a type (line 28)
            getitem___64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 21), subscript_call_result_63, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 28)
            subscript_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 28, 21), getitem___64, x_59)
            
            # Applying the binary operator '+=' (line 28)
            result_iadd_66 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 12), '+=', score_58, subscript_call_result_65)
            # Assigning a type to the variable 'score' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'score', result_iadd_66)
            
            
            # Getting the type of 'counters' (line 29)
            counters_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'counters')
            
            # Obtaining the type of the subscript
            # Getting the type of 'score' (line 29)
            score_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'score')
            int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'int')
            # Applying the binary operator '+' (line 29)
            result_add_70 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), '+', score_68, int_69)
            
            # Getting the type of 'counters' (line 29)
            counters_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'counters')
            # Obtaining the member '__getitem__' of a type (line 29)
            getitem___72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), counters_71, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 29)
            subscript_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), getitem___72, result_add_70)
            
            int_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'int')
            # Applying the binary operator '+=' (line 29)
            result_iadd_75 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 12), '+=', subscript_call_result_73, int_74)
            # Getting the type of 'counters' (line 29)
            counters_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'counters')
            # Getting the type of 'score' (line 29)
            score_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'score')
            int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'int')
            # Applying the binary operator '+' (line 29)
            result_add_79 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), '+', score_77, int_78)
            
            # Storing an element on a container (line 29)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), counters_76, (result_add_79, result_iadd_75))
            
            
            # Getting the type of 'score' (line 30)
            score_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'score')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 30)
            x_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'x')
            int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'int')
            # Applying the binary operator '-' (line 30)
            result_sub_83 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 30), '-', x_81, int_82)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 30)
            y_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'y')
            # Getting the type of 'board' (line 30)
            board_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'board')
            # Obtaining the member '__getitem__' of a type (line 30)
            getitem___86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), board_85, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 30)
            subscript_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), getitem___86, y_84)
            
            # Obtaining the member '__getitem__' of a type (line 30)
            getitem___88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), subscript_call_result_87, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 30)
            subscript_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), getitem___88, result_sub_83)
            
            # Applying the binary operator '-=' (line 30)
            result_isub_90 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '-=', score_80, subscript_call_result_89)
            # Assigning a type to the variable 'score' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'score', result_isub_90)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to xrange(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'WIDTH' (line 33)
    WIDTH_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'WIDTH', False)
    # Processing the call keyword arguments (line 33)
    kwargs_93 = {}
    # Getting the type of 'xrange' (line 33)
    xrange_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 33)
    xrange_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), xrange_91, *[WIDTH_92], **kwargs_93)
    
    # Assigning a type to the variable 'xrange_call_result_94' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'xrange_call_result_94', xrange_call_result_94)
    # Testing if the for loop is going to be iterated (line 33)
    # Testing the type of a for loop iterable (line 33)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 33, 4), xrange_call_result_94)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 33, 4), xrange_call_result_94):
        # Getting the type of the for loop variable (line 33)
        for_loop_var_95 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 33, 4), xrange_call_result_94)
        # Assigning a type to the variable 'x' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'x', for_loop_var_95)
        # SSA begins for a for statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 34):
        
        # Assigning a BinOp to a Name (line 34):
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 34)
        x_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'x')
        
        # Obtaining the type of the subscript
        int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'int')
        # Getting the type of 'board' (line 34)
        board_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'board')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), board_98, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_100 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), getitem___99, int_97)
        
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), subscript_call_result_100, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), getitem___101, x_96)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 34)
        x_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), 'x')
        
        # Obtaining the type of the subscript
        int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 36), 'int')
        # Getting the type of 'board' (line 34)
        board_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'board')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 30), board_105, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 34, 30), getitem___106, int_104)
        
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 30), subscript_call_result_107, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 34, 30), getitem___108, x_103)
        
        # Applying the binary operator '+' (line 34)
        result_add_110 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 16), '+', subscript_call_result_102, subscript_call_result_109)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 34)
        x_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 53), 'x')
        
        # Obtaining the type of the subscript
        int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 50), 'int')
        # Getting the type of 'board' (line 34)
        board_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 44), 'board')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 44), board_113, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 34, 44), getitem___114, int_112)
        
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 44), subscript_call_result_115, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 34, 44), getitem___116, x_111)
        
        # Applying the binary operator '+' (line 34)
        result_add_118 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 42), '+', result_add_110, subscript_call_result_117)
        
        # Assigning a type to the variable 'score' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'score', result_add_118)
        
        
        # Call to xrange(...): (line 35)
        # Processing the call arguments (line 35)
        int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'int')
        # Getting the type of 'HEIGHT' (line 35)
        HEIGHT_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'HEIGHT', False)
        # Processing the call keyword arguments (line 35)
        kwargs_122 = {}
        # Getting the type of 'xrange' (line 35)
        xrange_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 35)
        xrange_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), xrange_119, *[int_120, HEIGHT_121], **kwargs_122)
        
        # Assigning a type to the variable 'xrange_call_result_123' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'xrange_call_result_123', xrange_call_result_123)
        # Testing if the for loop is going to be iterated (line 35)
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), xrange_call_result_123)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 35, 8), xrange_call_result_123):
            # Getting the type of the for loop variable (line 35)
            for_loop_var_124 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), xrange_call_result_123)
            # Assigning a type to the variable 'y' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'y', for_loop_var_124)
            # SSA begins for a for statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'score' (line 36)
            score_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'score')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 36)
            x_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'x')
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 36)
            y_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'y')
            # Getting the type of 'board' (line 36)
            board_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'board')
            # Obtaining the member '__getitem__' of a type (line 36)
            getitem___129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), board_128, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 36)
            subscript_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), getitem___129, y_127)
            
            # Obtaining the member '__getitem__' of a type (line 36)
            getitem___131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), subscript_call_result_130, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 36)
            subscript_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), getitem___131, x_126)
            
            # Applying the binary operator '+=' (line 36)
            result_iadd_133 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '+=', score_125, subscript_call_result_132)
            # Assigning a type to the variable 'score' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'score', result_iadd_133)
            
            
            # Getting the type of 'counters' (line 37)
            counters_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'counters')
            
            # Obtaining the type of the subscript
            # Getting the type of 'score' (line 37)
            score_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'score')
            int_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'int')
            # Applying the binary operator '+' (line 37)
            result_add_137 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 21), '+', score_135, int_136)
            
            # Getting the type of 'counters' (line 37)
            counters_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'counters')
            # Obtaining the member '__getitem__' of a type (line 37)
            getitem___139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), counters_138, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 37)
            subscript_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), getitem___139, result_add_137)
            
            int_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'int')
            # Applying the binary operator '+=' (line 37)
            result_iadd_142 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '+=', subscript_call_result_140, int_141)
            # Getting the type of 'counters' (line 37)
            counters_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'counters')
            # Getting the type of 'score' (line 37)
            score_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'score')
            int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'int')
            # Applying the binary operator '+' (line 37)
            result_add_146 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 21), '+', score_144, int_145)
            
            # Storing an element on a container (line 37)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 12), counters_143, (result_add_146, result_iadd_142))
            
            
            # Getting the type of 'score' (line 38)
            score_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'score')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 38)
            x_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'x')
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 38)
            y_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'y')
            int_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'int')
            # Applying the binary operator '-' (line 38)
            result_sub_151 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 27), '-', y_149, int_150)
            
            # Getting the type of 'board' (line 38)
            board_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'board')
            # Obtaining the member '__getitem__' of a type (line 38)
            getitem___153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), board_152, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 38)
            subscript_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), getitem___153, result_sub_151)
            
            # Obtaining the member '__getitem__' of a type (line 38)
            getitem___155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), subscript_call_result_154, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 38)
            subscript_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), getitem___155, x_148)
            
            # Applying the binary operator '-=' (line 38)
            result_isub_157 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 12), '-=', score_147, subscript_call_result_156)
            # Assigning a type to the variable 'score' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'score', result_isub_157)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to xrange(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'HEIGHT' (line 41)
    HEIGHT_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'HEIGHT', False)
    int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 29), 'int')
    # Applying the binary operator '-' (line 41)
    result_sub_161 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 20), '-', HEIGHT_159, int_160)
    
    # Processing the call keyword arguments (line 41)
    kwargs_162 = {}
    # Getting the type of 'xrange' (line 41)
    xrange_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 41)
    xrange_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 41, 13), xrange_158, *[result_sub_161], **kwargs_162)
    
    # Assigning a type to the variable 'xrange_call_result_163' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'xrange_call_result_163', xrange_call_result_163)
    # Testing if the for loop is going to be iterated (line 41)
    # Testing the type of a for loop iterable (line 41)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 4), xrange_call_result_163)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 41, 4), xrange_call_result_163):
        # Getting the type of the for loop variable (line 41)
        for_loop_var_164 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 4), xrange_call_result_163)
        # Assigning a type to the variable 'y' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'y', for_loop_var_164)
        # SSA begins for a for statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'WIDTH' (line 42)
        WIDTH_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'WIDTH', False)
        int_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'int')
        # Applying the binary operator '-' (line 42)
        result_sub_168 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 24), '-', WIDTH_166, int_167)
        
        # Processing the call keyword arguments (line 42)
        kwargs_169 = {}
        # Getting the type of 'xrange' (line 42)
        xrange_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 42)
        xrange_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 42, 17), xrange_165, *[result_sub_168], **kwargs_169)
        
        # Assigning a type to the variable 'xrange_call_result_170' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'xrange_call_result_170', xrange_call_result_170)
        # Testing if the for loop is going to be iterated (line 42)
        # Testing the type of a for loop iterable (line 42)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 8), xrange_call_result_170)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 42, 8), xrange_call_result_170):
            # Getting the type of the for loop variable (line 42)
            for_loop_var_171 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 8), xrange_call_result_170)
            # Assigning a type to the variable 'x' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'x', for_loop_var_171)
            # SSA begins for a for statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 43):
            
            # Assigning a Num to a Name (line 43):
            int_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'int')
            # Assigning a type to the variable 'score' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'score', int_172)
            
            
            # Call to xrange(...): (line 44)
            # Processing the call arguments (line 44)
            int_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'int')
            # Processing the call keyword arguments (line 44)
            kwargs_175 = {}
            # Getting the type of 'xrange' (line 44)
            xrange_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'xrange', False)
            # Calling xrange(args, kwargs) (line 44)
            xrange_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), xrange_173, *[int_174], **kwargs_175)
            
            # Assigning a type to the variable 'xrange_call_result_176' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'xrange_call_result_176', xrange_call_result_176)
            # Testing if the for loop is going to be iterated (line 44)
            # Testing the type of a for loop iterable (line 44)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 12), xrange_call_result_176)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 44, 12), xrange_call_result_176):
                # Getting the type of the for loop variable (line 44)
                for_loop_var_177 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 12), xrange_call_result_176)
                # Assigning a type to the variable 'idx' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'idx', for_loop_var_177)
                # SSA begins for a for statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 45):
                
                # Assigning a BinOp to a Name (line 45):
                # Getting the type of 'y' (line 45)
                y_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'y')
                # Getting the type of 'idx' (line 45)
                idx_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'idx')
                # Applying the binary operator '+' (line 45)
                result_add_180 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 21), '+', y_178, idx_179)
                
                # Assigning a type to the variable 'yy' (line 45)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'yy', result_add_180)
                
                # Assigning a BinOp to a Name (line 46):
                
                # Assigning a BinOp to a Name (line 46):
                # Getting the type of 'x' (line 46)
                x_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'x')
                # Getting the type of 'idx' (line 46)
                idx_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'idx')
                # Applying the binary operator '+' (line 46)
                result_add_183 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 21), '+', x_181, idx_182)
                
                # Assigning a type to the variable 'xx' (line 46)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'xx', result_add_183)
                
                # Getting the type of 'score' (line 47)
                score_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'score')
                
                # Obtaining the type of the subscript
                # Getting the type of 'xx' (line 47)
                xx_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 35), 'xx')
                
                # Obtaining the type of the subscript
                # Getting the type of 'yy' (line 47)
                yy_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'yy')
                # Getting the type of 'board' (line 47)
                board_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'board')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), board_187, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), getitem___188, yy_186)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), subscript_call_result_189, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_191 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), getitem___190, xx_185)
                
                # Applying the binary operator '+=' (line 47)
                result_iadd_192 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 16), '+=', score_184, subscript_call_result_191)
                # Assigning a type to the variable 'score' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'score', result_iadd_192)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'counters' (line 48)
            counters_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'counters')
            
            # Obtaining the type of the subscript
            # Getting the type of 'score' (line 48)
            score_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'score')
            int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'int')
            # Applying the binary operator '+' (line 48)
            result_add_196 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 21), '+', score_194, int_195)
            
            # Getting the type of 'counters' (line 48)
            counters_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'counters')
            # Obtaining the member '__getitem__' of a type (line 48)
            getitem___198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), counters_197, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 48)
            subscript_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), getitem___198, result_add_196)
            
            int_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 35), 'int')
            # Applying the binary operator '+=' (line 48)
            result_iadd_201 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 12), '+=', subscript_call_result_199, int_200)
            # Getting the type of 'counters' (line 48)
            counters_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'counters')
            # Getting the type of 'score' (line 48)
            score_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'score')
            int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'int')
            # Applying the binary operator '+' (line 48)
            result_add_205 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 21), '+', score_203, int_204)
            
            # Storing an element on a container (line 48)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 12), counters_202, (result_add_205, result_iadd_201))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to xrange(...): (line 51)
    # Processing the call arguments (line 51)
    int_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'int')
    # Getting the type of 'HEIGHT' (line 51)
    HEIGHT_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'HEIGHT', False)
    # Processing the call keyword arguments (line 51)
    kwargs_209 = {}
    # Getting the type of 'xrange' (line 51)
    xrange_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 51)
    xrange_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), xrange_206, *[int_207, HEIGHT_208], **kwargs_209)
    
    # Assigning a type to the variable 'xrange_call_result_210' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'xrange_call_result_210', xrange_call_result_210)
    # Testing if the for loop is going to be iterated (line 51)
    # Testing the type of a for loop iterable (line 51)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 4), xrange_call_result_210)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 51, 4), xrange_call_result_210):
        # Getting the type of the for loop variable (line 51)
        for_loop_var_211 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 4), xrange_call_result_210)
        # Assigning a type to the variable 'y' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'y', for_loop_var_211)
        # SSA begins for a for statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'WIDTH' (line 52)
        WIDTH_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'WIDTH', False)
        int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'int')
        # Applying the binary operator '-' (line 52)
        result_sub_215 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 24), '-', WIDTH_213, int_214)
        
        # Processing the call keyword arguments (line 52)
        kwargs_216 = {}
        # Getting the type of 'xrange' (line 52)
        xrange_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 52)
        xrange_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), xrange_212, *[result_sub_215], **kwargs_216)
        
        # Assigning a type to the variable 'xrange_call_result_217' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'xrange_call_result_217', xrange_call_result_217)
        # Testing if the for loop is going to be iterated (line 52)
        # Testing the type of a for loop iterable (line 52)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 8), xrange_call_result_217)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 52, 8), xrange_call_result_217):
            # Getting the type of the for loop variable (line 52)
            for_loop_var_218 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 8), xrange_call_result_217)
            # Assigning a type to the variable 'x' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'x', for_loop_var_218)
            # SSA begins for a for statement (line 52)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 53):
            
            # Assigning a Num to a Name (line 53):
            int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'int')
            # Assigning a type to the variable 'score' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'score', int_219)
            
            
            # Call to xrange(...): (line 54)
            # Processing the call arguments (line 54)
            int_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'int')
            # Processing the call keyword arguments (line 54)
            kwargs_222 = {}
            # Getting the type of 'xrange' (line 54)
            xrange_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'xrange', False)
            # Calling xrange(args, kwargs) (line 54)
            xrange_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 54, 23), xrange_220, *[int_221], **kwargs_222)
            
            # Assigning a type to the variable 'xrange_call_result_223' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'xrange_call_result_223', xrange_call_result_223)
            # Testing if the for loop is going to be iterated (line 54)
            # Testing the type of a for loop iterable (line 54)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 12), xrange_call_result_223)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 12), xrange_call_result_223):
                # Getting the type of the for loop variable (line 54)
                for_loop_var_224 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 12), xrange_call_result_223)
                # Assigning a type to the variable 'idx' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'idx', for_loop_var_224)
                # SSA begins for a for statement (line 54)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 55):
                
                # Assigning a BinOp to a Name (line 55):
                # Getting the type of 'y' (line 55)
                y_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'y')
                # Getting the type of 'idx' (line 55)
                idx_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'idx')
                # Applying the binary operator '-' (line 55)
                result_sub_227 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 21), '-', y_225, idx_226)
                
                # Assigning a type to the variable 'yy' (line 55)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'yy', result_sub_227)
                
                # Assigning a BinOp to a Name (line 56):
                
                # Assigning a BinOp to a Name (line 56):
                # Getting the type of 'x' (line 56)
                x_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'x')
                # Getting the type of 'idx' (line 56)
                idx_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'idx')
                # Applying the binary operator '+' (line 56)
                result_add_230 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 21), '+', x_228, idx_229)
                
                # Assigning a type to the variable 'xx' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'xx', result_add_230)
                
                # Getting the type of 'score' (line 57)
                score_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'score')
                
                # Obtaining the type of the subscript
                # Getting the type of 'xx' (line 57)
                xx_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 35), 'xx')
                
                # Obtaining the type of the subscript
                # Getting the type of 'yy' (line 57)
                yy_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'yy')
                # Getting the type of 'board' (line 57)
                board_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'board')
                # Obtaining the member '__getitem__' of a type (line 57)
                getitem___235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), board_234, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 57)
                subscript_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), getitem___235, yy_233)
                
                # Obtaining the member '__getitem__' of a type (line 57)
                getitem___237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), subscript_call_result_236, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 57)
                subscript_call_result_238 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), getitem___237, xx_232)
                
                # Applying the binary operator '+=' (line 57)
                result_iadd_239 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 16), '+=', score_231, subscript_call_result_238)
                # Assigning a type to the variable 'score' (line 57)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'score', result_iadd_239)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'counters' (line 58)
            counters_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'counters')
            
            # Obtaining the type of the subscript
            # Getting the type of 'score' (line 58)
            score_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'score')
            int_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'int')
            # Applying the binary operator '+' (line 58)
            result_add_243 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 21), '+', score_241, int_242)
            
            # Getting the type of 'counters' (line 58)
            counters_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'counters')
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), counters_244, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), getitem___245, result_add_243)
            
            int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 35), 'int')
            # Applying the binary operator '+=' (line 58)
            result_iadd_248 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 12), '+=', subscript_call_result_246, int_247)
            # Getting the type of 'counters' (line 58)
            counters_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'counters')
            # Getting the type of 'score' (line 58)
            score_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'score')
            int_251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'int')
            # Applying the binary operator '+' (line 58)
            result_add_252 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 21), '+', score_250, int_251)
            
            # Storing an element on a container (line 58)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 12), counters_249, (result_add_252, result_iadd_248))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Obtaining the type of the subscript
    int_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'int')
    # Getting the type of 'counters' (line 60)
    counters_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'counters')
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 7), counters_254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 60, 7), getitem___255, int_253)
    
    int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'int')
    # Applying the binary operator '!=' (line 60)
    result_ne_258 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 7), '!=', subscript_call_result_256, int_257)
    
    # Testing if the type of an if condition is none (line 60)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 4), result_ne_258):
        
        
        # Obtaining the type of the subscript
        int_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'int')
        # Getting the type of 'counters' (line 62)
        counters_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'counters')
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 9), counters_262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 62, 9), getitem___263, int_261)
        
        int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'int')
        # Applying the binary operator '!=' (line 62)
        result_ne_266 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 9), '!=', subscript_call_result_264, int_265)
        
        # Testing if the type of an if condition is none (line 62)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 9), result_ne_266):
            
            # Obtaining the type of the subscript
            int_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
            # Getting the type of 'counters' (line 65)
            counters_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), counters_270, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), getitem___271, int_269)
            
            int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
            
            # Obtaining the type of the subscript
            int_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 43), 'int')
            # Getting the type of 'counters' (line 65)
            counters_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 34), counters_275, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 65, 34), getitem___276, int_274)
            
            # Applying the binary operator '*' (line 65)
            result_mul_278 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '*', int_273, subscript_call_result_277)
            
            # Applying the binary operator '+' (line 65)
            result_add_279 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '+', subscript_call_result_272, result_mul_278)
            
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'int')
            
            # Obtaining the type of the subscript
            int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 61), 'int')
            # Getting the type of 'counters' (line 65)
            counters_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 52), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 52), counters_282, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 65, 52), getitem___283, int_281)
            
            # Applying the binary operator '*' (line 65)
            result_mul_285 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 48), '*', int_280, subscript_call_result_284)
            
            # Applying the binary operator '+' (line 65)
            result_add_286 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 46), '+', result_add_279, result_mul_285)
            
            int_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'int')
            
            # Obtaining the type of the subscript
            int_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'int')
            # Getting the type of 'counters' (line 66)
            counters_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), counters_289, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 66, 21), getitem___290, int_288)
            
            # Applying the binary operator '*' (line 66)
            result_mul_292 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 16), '*', int_287, subscript_call_result_291)
            
            # Applying the binary operator '+' (line 65)
            result_add_293 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 64), '+', result_add_286, result_mul_292)
            
            
            # Obtaining the type of the subscript
            int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'int')
            # Getting the type of 'counters' (line 66)
            counters_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 35), counters_295, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 66, 35), getitem___296, int_294)
            
            # Applying the binary operator '-' (line 66)
            result_sub_298 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 33), '-', result_add_293, subscript_call_result_297)
            
            int_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'int')
            
            # Obtaining the type of the subscript
            int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 62), 'int')
            # Getting the type of 'counters' (line 66)
            counters_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 53), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 53), counters_301, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 66, 53), getitem___302, int_300)
            
            # Applying the binary operator '*' (line 66)
            result_mul_304 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 49), '*', int_299, subscript_call_result_303)
            
            # Applying the binary operator '-' (line 66)
            result_sub_305 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 47), '-', result_sub_298, result_mul_304)
            
            int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'int')
            
            # Obtaining the type of the subscript
            int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
            # Getting the type of 'counters' (line 67)
            counters_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), counters_308, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), getitem___309, int_307)
            
            # Applying the binary operator '*' (line 67)
            result_mul_311 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 16), '*', int_306, subscript_call_result_310)
            
            # Applying the binary operator '-' (line 66)
            result_sub_312 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 65), '-', result_sub_305, result_mul_311)
            
            int_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'int')
            
            # Obtaining the type of the subscript
            int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'int')
            # Getting the type of 'counters' (line 67)
            counters_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 39), counters_315, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 67, 39), getitem___316, int_314)
            
            # Applying the binary operator '*' (line 67)
            result_mul_318 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 34), '*', int_313, subscript_call_result_317)
            
            # Applying the binary operator '-' (line 67)
            result_sub_319 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 32), '-', result_sub_312, result_mul_318)
            
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', result_sub_319)
        else:
            
            # Testing the type of an if condition (line 62)
            if_condition_267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 9), result_ne_266)
            # Assigning a type to the variable 'if_condition_267' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'if_condition_267', if_condition_267)
            # SSA begins for if statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'ORANGE_WINS' (line 63)
            ORANGE_WINS_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'ORANGE_WINS')
            # Assigning a type to the variable 'stypy_return_type' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', ORANGE_WINS_268)
            # SSA branch for the else part of an if statement (line 62)
            module_type_store.open_ssa_branch('else')
            
            # Obtaining the type of the subscript
            int_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
            # Getting the type of 'counters' (line 65)
            counters_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), counters_270, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), getitem___271, int_269)
            
            int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
            
            # Obtaining the type of the subscript
            int_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 43), 'int')
            # Getting the type of 'counters' (line 65)
            counters_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 34), counters_275, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 65, 34), getitem___276, int_274)
            
            # Applying the binary operator '*' (line 65)
            result_mul_278 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '*', int_273, subscript_call_result_277)
            
            # Applying the binary operator '+' (line 65)
            result_add_279 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '+', subscript_call_result_272, result_mul_278)
            
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'int')
            
            # Obtaining the type of the subscript
            int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 61), 'int')
            # Getting the type of 'counters' (line 65)
            counters_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 52), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 52), counters_282, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 65, 52), getitem___283, int_281)
            
            # Applying the binary operator '*' (line 65)
            result_mul_285 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 48), '*', int_280, subscript_call_result_284)
            
            # Applying the binary operator '+' (line 65)
            result_add_286 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 46), '+', result_add_279, result_mul_285)
            
            int_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'int')
            
            # Obtaining the type of the subscript
            int_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'int')
            # Getting the type of 'counters' (line 66)
            counters_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), counters_289, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 66, 21), getitem___290, int_288)
            
            # Applying the binary operator '*' (line 66)
            result_mul_292 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 16), '*', int_287, subscript_call_result_291)
            
            # Applying the binary operator '+' (line 65)
            result_add_293 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 64), '+', result_add_286, result_mul_292)
            
            
            # Obtaining the type of the subscript
            int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'int')
            # Getting the type of 'counters' (line 66)
            counters_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 35), counters_295, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 66, 35), getitem___296, int_294)
            
            # Applying the binary operator '-' (line 66)
            result_sub_298 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 33), '-', result_add_293, subscript_call_result_297)
            
            int_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'int')
            
            # Obtaining the type of the subscript
            int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 62), 'int')
            # Getting the type of 'counters' (line 66)
            counters_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 53), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 53), counters_301, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 66, 53), getitem___302, int_300)
            
            # Applying the binary operator '*' (line 66)
            result_mul_304 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 49), '*', int_299, subscript_call_result_303)
            
            # Applying the binary operator '-' (line 66)
            result_sub_305 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 47), '-', result_sub_298, result_mul_304)
            
            int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'int')
            
            # Obtaining the type of the subscript
            int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
            # Getting the type of 'counters' (line 67)
            counters_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), counters_308, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), getitem___309, int_307)
            
            # Applying the binary operator '*' (line 67)
            result_mul_311 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 16), '*', int_306, subscript_call_result_310)
            
            # Applying the binary operator '-' (line 66)
            result_sub_312 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 65), '-', result_sub_305, result_mul_311)
            
            int_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'int')
            
            # Obtaining the type of the subscript
            int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'int')
            # Getting the type of 'counters' (line 67)
            counters_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 39), counters_315, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 67, 39), getitem___316, int_314)
            
            # Applying the binary operator '*' (line 67)
            result_mul_318 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 34), '*', int_313, subscript_call_result_317)
            
            # Applying the binary operator '-' (line 67)
            result_sub_319 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 32), '-', result_sub_312, result_mul_318)
            
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', result_sub_319)
            # SSA join for if statement (line 62)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 60)
        if_condition_259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), result_ne_258)
        # Assigning a type to the variable 'if_condition_259' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_259', if_condition_259)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'YELLOW_WINS' (line 61)
        YELLOW_WINS_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'YELLOW_WINS')
        # Assigning a type to the variable 'stypy_return_type' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', YELLOW_WINS_260)
        # SSA branch for the else part of an if statement (line 60)
        module_type_store.open_ssa_branch('else')
        
        
        # Obtaining the type of the subscript
        int_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'int')
        # Getting the type of 'counters' (line 62)
        counters_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'counters')
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 9), counters_262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 62, 9), getitem___263, int_261)
        
        int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'int')
        # Applying the binary operator '!=' (line 62)
        result_ne_266 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 9), '!=', subscript_call_result_264, int_265)
        
        # Testing if the type of an if condition is none (line 62)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 9), result_ne_266):
            
            # Obtaining the type of the subscript
            int_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
            # Getting the type of 'counters' (line 65)
            counters_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), counters_270, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), getitem___271, int_269)
            
            int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
            
            # Obtaining the type of the subscript
            int_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 43), 'int')
            # Getting the type of 'counters' (line 65)
            counters_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 34), counters_275, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 65, 34), getitem___276, int_274)
            
            # Applying the binary operator '*' (line 65)
            result_mul_278 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '*', int_273, subscript_call_result_277)
            
            # Applying the binary operator '+' (line 65)
            result_add_279 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '+', subscript_call_result_272, result_mul_278)
            
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'int')
            
            # Obtaining the type of the subscript
            int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 61), 'int')
            # Getting the type of 'counters' (line 65)
            counters_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 52), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 52), counters_282, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 65, 52), getitem___283, int_281)
            
            # Applying the binary operator '*' (line 65)
            result_mul_285 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 48), '*', int_280, subscript_call_result_284)
            
            # Applying the binary operator '+' (line 65)
            result_add_286 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 46), '+', result_add_279, result_mul_285)
            
            int_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'int')
            
            # Obtaining the type of the subscript
            int_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'int')
            # Getting the type of 'counters' (line 66)
            counters_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), counters_289, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 66, 21), getitem___290, int_288)
            
            # Applying the binary operator '*' (line 66)
            result_mul_292 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 16), '*', int_287, subscript_call_result_291)
            
            # Applying the binary operator '+' (line 65)
            result_add_293 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 64), '+', result_add_286, result_mul_292)
            
            
            # Obtaining the type of the subscript
            int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'int')
            # Getting the type of 'counters' (line 66)
            counters_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 35), counters_295, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 66, 35), getitem___296, int_294)
            
            # Applying the binary operator '-' (line 66)
            result_sub_298 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 33), '-', result_add_293, subscript_call_result_297)
            
            int_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'int')
            
            # Obtaining the type of the subscript
            int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 62), 'int')
            # Getting the type of 'counters' (line 66)
            counters_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 53), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 53), counters_301, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 66, 53), getitem___302, int_300)
            
            # Applying the binary operator '*' (line 66)
            result_mul_304 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 49), '*', int_299, subscript_call_result_303)
            
            # Applying the binary operator '-' (line 66)
            result_sub_305 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 47), '-', result_sub_298, result_mul_304)
            
            int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'int')
            
            # Obtaining the type of the subscript
            int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
            # Getting the type of 'counters' (line 67)
            counters_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), counters_308, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), getitem___309, int_307)
            
            # Applying the binary operator '*' (line 67)
            result_mul_311 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 16), '*', int_306, subscript_call_result_310)
            
            # Applying the binary operator '-' (line 66)
            result_sub_312 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 65), '-', result_sub_305, result_mul_311)
            
            int_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'int')
            
            # Obtaining the type of the subscript
            int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'int')
            # Getting the type of 'counters' (line 67)
            counters_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 39), counters_315, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 67, 39), getitem___316, int_314)
            
            # Applying the binary operator '*' (line 67)
            result_mul_318 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 34), '*', int_313, subscript_call_result_317)
            
            # Applying the binary operator '-' (line 67)
            result_sub_319 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 32), '-', result_sub_312, result_mul_318)
            
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', result_sub_319)
        else:
            
            # Testing the type of an if condition (line 62)
            if_condition_267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 9), result_ne_266)
            # Assigning a type to the variable 'if_condition_267' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'if_condition_267', if_condition_267)
            # SSA begins for if statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'ORANGE_WINS' (line 63)
            ORANGE_WINS_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'ORANGE_WINS')
            # Assigning a type to the variable 'stypy_return_type' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', ORANGE_WINS_268)
            # SSA branch for the else part of an if statement (line 62)
            module_type_store.open_ssa_branch('else')
            
            # Obtaining the type of the subscript
            int_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
            # Getting the type of 'counters' (line 65)
            counters_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), counters_270, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), getitem___271, int_269)
            
            int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
            
            # Obtaining the type of the subscript
            int_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 43), 'int')
            # Getting the type of 'counters' (line 65)
            counters_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 34), counters_275, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 65, 34), getitem___276, int_274)
            
            # Applying the binary operator '*' (line 65)
            result_mul_278 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '*', int_273, subscript_call_result_277)
            
            # Applying the binary operator '+' (line 65)
            result_add_279 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '+', subscript_call_result_272, result_mul_278)
            
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'int')
            
            # Obtaining the type of the subscript
            int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 61), 'int')
            # Getting the type of 'counters' (line 65)
            counters_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 52), 'counters')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 52), counters_282, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 65, 52), getitem___283, int_281)
            
            # Applying the binary operator '*' (line 65)
            result_mul_285 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 48), '*', int_280, subscript_call_result_284)
            
            # Applying the binary operator '+' (line 65)
            result_add_286 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 46), '+', result_add_279, result_mul_285)
            
            int_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'int')
            
            # Obtaining the type of the subscript
            int_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'int')
            # Getting the type of 'counters' (line 66)
            counters_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), counters_289, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 66, 21), getitem___290, int_288)
            
            # Applying the binary operator '*' (line 66)
            result_mul_292 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 16), '*', int_287, subscript_call_result_291)
            
            # Applying the binary operator '+' (line 65)
            result_add_293 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 64), '+', result_add_286, result_mul_292)
            
            
            # Obtaining the type of the subscript
            int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'int')
            # Getting the type of 'counters' (line 66)
            counters_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 35), counters_295, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 66, 35), getitem___296, int_294)
            
            # Applying the binary operator '-' (line 66)
            result_sub_298 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 33), '-', result_add_293, subscript_call_result_297)
            
            int_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'int')
            
            # Obtaining the type of the subscript
            int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 62), 'int')
            # Getting the type of 'counters' (line 66)
            counters_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 53), 'counters')
            # Obtaining the member '__getitem__' of a type (line 66)
            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 53), counters_301, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 66)
            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 66, 53), getitem___302, int_300)
            
            # Applying the binary operator '*' (line 66)
            result_mul_304 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 49), '*', int_299, subscript_call_result_303)
            
            # Applying the binary operator '-' (line 66)
            result_sub_305 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 47), '-', result_sub_298, result_mul_304)
            
            int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'int')
            
            # Obtaining the type of the subscript
            int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
            # Getting the type of 'counters' (line 67)
            counters_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), counters_308, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), getitem___309, int_307)
            
            # Applying the binary operator '*' (line 67)
            result_mul_311 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 16), '*', int_306, subscript_call_result_310)
            
            # Applying the binary operator '-' (line 66)
            result_sub_312 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 65), '-', result_sub_305, result_mul_311)
            
            int_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'int')
            
            # Obtaining the type of the subscript
            int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'int')
            # Getting the type of 'counters' (line 67)
            counters_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'counters')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 39), counters_315, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 67, 39), getitem___316, int_314)
            
            # Applying the binary operator '*' (line 67)
            result_mul_318 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 34), '*', int_313, subscript_call_result_317)
            
            # Applying the binary operator '-' (line 67)
            result_sub_319 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 32), '-', result_sub_312, result_mul_318)
            
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', result_sub_319)
            # SSA join for if statement (line 62)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'score_board(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'score_board' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_320)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'score_board'
    return stypy_return_type_320

# Assigning a type to the variable 'score_board' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'score_board', score_board)

@norecursion
def drop_disk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'drop_disk'
    module_type_store = module_type_store.open_function_context('drop_disk', 70, 0, False)
    
    # Passed parameters checking function
    drop_disk.stypy_localization = localization
    drop_disk.stypy_type_of_self = None
    drop_disk.stypy_type_store = module_type_store
    drop_disk.stypy_function_name = 'drop_disk'
    drop_disk.stypy_param_names_list = ['board', 'column', 'color']
    drop_disk.stypy_varargs_param_name = None
    drop_disk.stypy_kwargs_param_name = None
    drop_disk.stypy_call_defaults = defaults
    drop_disk.stypy_call_varargs = varargs
    drop_disk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'drop_disk', ['board', 'column', 'color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'drop_disk', localization, ['board', 'column', 'color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'drop_disk(...)' code ##################

    
    
    # Call to xrange(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'HEIGHT' (line 71)
    HEIGHT_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'HEIGHT', False)
    int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
    # Applying the binary operator '-' (line 71)
    result_sub_324 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 20), '-', HEIGHT_322, int_323)
    
    int_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 32), 'int')
    int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 36), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_327 = {}
    # Getting the type of 'xrange' (line 71)
    xrange_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 71)
    xrange_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 71, 13), xrange_321, *[result_sub_324, int_325, int_326], **kwargs_327)
    
    # Assigning a type to the variable 'xrange_call_result_328' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'xrange_call_result_328', xrange_call_result_328)
    # Testing if the for loop is going to be iterated (line 71)
    # Testing the type of a for loop iterable (line 71)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 4), xrange_call_result_328)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 71, 4), xrange_call_result_328):
        # Getting the type of the for loop variable (line 71)
        for_loop_var_329 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 4), xrange_call_result_328)
        # Assigning a type to the variable 'y' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'y', for_loop_var_329)
        # SSA begins for a for statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'column' (line 72)
        column_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'column')
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 72)
        y_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'y')
        # Getting the type of 'board' (line 72)
        board_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'board')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), board_332, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 72, 11), getitem___333, y_331)
        
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), subscript_call_result_334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 72, 11), getitem___335, column_330)
        
        # Getting the type of 'Cell' (line 72)
        Cell_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'Cell')
        # Obtaining the member 'Barren' of a type (line 72)
        Barren_338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 31), Cell_337, 'Barren')
        # Applying the binary operator '==' (line 72)
        result_eq_339 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), '==', subscript_call_result_336, Barren_338)
        
        # Testing if the type of an if condition is none (line 72)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 8), result_eq_339):
            pass
        else:
            
            # Testing the type of an if condition (line 72)
            if_condition_340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 8), result_eq_339)
            # Assigning a type to the variable 'if_condition_340' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'if_condition_340', if_condition_340)
            # SSA begins for if statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Subscript (line 73):
            
            # Assigning a Name to a Subscript (line 73):
            # Getting the type of 'color' (line 73)
            color_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'color')
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 73)
            y_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'y')
            # Getting the type of 'board' (line 73)
            board_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'board')
            # Obtaining the member '__getitem__' of a type (line 73)
            getitem___344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), board_343, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 73)
            subscript_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), getitem___344, y_342)
            
            # Getting the type of 'column' (line 73)
            column_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'column')
            # Storing an element on a container (line 73)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), subscript_call_result_345, (column_346, color_341))
            # Getting the type of 'y' (line 74)
            y_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'y')
            # Assigning a type to the variable 'stypy_return_type' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', y_347)
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    int_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', int_348)
    
    # ################# End of 'drop_disk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'drop_disk' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_349)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'drop_disk'
    return stypy_return_type_349

# Assigning a type to the variable 'drop_disk' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'drop_disk', drop_disk)

@norecursion
def load_board(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'load_board'
    module_type_store = module_type_store.open_function_context('load_board', 78, 0, False)
    
    # Passed parameters checking function
    load_board.stypy_localization = localization
    load_board.stypy_type_of_self = None
    load_board.stypy_type_store = module_type_store
    load_board.stypy_function_name = 'load_board'
    load_board.stypy_param_names_list = ['args']
    load_board.stypy_varargs_param_name = None
    load_board.stypy_kwargs_param_name = None
    load_board.stypy_call_defaults = defaults
    load_board.stypy_call_varargs = varargs
    load_board.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'load_board', ['args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'load_board', localization, ['args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'load_board(...)' code ##################

    # Marking variables as global (line 79)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 79, 4), 'g_debug')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 79, 4), 'g_max_depth')
    
    # Assigning a ListComp to a Name (line 80):
    
    # Assigning a ListComp to a Name (line 80):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'HEIGHT' (line 80)
    HEIGHT_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 55), 'HEIGHT', False)
    # Processing the call keyword arguments (line 80)
    kwargs_357 = {}
    # Getting the type of 'xrange' (line 80)
    xrange_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 48), 'xrange', False)
    # Calling xrange(args, kwargs) (line 80)
    xrange_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 80, 48), xrange_355, *[HEIGHT_356], **kwargs_357)
    
    comprehension_359 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), xrange_call_result_358)
    # Assigning a type to the variable '_' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), '_', comprehension_359)
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    # Getting the type of 'Cell' (line 80)
    Cell_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'Cell')
    # Obtaining the member 'Barren' of a type (line 80)
    Barren_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), Cell_351, 'Barren')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), list_350, Barren_352)
    
    # Getting the type of 'WIDTH' (line 80)
    WIDTH_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'WIDTH')
    # Applying the binary operator '*' (line 80)
    result_mul_354 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 17), '*', list_350, WIDTH_353)
    
    list_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), list_360, result_mul_354)
    # Assigning a type to the variable 'new_board' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'new_board', list_360)
    
    
    # Call to enumerate(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Obtaining the type of the subscript
    int_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 33), 'int')
    slice_363 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 28), int_362, None, None)
    # Getting the type of 'args' (line 82)
    args_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 28), args_364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_366 = invoke(stypy.reporting.localization.Localization(__file__, 82, 28), getitem___365, slice_363)
    
    # Processing the call keyword arguments (line 82)
    kwargs_367 = {}
    # Getting the type of 'enumerate' (line 82)
    enumerate_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 82)
    enumerate_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), enumerate_361, *[subscript_call_result_366], **kwargs_367)
    
    # Assigning a type to the variable 'enumerate_call_result_368' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'enumerate_call_result_368', enumerate_call_result_368)
    # Testing if the for loop is going to be iterated (line 82)
    # Testing the type of a for loop iterable (line 82)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 4), enumerate_call_result_368)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 82, 4), enumerate_call_result_368):
        # Getting the type of the for loop variable (line 82)
        for_loop_var_369 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 4), enumerate_call_result_368)
        # Assigning a type to the variable 'i' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 4), for_loop_var_369, 2, 0))
        # Assigning a type to the variable 'arg' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 4), for_loop_var_369, 2, 1))
        # SSA begins for a for statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'int')
        # Getting the type of 'arg' (line 83)
        arg_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'arg')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), arg_371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), getitem___372, int_370)
        
        str_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'str', 'o')
        # Applying the binary operator '==' (line 83)
        result_eq_375 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '==', subscript_call_result_373, str_374)
        
        
        
        # Obtaining the type of the subscript
        int_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 32), 'int')
        # Getting the type of 'arg' (line 83)
        arg_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'arg')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 28), arg_377, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_379 = invoke(stypy.reporting.localization.Localization(__file__, 83, 28), getitem___378, int_376)
        
        str_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 38), 'str', 'y')
        # Applying the binary operator '==' (line 83)
        result_eq_381 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 28), '==', subscript_call_result_379, str_380)
        
        # Applying the binary operator 'or' (line 83)
        result_or_keyword_382 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), 'or', result_eq_375, result_eq_381)
        
        # Testing if the type of an if condition is none (line 83)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 8), result_or_keyword_382):
            
            # Getting the type of 'arg' (line 86)
            arg_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'arg')
            str_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'str', '-debug')
            # Applying the binary operator '==' (line 86)
            result_eq_424 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 13), '==', arg_422, str_423)
            
            # Testing if the type of an if condition is none (line 86)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 13), result_eq_424):
                
                # Evaluating a boolean operation
                
                # Getting the type of 'arg' (line 88)
                arg_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'arg')
                str_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', '-level')
                # Applying the binary operator '==' (line 88)
                result_eq_429 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '==', arg_427, str_428)
                
                
                # Getting the type of 'i' (line 88)
                i_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'i')
                
                # Call to len(...): (line 88)
                # Processing the call arguments (line 88)
                # Getting the type of 'args' (line 88)
                args_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'args', False)
                # Processing the call keyword arguments (line 88)
                kwargs_433 = {}
                # Getting the type of 'len' (line 88)
                len_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'len', False)
                # Calling len(args, kwargs) (line 88)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 88, 38), len_431, *[args_432], **kwargs_433)
                
                int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'int')
                # Applying the binary operator '-' (line 88)
                result_sub_436 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 38), '-', len_call_result_434, int_435)
                
                # Applying the binary operator '<' (line 88)
                result_lt_437 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 33), '<', i_430, result_sub_436)
                
                # Applying the binary operator 'and' (line 88)
                result_and_keyword_438 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), 'and', result_eq_429, result_lt_437)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438)
                    # Assigning a type to the variable 'if_condition_439' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'if_condition_439', if_condition_439)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to int(...): (line 89)
                    # Processing the call arguments (line 89)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 89)
                    i_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'i', False)
                    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
                    # Applying the binary operator '+' (line 89)
                    result_add_443 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 35), '+', i_441, int_442)
                    
                    # Getting the type of 'args' (line 89)
                    args_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'args', False)
                    # Obtaining the member '__getitem__' of a type (line 89)
                    getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), args_444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                    subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 89, 30), getitem___445, result_add_443)
                    
                    # Processing the call keyword arguments (line 89)
                    kwargs_447 = {}
                    # Getting the type of 'int' (line 89)
                    int_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'int', False)
                    # Calling int(args, kwargs) (line 89)
                    int_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), int_440, *[subscript_call_result_446], **kwargs_447)
                    
                    # Assigning a type to the variable 'g_max_depth' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'g_max_depth', int_call_result_448)
                    # SSA join for if statement (line 88)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 86)
                if_condition_425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 13), result_eq_424)
                # Assigning a type to the variable 'if_condition_425' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'if_condition_425', if_condition_425)
                # SSA begins for if statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 87):
                
                # Assigning a Name to a Name (line 87):
                # Getting the type of 'True' (line 87)
                True_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'True')
                # Assigning a type to the variable 'g_debug' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'g_debug', True_426)
                # SSA branch for the else part of an if statement (line 86)
                module_type_store.open_ssa_branch('else')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'arg' (line 88)
                arg_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'arg')
                str_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', '-level')
                # Applying the binary operator '==' (line 88)
                result_eq_429 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '==', arg_427, str_428)
                
                
                # Getting the type of 'i' (line 88)
                i_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'i')
                
                # Call to len(...): (line 88)
                # Processing the call arguments (line 88)
                # Getting the type of 'args' (line 88)
                args_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'args', False)
                # Processing the call keyword arguments (line 88)
                kwargs_433 = {}
                # Getting the type of 'len' (line 88)
                len_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'len', False)
                # Calling len(args, kwargs) (line 88)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 88, 38), len_431, *[args_432], **kwargs_433)
                
                int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'int')
                # Applying the binary operator '-' (line 88)
                result_sub_436 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 38), '-', len_call_result_434, int_435)
                
                # Applying the binary operator '<' (line 88)
                result_lt_437 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 33), '<', i_430, result_sub_436)
                
                # Applying the binary operator 'and' (line 88)
                result_and_keyword_438 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), 'and', result_eq_429, result_lt_437)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438)
                    # Assigning a type to the variable 'if_condition_439' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'if_condition_439', if_condition_439)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to int(...): (line 89)
                    # Processing the call arguments (line 89)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 89)
                    i_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'i', False)
                    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
                    # Applying the binary operator '+' (line 89)
                    result_add_443 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 35), '+', i_441, int_442)
                    
                    # Getting the type of 'args' (line 89)
                    args_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'args', False)
                    # Obtaining the member '__getitem__' of a type (line 89)
                    getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), args_444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                    subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 89, 30), getitem___445, result_add_443)
                    
                    # Processing the call keyword arguments (line 89)
                    kwargs_447 = {}
                    # Getting the type of 'int' (line 89)
                    int_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'int', False)
                    # Calling int(args, kwargs) (line 89)
                    int_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), int_440, *[subscript_call_result_446], **kwargs_447)
                    
                    # Assigning a type to the variable 'g_max_depth' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'g_max_depth', int_call_result_448)
                    # SSA join for if statement (line 88)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 83)
            if_condition_383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_or_keyword_382)
            # Assigning a type to the variable 'if_condition_383' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_383', if_condition_383)
            # SSA begins for if statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a IfExp to a Subscript (line 84):
            
            # Assigning a IfExp to a Subscript (line 84):
            
            
            
            # Obtaining the type of the subscript
            int_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'int')
            # Getting the type of 'arg' (line 85)
            arg_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'arg')
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 31), arg_385, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_387 = invoke(stypy.reporting.localization.Localization(__file__, 85, 31), getitem___386, int_384)
            
            str_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 41), 'str', 'o')
            # Applying the binary operator '==' (line 85)
            result_eq_389 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 31), '==', subscript_call_result_387, str_388)
            
            # Testing the type of an if expression (line 85)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 16), result_eq_389)
            # SSA begins for if expression (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'Cell' (line 85)
            Cell_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'Cell')
            # Obtaining the member 'Orange' of a type (line 85)
            Orange_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), Cell_390, 'Orange')
            # SSA branch for the else part of an if expression (line 85)
            module_type_store.open_ssa_branch('if expression else')
            # Getting the type of 'Cell' (line 85)
            Cell_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 50), 'Cell')
            # Obtaining the member 'Yellow' of a type (line 85)
            Yellow_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 50), Cell_392, 'Yellow')
            # SSA join for if expression (line 85)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_394 = union_type.UnionType.add(Orange_391, Yellow_393)
            
            
            # Obtaining the type of the subscript
            
            # Call to ord(...): (line 84)
            # Processing the call arguments (line 84)
            
            # Obtaining the type of the subscript
            int_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
            # Getting the type of 'arg' (line 84)
            arg_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'arg', False)
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 26), arg_397, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 84, 26), getitem___398, int_396)
            
            # Processing the call keyword arguments (line 84)
            kwargs_400 = {}
            # Getting the type of 'ord' (line 84)
            ord_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'ord', False)
            # Calling ord(args, kwargs) (line 84)
            ord_call_result_401 = invoke(stypy.reporting.localization.Localization(__file__, 84, 22), ord_395, *[subscript_call_result_399], **kwargs_400)
            
            
            # Call to ord(...): (line 84)
            # Processing the call arguments (line 84)
            str_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 40), 'str', '0')
            # Processing the call keyword arguments (line 84)
            kwargs_404 = {}
            # Getting the type of 'ord' (line 84)
            ord_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'ord', False)
            # Calling ord(args, kwargs) (line 84)
            ord_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 84, 36), ord_402, *[str_403], **kwargs_404)
            
            # Applying the binary operator '-' (line 84)
            result_sub_406 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 22), '-', ord_call_result_401, ord_call_result_405)
            
            # Getting the type of 'new_board' (line 84)
            new_board_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'new_board')
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), new_board_407, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), getitem___408, result_sub_406)
            
            
            # Call to ord(...): (line 84)
            # Processing the call arguments (line 84)
            
            # Obtaining the type of the subscript
            int_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 54), 'int')
            # Getting the type of 'arg' (line 84)
            arg_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 50), 'arg', False)
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 50), arg_412, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 84, 50), getitem___413, int_411)
            
            # Processing the call keyword arguments (line 84)
            kwargs_415 = {}
            # Getting the type of 'ord' (line 84)
            ord_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 46), 'ord', False)
            # Calling ord(args, kwargs) (line 84)
            ord_call_result_416 = invoke(stypy.reporting.localization.Localization(__file__, 84, 46), ord_410, *[subscript_call_result_414], **kwargs_415)
            
            
            # Call to ord(...): (line 84)
            # Processing the call arguments (line 84)
            str_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 64), 'str', '0')
            # Processing the call keyword arguments (line 84)
            kwargs_419 = {}
            # Getting the type of 'ord' (line 84)
            ord_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 60), 'ord', False)
            # Calling ord(args, kwargs) (line 84)
            ord_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 84, 60), ord_417, *[str_418], **kwargs_419)
            
            # Applying the binary operator '-' (line 84)
            result_sub_421 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 46), '-', ord_call_result_416, ord_call_result_420)
            
            # Storing an element on a container (line 84)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 12), subscript_call_result_409, (result_sub_421, if_exp_394))
            # SSA branch for the else part of an if statement (line 83)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'arg' (line 86)
            arg_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'arg')
            str_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'str', '-debug')
            # Applying the binary operator '==' (line 86)
            result_eq_424 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 13), '==', arg_422, str_423)
            
            # Testing if the type of an if condition is none (line 86)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 13), result_eq_424):
                
                # Evaluating a boolean operation
                
                # Getting the type of 'arg' (line 88)
                arg_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'arg')
                str_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', '-level')
                # Applying the binary operator '==' (line 88)
                result_eq_429 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '==', arg_427, str_428)
                
                
                # Getting the type of 'i' (line 88)
                i_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'i')
                
                # Call to len(...): (line 88)
                # Processing the call arguments (line 88)
                # Getting the type of 'args' (line 88)
                args_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'args', False)
                # Processing the call keyword arguments (line 88)
                kwargs_433 = {}
                # Getting the type of 'len' (line 88)
                len_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'len', False)
                # Calling len(args, kwargs) (line 88)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 88, 38), len_431, *[args_432], **kwargs_433)
                
                int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'int')
                # Applying the binary operator '-' (line 88)
                result_sub_436 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 38), '-', len_call_result_434, int_435)
                
                # Applying the binary operator '<' (line 88)
                result_lt_437 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 33), '<', i_430, result_sub_436)
                
                # Applying the binary operator 'and' (line 88)
                result_and_keyword_438 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), 'and', result_eq_429, result_lt_437)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438)
                    # Assigning a type to the variable 'if_condition_439' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'if_condition_439', if_condition_439)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to int(...): (line 89)
                    # Processing the call arguments (line 89)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 89)
                    i_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'i', False)
                    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
                    # Applying the binary operator '+' (line 89)
                    result_add_443 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 35), '+', i_441, int_442)
                    
                    # Getting the type of 'args' (line 89)
                    args_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'args', False)
                    # Obtaining the member '__getitem__' of a type (line 89)
                    getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), args_444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                    subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 89, 30), getitem___445, result_add_443)
                    
                    # Processing the call keyword arguments (line 89)
                    kwargs_447 = {}
                    # Getting the type of 'int' (line 89)
                    int_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'int', False)
                    # Calling int(args, kwargs) (line 89)
                    int_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), int_440, *[subscript_call_result_446], **kwargs_447)
                    
                    # Assigning a type to the variable 'g_max_depth' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'g_max_depth', int_call_result_448)
                    # SSA join for if statement (line 88)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 86)
                if_condition_425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 13), result_eq_424)
                # Assigning a type to the variable 'if_condition_425' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'if_condition_425', if_condition_425)
                # SSA begins for if statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 87):
                
                # Assigning a Name to a Name (line 87):
                # Getting the type of 'True' (line 87)
                True_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'True')
                # Assigning a type to the variable 'g_debug' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'g_debug', True_426)
                # SSA branch for the else part of an if statement (line 86)
                module_type_store.open_ssa_branch('else')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'arg' (line 88)
                arg_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'arg')
                str_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', '-level')
                # Applying the binary operator '==' (line 88)
                result_eq_429 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '==', arg_427, str_428)
                
                
                # Getting the type of 'i' (line 88)
                i_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'i')
                
                # Call to len(...): (line 88)
                # Processing the call arguments (line 88)
                # Getting the type of 'args' (line 88)
                args_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'args', False)
                # Processing the call keyword arguments (line 88)
                kwargs_433 = {}
                # Getting the type of 'len' (line 88)
                len_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'len', False)
                # Calling len(args, kwargs) (line 88)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 88, 38), len_431, *[args_432], **kwargs_433)
                
                int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'int')
                # Applying the binary operator '-' (line 88)
                result_sub_436 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 38), '-', len_call_result_434, int_435)
                
                # Applying the binary operator '<' (line 88)
                result_lt_437 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 33), '<', i_430, result_sub_436)
                
                # Applying the binary operator 'and' (line 88)
                result_and_keyword_438 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), 'and', result_eq_429, result_lt_437)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 13), result_and_keyword_438)
                    # Assigning a type to the variable 'if_condition_439' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'if_condition_439', if_condition_439)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to int(...): (line 89)
                    # Processing the call arguments (line 89)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 89)
                    i_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'i', False)
                    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
                    # Applying the binary operator '+' (line 89)
                    result_add_443 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 35), '+', i_441, int_442)
                    
                    # Getting the type of 'args' (line 89)
                    args_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'args', False)
                    # Obtaining the member '__getitem__' of a type (line 89)
                    getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), args_444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                    subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 89, 30), getitem___445, result_add_443)
                    
                    # Processing the call keyword arguments (line 89)
                    kwargs_447 = {}
                    # Getting the type of 'int' (line 89)
                    int_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'int', False)
                    # Calling int(args, kwargs) (line 89)
                    int_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), int_440, *[subscript_call_result_446], **kwargs_447)
                    
                    # Assigning a type to the variable 'g_max_depth' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'g_max_depth', int_call_result_448)
                    # SSA join for if statement (line 88)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'new_board' (line 91)
    new_board_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'new_board')
    # Assigning a type to the variable 'stypy_return_type' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type', new_board_449)
    
    # ################# End of 'load_board(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_board' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_board'
    return stypy_return_type_450

# Assigning a type to the variable 'load_board' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'load_board', load_board)

@norecursion
def ab_minimax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ab_minimax'
    module_type_store = module_type_store.open_function_context('ab_minimax', 94, 0, False)
    
    # Passed parameters checking function
    ab_minimax.stypy_localization = localization
    ab_minimax.stypy_type_of_self = None
    ab_minimax.stypy_type_store = module_type_store
    ab_minimax.stypy_function_name = 'ab_minimax'
    ab_minimax.stypy_param_names_list = ['maximize_or_minimize', 'color', 'depth', 'board']
    ab_minimax.stypy_varargs_param_name = None
    ab_minimax.stypy_kwargs_param_name = None
    ab_minimax.stypy_call_defaults = defaults
    ab_minimax.stypy_call_varargs = varargs
    ab_minimax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ab_minimax', ['maximize_or_minimize', 'color', 'depth', 'board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ab_minimax', localization, ['maximize_or_minimize', 'color', 'depth', 'board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ab_minimax(...)' code ##################

    # Marking variables as global (line 95)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 95, 4), 'g_max_depth')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 95, 4), 'g_debug')
    
    # Getting the type of 'depth' (line 96)
    depth_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'depth')
    int_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 16), 'int')
    # Applying the binary operator '==' (line 96)
    result_eq_453 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '==', depth_451, int_452)
    
    # Testing if the type of an if condition is none (line 96)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 4), result_eq_453):
        
        # Assigning a IfExp to a Name (line 99):
        
        # Assigning a IfExp to a Name (line 99):
        
        # Getting the type of 'maximize_or_minimize' (line 99)
        maximize_or_minimize_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'maximize_or_minimize')
        # Testing the type of an if expression (line 99)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 21), maximize_or_minimize_461)
        # SSA begins for if expression (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'int')
        # SSA branch for the else part of an if expression (line 99)
        module_type_store.open_ssa_branch('if expression else')
        int_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 60), 'int')
        # SSA join for if expression (line 99)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_464 = union_type.UnionType.add(int_462, int_463)
        
        # Assigning a type to the variable 'best_score' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'best_score', if_exp_464)
        
        # Assigning a Num to a Name (line 100):
        
        # Assigning a Num to a Name (line 100):
        int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'int')
        # Assigning a type to the variable 'bestMove' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'bestMove', int_465)
        
        
        # Call to xrange(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'WIDTH' (line 101)
        WIDTH_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'WIDTH', False)
        # Processing the call keyword arguments (line 101)
        kwargs_468 = {}
        # Getting the type of 'xrange' (line 101)
        xrange_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'xrange', False)
        # Calling xrange(args, kwargs) (line 101)
        xrange_call_result_469 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), xrange_466, *[WIDTH_467], **kwargs_468)
        
        # Assigning a type to the variable 'xrange_call_result_469' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'xrange_call_result_469', xrange_call_result_469)
        # Testing if the for loop is going to be iterated (line 101)
        # Testing the type of a for loop iterable (line 101)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 8), xrange_call_result_469)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 101, 8), xrange_call_result_469):
            # Getting the type of the for loop variable (line 101)
            for_loop_var_470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 8), xrange_call_result_469)
            # Assigning a type to the variable 'column' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'column', for_loop_var_470)
            # SSA begins for a for statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'column' (line 102)
            column_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'column')
            
            # Obtaining the type of the subscript
            int_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'int')
            # Getting the type of 'board' (line 102)
            board_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'board')
            # Obtaining the member '__getitem__' of a type (line 102)
            getitem___474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), board_473, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 102)
            subscript_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), getitem___474, int_472)
            
            # Obtaining the member '__getitem__' of a type (line 102)
            getitem___476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), subscript_call_result_475, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 102)
            subscript_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), getitem___476, column_471)
            
            # Getting the type of 'Cell' (line 102)
            Cell_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'Cell')
            # Obtaining the member 'Barren' of a type (line 102)
            Barren_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 35), Cell_478, 'Barren')
            # Applying the binary operator '!=' (line 102)
            result_ne_480 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 15), '!=', subscript_call_result_477, Barren_479)
            
            # Testing if the type of an if condition is none (line 102)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 12), result_ne_480):
                pass
            else:
                
                # Testing the type of an if condition (line 102)
                if_condition_481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 12), result_ne_480)
                # Assigning a type to the variable 'if_condition_481' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'if_condition_481', if_condition_481)
                # SSA begins for if statement (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 102)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 104):
            
            # Assigning a Call to a Name (line 104):
            
            # Call to drop_disk(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'board' (line 104)
            board_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'board', False)
            # Getting the type of 'column' (line 104)
            column_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 41), 'column', False)
            # Getting the type of 'color' (line 104)
            color_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 49), 'color', False)
            # Processing the call keyword arguments (line 104)
            kwargs_486 = {}
            # Getting the type of 'drop_disk' (line 104)
            drop_disk_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'drop_disk', False)
            # Calling drop_disk(args, kwargs) (line 104)
            drop_disk_call_result_487 = invoke(stypy.reporting.localization.Localization(__file__, 104, 24), drop_disk_482, *[board_483, column_484, color_485], **kwargs_486)
            
            # Assigning a type to the variable 'rowFilled' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'rowFilled', drop_disk_call_result_487)
            
            # Getting the type of 'rowFilled' (line 105)
            rowFilled_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'rowFilled')
            int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'int')
            # Applying the binary operator '==' (line 105)
            result_eq_490 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), '==', rowFilled_488, int_489)
            
            # Testing if the type of an if condition is none (line 105)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 105, 12), result_eq_490):
                pass
            else:
                
                # Testing the type of an if condition (line 105)
                if_condition_491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 12), result_eq_490)
                # Assigning a type to the variable 'if_condition_491' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'if_condition_491', if_condition_491)
                # SSA begins for if statement (line 105)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 105)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 107):
            
            # Assigning a Call to a Name (line 107):
            
            # Call to score_board(...): (line 107)
            # Processing the call arguments (line 107)
            # Getting the type of 'board' (line 107)
            board_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'board', False)
            # Processing the call keyword arguments (line 107)
            kwargs_494 = {}
            # Getting the type of 'score_board' (line 107)
            score_board_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'score_board', False)
            # Calling score_board(args, kwargs) (line 107)
            score_board_call_result_495 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), score_board_492, *[board_493], **kwargs_494)
            
            # Assigning a type to the variable 's' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 's', score_board_call_result_495)
            
            # Getting the type of 's' (line 108)
            s_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 's')
            
            # Getting the type of 'maximize_or_minimize' (line 108)
            maximize_or_minimize_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 36), 'maximize_or_minimize')
            # Testing the type of an if expression (line 108)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 21), maximize_or_minimize_497)
            # SSA begins for if expression (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'ORANGE_WINS' (line 108)
            ORANGE_WINS_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'ORANGE_WINS')
            # SSA branch for the else part of an if expression (line 108)
            module_type_store.open_ssa_branch('if expression else')
            # Getting the type of 'YELLOW_WINS' (line 108)
            YELLOW_WINS_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 62), 'YELLOW_WINS')
            # SSA join for if expression (line 108)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_500 = union_type.UnionType.add(ORANGE_WINS_498, YELLOW_WINS_499)
            
            # Applying the binary operator '==' (line 108)
            result_eq_501 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 15), '==', s_496, if_exp_500)
            
            # Testing if the type of an if condition is none (line 108)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 12), result_eq_501):
                pass
            else:
                
                # Testing the type of an if condition (line 108)
                if_condition_502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), result_eq_501)
                # Assigning a type to the variable 'if_condition_502' (line 108)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_502', if_condition_502)
                # SSA begins for if statement (line 108)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 109):
                
                # Assigning a Name to a Name (line 109):
                # Getting the type of 'column' (line 109)
                column_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'column')
                # Assigning a type to the variable 'bestMove' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'bestMove', column_503)
                
                # Assigning a Name to a Name (line 110):
                
                # Assigning a Name to a Name (line 110):
                # Getting the type of 's' (line 110)
                s_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 's')
                # Assigning a type to the variable 'best_score' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'best_score', s_504)
                
                # Assigning a Attribute to a Subscript (line 111):
                
                # Assigning a Attribute to a Subscript (line 111):
                # Getting the type of 'Cell' (line 111)
                Cell_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'Cell')
                # Obtaining the member 'Barren' of a type (line 111)
                Barren_506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 43), Cell_505, 'Barren')
                
                # Obtaining the type of the subscript
                # Getting the type of 'rowFilled' (line 111)
                rowFilled_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'rowFilled')
                # Getting the type of 'board' (line 111)
                board_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'board')
                # Obtaining the member '__getitem__' of a type (line 111)
                getitem___509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), board_508, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                subscript_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), getitem___509, rowFilled_507)
                
                # Getting the type of 'column' (line 111)
                column_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'column')
                # Storing an element on a container (line 111)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 16), subscript_call_result_510, (column_511, Barren_506))
                # SSA join for if statement (line 108)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Tuple (line 114):
            
            # Assigning a Call to a Name:
            
            # Call to ab_minimax(...): (line 114)
            # Processing the call arguments (line 114)
            
            # Getting the type of 'maximize_or_minimize' (line 114)
            maximize_or_minimize_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 41), 'maximize_or_minimize', False)
            # Applying the 'not' unary operator (line 114)
            result_not__514 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 37), 'not', maximize_or_minimize_513)
            
            
            
            # Getting the type of 'color' (line 115)
            color_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 52), 'color', False)
            # Getting the type of 'Cell' (line 115)
            Cell_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 61), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 115)
            Orange_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 61), Cell_516, 'Orange')
            # Applying the binary operator '==' (line 115)
            result_eq_518 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 52), '==', color_515, Orange_517)
            
            # Testing the type of an if expression (line 115)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 37), result_eq_518)
            # SSA begins for if expression (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'Cell' (line 115)
            Cell_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'Cell', False)
            # Obtaining the member 'Yellow' of a type (line 115)
            Yellow_520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 37), Cell_519, 'Yellow')
            # SSA branch for the else part of an if expression (line 115)
            module_type_store.open_ssa_branch('if expression else')
            # Getting the type of 'Cell' (line 115)
            Cell_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 78), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 115)
            Orange_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 78), Cell_521, 'Orange')
            # SSA join for if expression (line 115)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_523 = union_type.UnionType.add(Yellow_520, Orange_522)
            
            # Getting the type of 'depth' (line 116)
            depth_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'depth', False)
            int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 45), 'int')
            # Applying the binary operator '-' (line 116)
            result_sub_526 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 37), '-', depth_524, int_525)
            
            # Getting the type of 'board' (line 116)
            board_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 48), 'board', False)
            # Processing the call keyword arguments (line 114)
            kwargs_528 = {}
            # Getting the type of 'ab_minimax' (line 114)
            ab_minimax_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'ab_minimax', False)
            # Calling ab_minimax(args, kwargs) (line 114)
            ab_minimax_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 114, 26), ab_minimax_512, *[result_not__514, if_exp_523, result_sub_526, board_527], **kwargs_528)
            
            # Assigning a type to the variable 'call_assignment_1' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_1', ab_minimax_call_result_529)
            
            # Assigning a Call to a Name (line 114):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 114)
            call_assignment_1_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_531 = stypy_get_value_from_tuple(call_assignment_1_530, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_2' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_2', stypy_get_value_from_tuple_call_result_531)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'call_assignment_2' (line 114)
            call_assignment_2_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_2')
            # Assigning a type to the variable 'move' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'move', call_assignment_2_532)
            
            # Assigning a Call to a Name (line 114):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 114)
            call_assignment_1_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_534 = stypy_get_value_from_tuple(call_assignment_1_533, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_3' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_3', stypy_get_value_from_tuple_call_result_534)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'call_assignment_3' (line 114)
            call_assignment_3_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_3')
            # Assigning a type to the variable 'score' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'score', call_assignment_3_535)
            
            # Assigning a Attribute to a Subscript (line 117):
            
            # Assigning a Attribute to a Subscript (line 117):
            # Getting the type of 'Cell' (line 117)
            Cell_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 39), 'Cell')
            # Obtaining the member 'Barren' of a type (line 117)
            Barren_537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 39), Cell_536, 'Barren')
            
            # Obtaining the type of the subscript
            # Getting the type of 'rowFilled' (line 117)
            rowFilled_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'rowFilled')
            # Getting the type of 'board' (line 117)
            board_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'board')
            # Obtaining the member '__getitem__' of a type (line 117)
            getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), board_539, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 117)
            subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___540, rowFilled_538)
            
            # Getting the type of 'column' (line 117)
            column_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'column')
            # Storing an element on a container (line 117)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 12), subscript_call_result_541, (column_542, Barren_537))
            
            # Evaluating a boolean operation
            
            # Getting the type of 'depth' (line 118)
            depth_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'depth')
            # Getting the type of 'g_max_depth' (line 118)
            g_max_depth_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'g_max_depth')
            # Applying the binary operator '==' (line 118)
            result_eq_545 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '==', depth_543, g_max_depth_544)
            
            # Getting the type of 'g_debug' (line 118)
            g_debug_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'g_debug')
            # Applying the binary operator 'and' (line 118)
            result_and_keyword_547 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), 'and', result_eq_545, g_debug_546)
            
            # Testing if the type of an if condition is none (line 118)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 118, 12), result_and_keyword_547):
                pass
            else:
                
                # Testing the type of an if condition (line 118)
                if_condition_548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 12), result_and_keyword_547)
                # Assigning a type to the variable 'if_condition_548' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'if_condition_548', if_condition_548)
                # SSA begins for if statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA join for if statement (line 118)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'maximize_or_minimize' (line 120)
            maximize_or_minimize_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'maximize_or_minimize')
            # Testing if the type of an if condition is none (line 120)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 120, 12), maximize_or_minimize_549):
                
                # Getting the type of 'score' (line 125)
                score_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'score')
                # Getting the type of 'best_score' (line 125)
                best_score_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'best_score')
                # Applying the binary operator '<=' (line 125)
                result_le_559 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 19), '<=', score_557, best_score_558)
                
                # Testing if the type of an if condition is none (line 125)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 125)
                    if_condition_560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559)
                    # Assigning a type to the variable 'if_condition_560' (line 125)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'if_condition_560', if_condition_560)
                    # SSA begins for if statement (line 125)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 126):
                    
                    # Assigning a Name to a Name (line 126):
                    # Getting the type of 'score' (line 126)
                    score_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'score')
                    # Assigning a type to the variable 'best_score' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'best_score', score_561)
                    
                    # Assigning a Name to a Name (line 127):
                    
                    # Assigning a Name to a Name (line 127):
                    # Getting the type of 'column' (line 127)
                    column_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'column')
                    # Assigning a type to the variable 'bestMove' (line 127)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'bestMove', column_562)
                    # SSA join for if statement (line 125)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 120)
                if_condition_550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 12), maximize_or_minimize_549)
                # Assigning a type to the variable 'if_condition_550' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'if_condition_550', if_condition_550)
                # SSA begins for if statement (line 120)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'score' (line 121)
                score_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'score')
                # Getting the type of 'best_score' (line 121)
                best_score_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'best_score')
                # Applying the binary operator '>=' (line 121)
                result_ge_553 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), '>=', score_551, best_score_552)
                
                # Testing if the type of an if condition is none (line 121)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 16), result_ge_553):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 121)
                    if_condition_554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_ge_553)
                    # Assigning a type to the variable 'if_condition_554' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_554', if_condition_554)
                    # SSA begins for if statement (line 121)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 122):
                    
                    # Assigning a Name to a Name (line 122):
                    # Getting the type of 'score' (line 122)
                    score_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'score')
                    # Assigning a type to the variable 'best_score' (line 122)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'best_score', score_555)
                    
                    # Assigning a Name to a Name (line 123):
                    
                    # Assigning a Name to a Name (line 123):
                    # Getting the type of 'column' (line 123)
                    column_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'column')
                    # Assigning a type to the variable 'bestMove' (line 123)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'bestMove', column_556)
                    # SSA join for if statement (line 121)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 120)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'score' (line 125)
                score_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'score')
                # Getting the type of 'best_score' (line 125)
                best_score_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'best_score')
                # Applying the binary operator '<=' (line 125)
                result_le_559 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 19), '<=', score_557, best_score_558)
                
                # Testing if the type of an if condition is none (line 125)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 125)
                    if_condition_560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559)
                    # Assigning a type to the variable 'if_condition_560' (line 125)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'if_condition_560', if_condition_560)
                    # SSA begins for if statement (line 125)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 126):
                    
                    # Assigning a Name to a Name (line 126):
                    # Getting the type of 'score' (line 126)
                    score_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'score')
                    # Assigning a type to the variable 'best_score' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'best_score', score_561)
                    
                    # Assigning a Name to a Name (line 127):
                    
                    # Assigning a Name to a Name (line 127):
                    # Getting the type of 'column' (line 127)
                    column_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'column')
                    # Assigning a type to the variable 'bestMove' (line 127)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'bestMove', column_562)
                    # SSA join for if statement (line 125)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 120)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        # Getting the type of 'bestMove' (line 129)
        bestMove_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'bestMove')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 16), tuple_563, bestMove_564)
        # Adding element type (line 129)
        # Getting the type of 'best_score' (line 129)
        best_score_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'best_score')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 16), tuple_563, best_score_565)
        
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', tuple_563)
    else:
        
        # Testing the type of an if condition (line 96)
        if_condition_454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_eq_453)
        # Assigning a type to the variable 'if_condition_454' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_454', if_condition_454)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        int_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 16), tuple_455, int_456)
        # Adding element type (line 97)
        
        # Call to score_board(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'board' (line 97)
        board_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'board', False)
        # Processing the call keyword arguments (line 97)
        kwargs_459 = {}
        # Getting the type of 'score_board' (line 97)
        score_board_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'score_board', False)
        # Calling score_board(args, kwargs) (line 97)
        score_board_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 97, 20), score_board_457, *[board_458], **kwargs_459)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 16), tuple_455, score_board_call_result_460)
        
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', tuple_455)
        # SSA branch for the else part of an if statement (line 96)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a IfExp to a Name (line 99):
        
        # Assigning a IfExp to a Name (line 99):
        
        # Getting the type of 'maximize_or_minimize' (line 99)
        maximize_or_minimize_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'maximize_or_minimize')
        # Testing the type of an if expression (line 99)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 21), maximize_or_minimize_461)
        # SSA begins for if expression (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'int')
        # SSA branch for the else part of an if expression (line 99)
        module_type_store.open_ssa_branch('if expression else')
        int_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 60), 'int')
        # SSA join for if expression (line 99)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_464 = union_type.UnionType.add(int_462, int_463)
        
        # Assigning a type to the variable 'best_score' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'best_score', if_exp_464)
        
        # Assigning a Num to a Name (line 100):
        
        # Assigning a Num to a Name (line 100):
        int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'int')
        # Assigning a type to the variable 'bestMove' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'bestMove', int_465)
        
        
        # Call to xrange(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'WIDTH' (line 101)
        WIDTH_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'WIDTH', False)
        # Processing the call keyword arguments (line 101)
        kwargs_468 = {}
        # Getting the type of 'xrange' (line 101)
        xrange_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'xrange', False)
        # Calling xrange(args, kwargs) (line 101)
        xrange_call_result_469 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), xrange_466, *[WIDTH_467], **kwargs_468)
        
        # Assigning a type to the variable 'xrange_call_result_469' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'xrange_call_result_469', xrange_call_result_469)
        # Testing if the for loop is going to be iterated (line 101)
        # Testing the type of a for loop iterable (line 101)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 8), xrange_call_result_469)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 101, 8), xrange_call_result_469):
            # Getting the type of the for loop variable (line 101)
            for_loop_var_470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 8), xrange_call_result_469)
            # Assigning a type to the variable 'column' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'column', for_loop_var_470)
            # SSA begins for a for statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'column' (line 102)
            column_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'column')
            
            # Obtaining the type of the subscript
            int_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'int')
            # Getting the type of 'board' (line 102)
            board_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'board')
            # Obtaining the member '__getitem__' of a type (line 102)
            getitem___474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), board_473, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 102)
            subscript_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), getitem___474, int_472)
            
            # Obtaining the member '__getitem__' of a type (line 102)
            getitem___476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), subscript_call_result_475, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 102)
            subscript_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), getitem___476, column_471)
            
            # Getting the type of 'Cell' (line 102)
            Cell_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'Cell')
            # Obtaining the member 'Barren' of a type (line 102)
            Barren_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 35), Cell_478, 'Barren')
            # Applying the binary operator '!=' (line 102)
            result_ne_480 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 15), '!=', subscript_call_result_477, Barren_479)
            
            # Testing if the type of an if condition is none (line 102)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 12), result_ne_480):
                pass
            else:
                
                # Testing the type of an if condition (line 102)
                if_condition_481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 12), result_ne_480)
                # Assigning a type to the variable 'if_condition_481' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'if_condition_481', if_condition_481)
                # SSA begins for if statement (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 102)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 104):
            
            # Assigning a Call to a Name (line 104):
            
            # Call to drop_disk(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'board' (line 104)
            board_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'board', False)
            # Getting the type of 'column' (line 104)
            column_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 41), 'column', False)
            # Getting the type of 'color' (line 104)
            color_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 49), 'color', False)
            # Processing the call keyword arguments (line 104)
            kwargs_486 = {}
            # Getting the type of 'drop_disk' (line 104)
            drop_disk_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'drop_disk', False)
            # Calling drop_disk(args, kwargs) (line 104)
            drop_disk_call_result_487 = invoke(stypy.reporting.localization.Localization(__file__, 104, 24), drop_disk_482, *[board_483, column_484, color_485], **kwargs_486)
            
            # Assigning a type to the variable 'rowFilled' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'rowFilled', drop_disk_call_result_487)
            
            # Getting the type of 'rowFilled' (line 105)
            rowFilled_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'rowFilled')
            int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'int')
            # Applying the binary operator '==' (line 105)
            result_eq_490 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), '==', rowFilled_488, int_489)
            
            # Testing if the type of an if condition is none (line 105)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 105, 12), result_eq_490):
                pass
            else:
                
                # Testing the type of an if condition (line 105)
                if_condition_491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 12), result_eq_490)
                # Assigning a type to the variable 'if_condition_491' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'if_condition_491', if_condition_491)
                # SSA begins for if statement (line 105)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 105)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 107):
            
            # Assigning a Call to a Name (line 107):
            
            # Call to score_board(...): (line 107)
            # Processing the call arguments (line 107)
            # Getting the type of 'board' (line 107)
            board_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'board', False)
            # Processing the call keyword arguments (line 107)
            kwargs_494 = {}
            # Getting the type of 'score_board' (line 107)
            score_board_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'score_board', False)
            # Calling score_board(args, kwargs) (line 107)
            score_board_call_result_495 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), score_board_492, *[board_493], **kwargs_494)
            
            # Assigning a type to the variable 's' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 's', score_board_call_result_495)
            
            # Getting the type of 's' (line 108)
            s_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 's')
            
            # Getting the type of 'maximize_or_minimize' (line 108)
            maximize_or_minimize_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 36), 'maximize_or_minimize')
            # Testing the type of an if expression (line 108)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 21), maximize_or_minimize_497)
            # SSA begins for if expression (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'ORANGE_WINS' (line 108)
            ORANGE_WINS_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'ORANGE_WINS')
            # SSA branch for the else part of an if expression (line 108)
            module_type_store.open_ssa_branch('if expression else')
            # Getting the type of 'YELLOW_WINS' (line 108)
            YELLOW_WINS_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 62), 'YELLOW_WINS')
            # SSA join for if expression (line 108)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_500 = union_type.UnionType.add(ORANGE_WINS_498, YELLOW_WINS_499)
            
            # Applying the binary operator '==' (line 108)
            result_eq_501 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 15), '==', s_496, if_exp_500)
            
            # Testing if the type of an if condition is none (line 108)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 12), result_eq_501):
                pass
            else:
                
                # Testing the type of an if condition (line 108)
                if_condition_502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), result_eq_501)
                # Assigning a type to the variable 'if_condition_502' (line 108)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_502', if_condition_502)
                # SSA begins for if statement (line 108)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 109):
                
                # Assigning a Name to a Name (line 109):
                # Getting the type of 'column' (line 109)
                column_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'column')
                # Assigning a type to the variable 'bestMove' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'bestMove', column_503)
                
                # Assigning a Name to a Name (line 110):
                
                # Assigning a Name to a Name (line 110):
                # Getting the type of 's' (line 110)
                s_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 's')
                # Assigning a type to the variable 'best_score' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'best_score', s_504)
                
                # Assigning a Attribute to a Subscript (line 111):
                
                # Assigning a Attribute to a Subscript (line 111):
                # Getting the type of 'Cell' (line 111)
                Cell_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'Cell')
                # Obtaining the member 'Barren' of a type (line 111)
                Barren_506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 43), Cell_505, 'Barren')
                
                # Obtaining the type of the subscript
                # Getting the type of 'rowFilled' (line 111)
                rowFilled_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'rowFilled')
                # Getting the type of 'board' (line 111)
                board_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'board')
                # Obtaining the member '__getitem__' of a type (line 111)
                getitem___509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), board_508, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                subscript_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), getitem___509, rowFilled_507)
                
                # Getting the type of 'column' (line 111)
                column_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'column')
                # Storing an element on a container (line 111)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 16), subscript_call_result_510, (column_511, Barren_506))
                # SSA join for if statement (line 108)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Tuple (line 114):
            
            # Assigning a Call to a Name:
            
            # Call to ab_minimax(...): (line 114)
            # Processing the call arguments (line 114)
            
            # Getting the type of 'maximize_or_minimize' (line 114)
            maximize_or_minimize_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 41), 'maximize_or_minimize', False)
            # Applying the 'not' unary operator (line 114)
            result_not__514 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 37), 'not', maximize_or_minimize_513)
            
            
            
            # Getting the type of 'color' (line 115)
            color_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 52), 'color', False)
            # Getting the type of 'Cell' (line 115)
            Cell_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 61), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 115)
            Orange_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 61), Cell_516, 'Orange')
            # Applying the binary operator '==' (line 115)
            result_eq_518 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 52), '==', color_515, Orange_517)
            
            # Testing the type of an if expression (line 115)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 37), result_eq_518)
            # SSA begins for if expression (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'Cell' (line 115)
            Cell_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'Cell', False)
            # Obtaining the member 'Yellow' of a type (line 115)
            Yellow_520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 37), Cell_519, 'Yellow')
            # SSA branch for the else part of an if expression (line 115)
            module_type_store.open_ssa_branch('if expression else')
            # Getting the type of 'Cell' (line 115)
            Cell_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 78), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 115)
            Orange_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 78), Cell_521, 'Orange')
            # SSA join for if expression (line 115)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_523 = union_type.UnionType.add(Yellow_520, Orange_522)
            
            # Getting the type of 'depth' (line 116)
            depth_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'depth', False)
            int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 45), 'int')
            # Applying the binary operator '-' (line 116)
            result_sub_526 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 37), '-', depth_524, int_525)
            
            # Getting the type of 'board' (line 116)
            board_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 48), 'board', False)
            # Processing the call keyword arguments (line 114)
            kwargs_528 = {}
            # Getting the type of 'ab_minimax' (line 114)
            ab_minimax_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'ab_minimax', False)
            # Calling ab_minimax(args, kwargs) (line 114)
            ab_minimax_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 114, 26), ab_minimax_512, *[result_not__514, if_exp_523, result_sub_526, board_527], **kwargs_528)
            
            # Assigning a type to the variable 'call_assignment_1' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_1', ab_minimax_call_result_529)
            
            # Assigning a Call to a Name (line 114):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 114)
            call_assignment_1_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_531 = stypy_get_value_from_tuple(call_assignment_1_530, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_2' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_2', stypy_get_value_from_tuple_call_result_531)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'call_assignment_2' (line 114)
            call_assignment_2_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_2')
            # Assigning a type to the variable 'move' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'move', call_assignment_2_532)
            
            # Assigning a Call to a Name (line 114):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 114)
            call_assignment_1_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_534 = stypy_get_value_from_tuple(call_assignment_1_533, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_3' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_3', stypy_get_value_from_tuple_call_result_534)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'call_assignment_3' (line 114)
            call_assignment_3_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'call_assignment_3')
            # Assigning a type to the variable 'score' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'score', call_assignment_3_535)
            
            # Assigning a Attribute to a Subscript (line 117):
            
            # Assigning a Attribute to a Subscript (line 117):
            # Getting the type of 'Cell' (line 117)
            Cell_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 39), 'Cell')
            # Obtaining the member 'Barren' of a type (line 117)
            Barren_537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 39), Cell_536, 'Barren')
            
            # Obtaining the type of the subscript
            # Getting the type of 'rowFilled' (line 117)
            rowFilled_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'rowFilled')
            # Getting the type of 'board' (line 117)
            board_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'board')
            # Obtaining the member '__getitem__' of a type (line 117)
            getitem___540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), board_539, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 117)
            subscript_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___540, rowFilled_538)
            
            # Getting the type of 'column' (line 117)
            column_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'column')
            # Storing an element on a container (line 117)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 12), subscript_call_result_541, (column_542, Barren_537))
            
            # Evaluating a boolean operation
            
            # Getting the type of 'depth' (line 118)
            depth_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'depth')
            # Getting the type of 'g_max_depth' (line 118)
            g_max_depth_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'g_max_depth')
            # Applying the binary operator '==' (line 118)
            result_eq_545 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '==', depth_543, g_max_depth_544)
            
            # Getting the type of 'g_debug' (line 118)
            g_debug_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'g_debug')
            # Applying the binary operator 'and' (line 118)
            result_and_keyword_547 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), 'and', result_eq_545, g_debug_546)
            
            # Testing if the type of an if condition is none (line 118)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 118, 12), result_and_keyword_547):
                pass
            else:
                
                # Testing the type of an if condition (line 118)
                if_condition_548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 12), result_and_keyword_547)
                # Assigning a type to the variable 'if_condition_548' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'if_condition_548', if_condition_548)
                # SSA begins for if statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA join for if statement (line 118)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'maximize_or_minimize' (line 120)
            maximize_or_minimize_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'maximize_or_minimize')
            # Testing if the type of an if condition is none (line 120)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 120, 12), maximize_or_minimize_549):
                
                # Getting the type of 'score' (line 125)
                score_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'score')
                # Getting the type of 'best_score' (line 125)
                best_score_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'best_score')
                # Applying the binary operator '<=' (line 125)
                result_le_559 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 19), '<=', score_557, best_score_558)
                
                # Testing if the type of an if condition is none (line 125)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 125)
                    if_condition_560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559)
                    # Assigning a type to the variable 'if_condition_560' (line 125)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'if_condition_560', if_condition_560)
                    # SSA begins for if statement (line 125)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 126):
                    
                    # Assigning a Name to a Name (line 126):
                    # Getting the type of 'score' (line 126)
                    score_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'score')
                    # Assigning a type to the variable 'best_score' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'best_score', score_561)
                    
                    # Assigning a Name to a Name (line 127):
                    
                    # Assigning a Name to a Name (line 127):
                    # Getting the type of 'column' (line 127)
                    column_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'column')
                    # Assigning a type to the variable 'bestMove' (line 127)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'bestMove', column_562)
                    # SSA join for if statement (line 125)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 120)
                if_condition_550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 12), maximize_or_minimize_549)
                # Assigning a type to the variable 'if_condition_550' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'if_condition_550', if_condition_550)
                # SSA begins for if statement (line 120)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'score' (line 121)
                score_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'score')
                # Getting the type of 'best_score' (line 121)
                best_score_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'best_score')
                # Applying the binary operator '>=' (line 121)
                result_ge_553 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), '>=', score_551, best_score_552)
                
                # Testing if the type of an if condition is none (line 121)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 16), result_ge_553):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 121)
                    if_condition_554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_ge_553)
                    # Assigning a type to the variable 'if_condition_554' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_554', if_condition_554)
                    # SSA begins for if statement (line 121)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 122):
                    
                    # Assigning a Name to a Name (line 122):
                    # Getting the type of 'score' (line 122)
                    score_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'score')
                    # Assigning a type to the variable 'best_score' (line 122)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'best_score', score_555)
                    
                    # Assigning a Name to a Name (line 123):
                    
                    # Assigning a Name to a Name (line 123):
                    # Getting the type of 'column' (line 123)
                    column_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'column')
                    # Assigning a type to the variable 'bestMove' (line 123)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'bestMove', column_556)
                    # SSA join for if statement (line 121)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 120)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'score' (line 125)
                score_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'score')
                # Getting the type of 'best_score' (line 125)
                best_score_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'best_score')
                # Applying the binary operator '<=' (line 125)
                result_le_559 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 19), '<=', score_557, best_score_558)
                
                # Testing if the type of an if condition is none (line 125)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 125)
                    if_condition_560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_le_559)
                    # Assigning a type to the variable 'if_condition_560' (line 125)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'if_condition_560', if_condition_560)
                    # SSA begins for if statement (line 125)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 126):
                    
                    # Assigning a Name to a Name (line 126):
                    # Getting the type of 'score' (line 126)
                    score_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'score')
                    # Assigning a type to the variable 'best_score' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'best_score', score_561)
                    
                    # Assigning a Name to a Name (line 127):
                    
                    # Assigning a Name to a Name (line 127):
                    # Getting the type of 'column' (line 127)
                    column_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'column')
                    # Assigning a type to the variable 'bestMove' (line 127)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'bestMove', column_562)
                    # SSA join for if statement (line 125)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 120)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        # Getting the type of 'bestMove' (line 129)
        bestMove_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'bestMove')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 16), tuple_563, bestMove_564)
        # Adding element type (line 129)
        # Getting the type of 'best_score' (line 129)
        best_score_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'best_score')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 16), tuple_563, best_score_565)
        
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', tuple_563)
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'ab_minimax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ab_minimax' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_566)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ab_minimax'
    return stypy_return_type_566

# Assigning a type to the variable 'ab_minimax' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'ab_minimax', ab_minimax)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 132, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = ['args']
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', ['args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, ['args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    # Marking variables as global (line 133)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 133, 4), 'g_max_depth')
    
    # Assigning a Call to a Name (line 134):
    
    # Assigning a Call to a Name (line 134):
    
    # Call to load_board(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'args' (line 134)
    args_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'args', False)
    # Processing the call keyword arguments (line 134)
    kwargs_569 = {}
    # Getting the type of 'load_board' (line 134)
    load_board_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'load_board', False)
    # Calling load_board(args, kwargs) (line 134)
    load_board_call_result_570 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), load_board_567, *[args_568], **kwargs_569)
    
    # Assigning a type to the variable 'board' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'board', load_board_call_result_570)
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to score_board(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'board' (line 135)
    board_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'board', False)
    # Processing the call keyword arguments (line 135)
    kwargs_573 = {}
    # Getting the type of 'score_board' (line 135)
    score_board_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'score_board', False)
    # Calling score_board(args, kwargs) (line 135)
    score_board_call_result_574 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), score_board_571, *[board_572], **kwargs_573)
    
    # Assigning a type to the variable 'score_orig' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'score_orig', score_board_call_result_574)
    
    # Getting the type of 'score_orig' (line 137)
    score_orig_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'score_orig')
    # Getting the type of 'ORANGE_WINS' (line 137)
    ORANGE_WINS_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'ORANGE_WINS')
    # Applying the binary operator '==' (line 137)
    result_eq_577 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), '==', score_orig_575, ORANGE_WINS_576)
    
    # Testing if the type of an if condition is none (line 137)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 137, 4), result_eq_577):
        
        # Getting the type of 'score_orig' (line 140)
        score_orig_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'score_orig')
        # Getting the type of 'YELLOW_WINS' (line 140)
        YELLOW_WINS_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'YELLOW_WINS')
        # Applying the binary operator '==' (line 140)
        result_eq_582 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 9), '==', score_orig_580, YELLOW_WINS_581)
        
        # Testing if the type of an if condition is none (line 140)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 140, 9), result_eq_582):
            
            # Assigning a Call to a Tuple (line 144):
            
            # Assigning a Call to a Name:
            
            # Call to ab_minimax(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'True' (line 144)
            True_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'True', False)
            # Getting the type of 'Cell' (line 144)
            Cell_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 39), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 144)
            Orange_588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 39), Cell_587, 'Orange')
            # Getting the type of 'g_max_depth' (line 144)
            g_max_depth_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'g_max_depth', False)
            # Getting the type of 'board' (line 144)
            board_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 65), 'board', False)
            # Processing the call keyword arguments (line 144)
            kwargs_591 = {}
            # Getting the type of 'ab_minimax' (line 144)
            ab_minimax_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'ab_minimax', False)
            # Calling ab_minimax(args, kwargs) (line 144)
            ab_minimax_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 144, 22), ab_minimax_585, *[True_586, Orange_588, g_max_depth_589, board_590], **kwargs_591)
            
            # Assigning a type to the variable 'call_assignment_4' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', ab_minimax_call_result_592)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_594 = stypy_get_value_from_tuple(call_assignment_4_593, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_5' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5', stypy_get_value_from_tuple_call_result_594)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_5' (line 144)
            call_assignment_5_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5')
            # Assigning a type to the variable 'move' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'move', call_assignment_5_595)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_597 = stypy_get_value_from_tuple(call_assignment_4_596, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_597)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_6' (line 144)
            call_assignment_6_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6')
            # Assigning a type to the variable 'score' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'score', call_assignment_6_598)
            
            # Getting the type of 'move' (line 146)
            move_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'move')
            int_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
            # Applying the binary operator '!=' (line 146)
            result_ne_601 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), '!=', move_599, int_600)
            
            # Testing if the type of an if condition is none (line 146)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601):
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
            else:
                
                # Testing the type of an if condition (line 146)
                if_condition_602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601)
                # Assigning a type to the variable 'if_condition_602' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_602', if_condition_602)
                # SSA begins for if statement (line 146)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to drop_disk(...): (line 148)
                # Processing the call arguments (line 148)
                # Getting the type of 'board' (line 148)
                board_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'board', False)
                # Getting the type of 'move' (line 148)
                move_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'move', False)
                # Getting the type of 'Cell' (line 148)
                Cell_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'Cell', False)
                # Obtaining the member 'Orange' of a type (line 148)
                Orange_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 35), Cell_606, 'Orange')
                # Processing the call keyword arguments (line 148)
                kwargs_608 = {}
                # Getting the type of 'drop_disk' (line 148)
                drop_disk_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'drop_disk', False)
                # Calling drop_disk(args, kwargs) (line 148)
                drop_disk_call_result_609 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), drop_disk_603, *[board_604, move_605, Orange_607], **kwargs_608)
                
                
                # Assigning a Call to a Name (line 149):
                
                # Assigning a Call to a Name (line 149):
                
                # Call to score_board(...): (line 149)
                # Processing the call arguments (line 149)
                # Getting the type of 'board' (line 149)
                board_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'board', False)
                # Processing the call keyword arguments (line 149)
                kwargs_612 = {}
                # Getting the type of 'score_board' (line 149)
                score_board_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'score_board', False)
                # Calling score_board(args, kwargs) (line 149)
                score_board_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), score_board_610, *[board_611], **kwargs_612)
                
                # Assigning a type to the variable 'score_orig' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'score_orig', score_board_call_result_613)
                
                # Getting the type of 'score_orig' (line 150)
                score_orig_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'score_orig')
                # Getting the type of 'ORANGE_WINS' (line 150)
                ORANGE_WINS_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'ORANGE_WINS')
                # Applying the binary operator '==' (line 150)
                result_eq_616 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '==', score_orig_614, ORANGE_WINS_615)
                
                # Testing if the type of an if condition is none (line 150)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616):
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 150)
                    if_condition_617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616)
                    # Assigning a type to the variable 'if_condition_617' (line 150)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_617', if_condition_617)
                    # SSA begins for if statement (line 150)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 152)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'stypy_return_type', int_618)
                    # SSA branch for the else part of an if statement (line 150)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 150)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 146)
                module_type_store.open_ssa_branch('else')
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
                # SSA join for if statement (line 146)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 140)
            if_condition_583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 9), result_eq_582)
            # Assigning a type to the variable 'if_condition_583' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'if_condition_583', if_condition_583)
            # SSA begins for if statement (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', int_584)
            # SSA branch for the else part of an if statement (line 140)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 144):
            
            # Assigning a Call to a Name:
            
            # Call to ab_minimax(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'True' (line 144)
            True_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'True', False)
            # Getting the type of 'Cell' (line 144)
            Cell_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 39), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 144)
            Orange_588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 39), Cell_587, 'Orange')
            # Getting the type of 'g_max_depth' (line 144)
            g_max_depth_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'g_max_depth', False)
            # Getting the type of 'board' (line 144)
            board_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 65), 'board', False)
            # Processing the call keyword arguments (line 144)
            kwargs_591 = {}
            # Getting the type of 'ab_minimax' (line 144)
            ab_minimax_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'ab_minimax', False)
            # Calling ab_minimax(args, kwargs) (line 144)
            ab_minimax_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 144, 22), ab_minimax_585, *[True_586, Orange_588, g_max_depth_589, board_590], **kwargs_591)
            
            # Assigning a type to the variable 'call_assignment_4' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', ab_minimax_call_result_592)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_594 = stypy_get_value_from_tuple(call_assignment_4_593, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_5' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5', stypy_get_value_from_tuple_call_result_594)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_5' (line 144)
            call_assignment_5_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5')
            # Assigning a type to the variable 'move' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'move', call_assignment_5_595)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_597 = stypy_get_value_from_tuple(call_assignment_4_596, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_597)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_6' (line 144)
            call_assignment_6_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6')
            # Assigning a type to the variable 'score' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'score', call_assignment_6_598)
            
            # Getting the type of 'move' (line 146)
            move_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'move')
            int_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
            # Applying the binary operator '!=' (line 146)
            result_ne_601 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), '!=', move_599, int_600)
            
            # Testing if the type of an if condition is none (line 146)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601):
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
            else:
                
                # Testing the type of an if condition (line 146)
                if_condition_602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601)
                # Assigning a type to the variable 'if_condition_602' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_602', if_condition_602)
                # SSA begins for if statement (line 146)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to drop_disk(...): (line 148)
                # Processing the call arguments (line 148)
                # Getting the type of 'board' (line 148)
                board_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'board', False)
                # Getting the type of 'move' (line 148)
                move_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'move', False)
                # Getting the type of 'Cell' (line 148)
                Cell_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'Cell', False)
                # Obtaining the member 'Orange' of a type (line 148)
                Orange_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 35), Cell_606, 'Orange')
                # Processing the call keyword arguments (line 148)
                kwargs_608 = {}
                # Getting the type of 'drop_disk' (line 148)
                drop_disk_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'drop_disk', False)
                # Calling drop_disk(args, kwargs) (line 148)
                drop_disk_call_result_609 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), drop_disk_603, *[board_604, move_605, Orange_607], **kwargs_608)
                
                
                # Assigning a Call to a Name (line 149):
                
                # Assigning a Call to a Name (line 149):
                
                # Call to score_board(...): (line 149)
                # Processing the call arguments (line 149)
                # Getting the type of 'board' (line 149)
                board_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'board', False)
                # Processing the call keyword arguments (line 149)
                kwargs_612 = {}
                # Getting the type of 'score_board' (line 149)
                score_board_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'score_board', False)
                # Calling score_board(args, kwargs) (line 149)
                score_board_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), score_board_610, *[board_611], **kwargs_612)
                
                # Assigning a type to the variable 'score_orig' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'score_orig', score_board_call_result_613)
                
                # Getting the type of 'score_orig' (line 150)
                score_orig_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'score_orig')
                # Getting the type of 'ORANGE_WINS' (line 150)
                ORANGE_WINS_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'ORANGE_WINS')
                # Applying the binary operator '==' (line 150)
                result_eq_616 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '==', score_orig_614, ORANGE_WINS_615)
                
                # Testing if the type of an if condition is none (line 150)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616):
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 150)
                    if_condition_617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616)
                    # Assigning a type to the variable 'if_condition_617' (line 150)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_617', if_condition_617)
                    # SSA begins for if statement (line 150)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 152)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'stypy_return_type', int_618)
                    # SSA branch for the else part of an if statement (line 150)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 150)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 146)
                module_type_store.open_ssa_branch('else')
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
                # SSA join for if statement (line 146)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 140)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 137)
        if_condition_578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), result_eq_577)
        # Assigning a type to the variable 'if_condition_578' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_578', if_condition_578)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', int_579)
        # SSA branch for the else part of an if statement (line 137)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'score_orig' (line 140)
        score_orig_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'score_orig')
        # Getting the type of 'YELLOW_WINS' (line 140)
        YELLOW_WINS_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'YELLOW_WINS')
        # Applying the binary operator '==' (line 140)
        result_eq_582 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 9), '==', score_orig_580, YELLOW_WINS_581)
        
        # Testing if the type of an if condition is none (line 140)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 140, 9), result_eq_582):
            
            # Assigning a Call to a Tuple (line 144):
            
            # Assigning a Call to a Name:
            
            # Call to ab_minimax(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'True' (line 144)
            True_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'True', False)
            # Getting the type of 'Cell' (line 144)
            Cell_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 39), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 144)
            Orange_588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 39), Cell_587, 'Orange')
            # Getting the type of 'g_max_depth' (line 144)
            g_max_depth_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'g_max_depth', False)
            # Getting the type of 'board' (line 144)
            board_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 65), 'board', False)
            # Processing the call keyword arguments (line 144)
            kwargs_591 = {}
            # Getting the type of 'ab_minimax' (line 144)
            ab_minimax_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'ab_minimax', False)
            # Calling ab_minimax(args, kwargs) (line 144)
            ab_minimax_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 144, 22), ab_minimax_585, *[True_586, Orange_588, g_max_depth_589, board_590], **kwargs_591)
            
            # Assigning a type to the variable 'call_assignment_4' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', ab_minimax_call_result_592)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_594 = stypy_get_value_from_tuple(call_assignment_4_593, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_5' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5', stypy_get_value_from_tuple_call_result_594)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_5' (line 144)
            call_assignment_5_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5')
            # Assigning a type to the variable 'move' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'move', call_assignment_5_595)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_597 = stypy_get_value_from_tuple(call_assignment_4_596, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_597)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_6' (line 144)
            call_assignment_6_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6')
            # Assigning a type to the variable 'score' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'score', call_assignment_6_598)
            
            # Getting the type of 'move' (line 146)
            move_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'move')
            int_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
            # Applying the binary operator '!=' (line 146)
            result_ne_601 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), '!=', move_599, int_600)
            
            # Testing if the type of an if condition is none (line 146)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601):
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
            else:
                
                # Testing the type of an if condition (line 146)
                if_condition_602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601)
                # Assigning a type to the variable 'if_condition_602' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_602', if_condition_602)
                # SSA begins for if statement (line 146)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to drop_disk(...): (line 148)
                # Processing the call arguments (line 148)
                # Getting the type of 'board' (line 148)
                board_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'board', False)
                # Getting the type of 'move' (line 148)
                move_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'move', False)
                # Getting the type of 'Cell' (line 148)
                Cell_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'Cell', False)
                # Obtaining the member 'Orange' of a type (line 148)
                Orange_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 35), Cell_606, 'Orange')
                # Processing the call keyword arguments (line 148)
                kwargs_608 = {}
                # Getting the type of 'drop_disk' (line 148)
                drop_disk_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'drop_disk', False)
                # Calling drop_disk(args, kwargs) (line 148)
                drop_disk_call_result_609 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), drop_disk_603, *[board_604, move_605, Orange_607], **kwargs_608)
                
                
                # Assigning a Call to a Name (line 149):
                
                # Assigning a Call to a Name (line 149):
                
                # Call to score_board(...): (line 149)
                # Processing the call arguments (line 149)
                # Getting the type of 'board' (line 149)
                board_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'board', False)
                # Processing the call keyword arguments (line 149)
                kwargs_612 = {}
                # Getting the type of 'score_board' (line 149)
                score_board_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'score_board', False)
                # Calling score_board(args, kwargs) (line 149)
                score_board_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), score_board_610, *[board_611], **kwargs_612)
                
                # Assigning a type to the variable 'score_orig' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'score_orig', score_board_call_result_613)
                
                # Getting the type of 'score_orig' (line 150)
                score_orig_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'score_orig')
                # Getting the type of 'ORANGE_WINS' (line 150)
                ORANGE_WINS_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'ORANGE_WINS')
                # Applying the binary operator '==' (line 150)
                result_eq_616 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '==', score_orig_614, ORANGE_WINS_615)
                
                # Testing if the type of an if condition is none (line 150)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616):
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 150)
                    if_condition_617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616)
                    # Assigning a type to the variable 'if_condition_617' (line 150)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_617', if_condition_617)
                    # SSA begins for if statement (line 150)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 152)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'stypy_return_type', int_618)
                    # SSA branch for the else part of an if statement (line 150)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 150)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 146)
                module_type_store.open_ssa_branch('else')
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
                # SSA join for if statement (line 146)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 140)
            if_condition_583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 9), result_eq_582)
            # Assigning a type to the variable 'if_condition_583' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'if_condition_583', if_condition_583)
            # SSA begins for if statement (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', int_584)
            # SSA branch for the else part of an if statement (line 140)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 144):
            
            # Assigning a Call to a Name:
            
            # Call to ab_minimax(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'True' (line 144)
            True_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'True', False)
            # Getting the type of 'Cell' (line 144)
            Cell_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 39), 'Cell', False)
            # Obtaining the member 'Orange' of a type (line 144)
            Orange_588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 39), Cell_587, 'Orange')
            # Getting the type of 'g_max_depth' (line 144)
            g_max_depth_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'g_max_depth', False)
            # Getting the type of 'board' (line 144)
            board_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 65), 'board', False)
            # Processing the call keyword arguments (line 144)
            kwargs_591 = {}
            # Getting the type of 'ab_minimax' (line 144)
            ab_minimax_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'ab_minimax', False)
            # Calling ab_minimax(args, kwargs) (line 144)
            ab_minimax_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 144, 22), ab_minimax_585, *[True_586, Orange_588, g_max_depth_589, board_590], **kwargs_591)
            
            # Assigning a type to the variable 'call_assignment_4' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', ab_minimax_call_result_592)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_594 = stypy_get_value_from_tuple(call_assignment_4_593, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_5' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5', stypy_get_value_from_tuple_call_result_594)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_5' (line 144)
            call_assignment_5_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_5')
            # Assigning a type to the variable 'move' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'move', call_assignment_5_595)
            
            # Assigning a Call to a Name (line 144):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4' (line 144)
            call_assignment_4_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_4', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_597 = stypy_get_value_from_tuple(call_assignment_4_596, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_597)
            
            # Assigning a Name to a Name (line 144):
            # Getting the type of 'call_assignment_6' (line 144)
            call_assignment_6_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_6')
            # Assigning a type to the variable 'score' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'score', call_assignment_6_598)
            
            # Getting the type of 'move' (line 146)
            move_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'move')
            int_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
            # Applying the binary operator '!=' (line 146)
            result_ne_601 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), '!=', move_599, int_600)
            
            # Testing if the type of an if condition is none (line 146)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601):
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
            else:
                
                # Testing the type of an if condition (line 146)
                if_condition_602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_601)
                # Assigning a type to the variable 'if_condition_602' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_602', if_condition_602)
                # SSA begins for if statement (line 146)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to drop_disk(...): (line 148)
                # Processing the call arguments (line 148)
                # Getting the type of 'board' (line 148)
                board_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'board', False)
                # Getting the type of 'move' (line 148)
                move_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'move', False)
                # Getting the type of 'Cell' (line 148)
                Cell_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'Cell', False)
                # Obtaining the member 'Orange' of a type (line 148)
                Orange_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 35), Cell_606, 'Orange')
                # Processing the call keyword arguments (line 148)
                kwargs_608 = {}
                # Getting the type of 'drop_disk' (line 148)
                drop_disk_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'drop_disk', False)
                # Calling drop_disk(args, kwargs) (line 148)
                drop_disk_call_result_609 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), drop_disk_603, *[board_604, move_605, Orange_607], **kwargs_608)
                
                
                # Assigning a Call to a Name (line 149):
                
                # Assigning a Call to a Name (line 149):
                
                # Call to score_board(...): (line 149)
                # Processing the call arguments (line 149)
                # Getting the type of 'board' (line 149)
                board_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'board', False)
                # Processing the call keyword arguments (line 149)
                kwargs_612 = {}
                # Getting the type of 'score_board' (line 149)
                score_board_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'score_board', False)
                # Calling score_board(args, kwargs) (line 149)
                score_board_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), score_board_610, *[board_611], **kwargs_612)
                
                # Assigning a type to the variable 'score_orig' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'score_orig', score_board_call_result_613)
                
                # Getting the type of 'score_orig' (line 150)
                score_orig_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'score_orig')
                # Getting the type of 'ORANGE_WINS' (line 150)
                ORANGE_WINS_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'ORANGE_WINS')
                # Applying the binary operator '==' (line 150)
                result_eq_616 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '==', score_orig_614, ORANGE_WINS_615)
                
                # Testing if the type of an if condition is none (line 150)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616):
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 150)
                    if_condition_617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_eq_616)
                    # Assigning a type to the variable 'if_condition_617' (line 150)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_617', if_condition_617)
                    # SSA begins for if statement (line 150)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 152)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'stypy_return_type', int_618)
                    # SSA branch for the else part of an if statement (line 150)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'score_orig' (line 153)
                    score_orig_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'score_orig')
                    # Getting the type of 'YELLOW_WINS' (line 153)
                    YELLOW_WINS_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'YELLOW_WINS')
                    # Applying the binary operator '==' (line 153)
                    result_eq_621 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '==', score_orig_619, YELLOW_WINS_620)
                    
                    # Testing if the type of an if condition is none (line 153)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621):
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                    else:
                        
                        # Testing the type of an if condition (line 153)
                        if_condition_622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 17), result_eq_621)
                        # Assigning a type to the variable 'if_condition_622' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'if_condition_622', if_condition_622)
                        # SSA begins for if statement (line 153)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 155)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', int_623)
                        # SSA branch for the else part of an if statement (line 153)
                        module_type_store.open_ssa_branch('else')
                        int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
                        # Assigning a type to the variable 'stypy_return_type' (line 157)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', int_624)
                        # SSA join for if statement (line 153)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 150)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 146)
                module_type_store.open_ssa_branch('else')
                int_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', int_625)
                # SSA join for if statement (line 146)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 140)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_626

# Assigning a type to the variable 'main' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 163, 0, False)
    
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

    
    # Call to main(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining an instance of the builtin type 'list' (line 164)
    list_628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 164)
    # Adding element type (line 164)
    str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 10), 'str', 'score4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 9), list_628, str_629)
    # Adding element type (line 164)
    str_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'str', 'o53')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 9), list_628, str_630)
    # Adding element type (line 164)
    str_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'str', 'y43')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 9), list_628, str_631)
    
    # Processing the call keyword arguments (line 164)
    kwargs_632 = {}
    # Getting the type of 'main' (line 164)
    main_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'main', False)
    # Calling main(args, kwargs) (line 164)
    main_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 164, 4), main_627, *[list_628], **kwargs_632)
    
    # Getting the type of 'True' (line 165)
    True_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type', True_634)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_635

# Assigning a type to the variable 'run' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'run', run)

# Call to run(...): (line 168)
# Processing the call keyword arguments (line 168)
kwargs_637 = {}
# Getting the type of 'run' (line 168)
run_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'run', False)
# Calling run(args, kwargs) (line 168)
run_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 168, 0), run_636, *[], **kwargs_637)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
